#!/bin/sh
# Smoke test a running VRAMForge stack through the proxy (used by CI and docs/deployment.md).
#
#   docker compose up -d --build --wait
#   infra/scripts/smoke-test.sh [base-url]        # default http://127.0.0.1:${VRAMFORGE_PORT:-8080}
#
# Checks: proxy liveness, API health through /api, web root, and the proxy security headers.
# Needs only POSIX sh and curl. Exit code 0 means every check passed.
set -eu

BASE_URL="${1:-http://127.0.0.1:${VRAMFORGE_PORT:-8080}}"
BASE_URL="${BASE_URL%/}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

fail() {
	echo "smoke: FAIL $*" >&2
	exit 1
}

# fetch <path>: writes body/headers to $TMP and prints "<status> <content-type>"
fetch() {
	curl -sS --max-time 20 -o "$TMP/body" -D "$TMP/headers" \
		-w '%{http_code} %{content_type}' "$BASE_URL$1" || fail "GET $1: connection failed"
}

result="$(fetch /healthz)"
[ "${result%% *}" = "200" ] || fail "GET /healthz: expected 200, got $result"
echo "smoke: ok   GET /healthz -> $result"

result="$(fetch /api/v1/health)"
[ "${result%% *}" = "200" ] || fail "GET /api/v1/health: expected 200, got $result"
case "$result" in
*application/json*) ;;
*) fail "GET /api/v1/health: expected application/json, got $result" ;;
esac
grep -q '"status"' "$TMP/body" || fail "GET /api/v1/health: body has no status field"
echo "smoke: ok   GET /api/v1/health -> $result $(cat "$TMP/body")"

result="$(fetch /)"
[ "${result%% *}" = "200" ] || fail "GET /: expected 200, got $result"
case "$result" in
*text/html*) ;;
*) fail "GET /: expected text/html, got $result" ;;
esac
for header in X-Content-Type-Options X-Frame-Options Referrer-Policy Content-Security-Policy; do
	grep -qi "^$header:" "$TMP/headers" || fail "GET /: missing $header header"
done
echo "smoke: ok   GET / -> $result (security headers present)"

echo "smoke: all checks passed for $BASE_URL"
