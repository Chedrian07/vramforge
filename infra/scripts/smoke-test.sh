#!/bin/sh
# Smoke test a running VRAMForge stack through the proxy (used by CI and docs/deployment.md).
#
#   docker compose up -d --build --wait
#   infra/scripts/smoke-test.sh [base-url]        # default http://127.0.0.1:${VRAMFORGE_PORT:-8080}
#
# Checks: proxy liveness, API health through /api with the database, Redis and a worker heartbeat
# all "ok", web root, and the proxy security headers.
# Needs only POSIX sh and curl. Exit code 0 means every check passed.
set -eu

BASE_URL="${1:-http://127.0.0.1:${VRAMFORGE_PORT:-8080}}"
BASE_URL="${BASE_URL%/}"
# /api/v1/health always answers 200 (it is the liveness probe); its top-level "status" is "ok"
# only when db, redis and worker are all ok. Poll up to this many seconds for that state.
HEALTH_WAIT_S="${SMOKE_HEALTH_WAIT_S:-60}"
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

waited=0
while :; do
	result="$(fetch /api/v1/health)"
	[ "${result%% *}" = "200" ] || fail "GET /api/v1/health: expected 200, got $result"
	case "$result" in
	*application/json*) ;;
	*) fail "GET /api/v1/health: expected application/json, got $result" ;;
	esac
	grep -q '"status"' "$TMP/body" || fail "GET /api/v1/health: no status field: $(cat "$TMP/body")"
	grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$TMP/body" && break
	[ "$waited" -lt "$HEALTH_WAIT_S" ] ||
		fail "GET /api/v1/health: not ok after ${HEALTH_WAIT_S}s: $(cat "$TMP/body")"
	sleep 2
	waited=$((waited + 2))
done
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
