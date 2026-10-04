"""Access control for `/api/v1` (plan.md §18, docs/architecture.md §6).

A pure ASGI middleware (safe for SSE streaming) that, in order:

1. enforces the optional access token (`vf_access` cookie or `Authorization: Bearer`) on every
   route except `/health` and `/session`;
2. requires `X-VramForge-Request: 1` on state-changing methods (CSRF; SameSite=Strict cookies
   are a second layer);
3. caps request bodies (uploads have their own streaming cap);
4. issues the anonymous `vf_owner` cookie (256-bit random, httpOnly, SameSite=Strict).

The owner id stored in the database is the sha256 of the cookie value.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Any

from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import ErrorCode, Issue, Stage

from .errors import BodyTooLarge, error_response
from .settings import Settings

API_PREFIX = "/api/v1"
OWNER_COOKIE = "vf_owner"
ACCESS_COOKIE = "vf_access"
CSRF_HEADER = "x-vramforge-request"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
OWNER_COOKIE_MAX_AGE = 400 * 86_400  # browsers cap cookie lifetime at ~400 days

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # token_urlsafe(32)
_AUTH_EXEMPT = (f"{API_PREFIX}/health", f"{API_PREFIX}/session")
_OWNER_EXEMPT = (f"{API_PREFIX}/health",)
UPLOAD_PATH = f"{API_PREFIX}/uploads"
# Multipart framing around the file (headers, boundaries) on top of the upload cap.
MULTIPART_OVERHEAD = 64 * 1024


def new_owner_token() -> str:
    return secrets.token_urlsafe(32)


def owner_key_for(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def access_cookie_value(access_token: str) -> str:
    """Derived cookie value, so the raw token is not stored in the browser."""
    return hmac.new(access_token.encode(), b"vramforge-access-v1", hashlib.sha256).hexdigest()


def _digest_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(), hashlib.sha256(b.encode()).digest()
    )


def token_matches(candidate: str | None, access_token: str) -> bool:
    return bool(candidate) and _digest_equal(candidate or "", access_token)


def is_authenticated(headers: Headers, cookies: dict[str, str], access_token: str) -> bool:
    cookie = cookies.get(ACCESS_COOKIE)
    if cookie and _digest_equal(cookie, access_cookie_value(access_token)):
        return True
    auth = headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    return scheme.lower() == "bearer" and token_matches(value.strip(), access_token)


def cookie_header(
    name: str, value: str, *, secure: bool, max_age: int | None, path: str = "/"
) -> str:
    parts = [f"{name}={value}", f"Path={path}", "HttpOnly", "SameSite=Strict"]
    if max_age is not None:
        parts.append(f"Max-Age={max_age}")
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def parse_cookies(headers: Headers) -> dict[str, str]:
    """Browser-like cookie parsing. `http.cookies.SimpleCookie` drops every cookie after one
    malformed value, and cookies are not port-scoped: another app on the same host could
    otherwise make this service mint a new owner on every request."""
    return cookie_parser("; ".join(headers.getlist("cookie")))


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(API_PREFIX):
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        method: str = scope["method"].upper()
        headers = Headers(scope=scope)
        cookies = parse_cookies(headers)

        access_token = self.settings.access_token_value()
        if (
            access_token
            and not path.startswith(_AUTH_EXEMPT)
            and not is_authenticated(headers, cookies, access_token)
        ):
            issue = make_issue(ErrorCode.UNAUTHORIZED, "접근 토큰이 필요합니다.", stage=Stage.API)
            await error_response(401, issue)(scope, receive, send)
            return

        if method in UNSAFE_METHODS and headers.get(CSRF_HEADER) != "1":
            issue = make_issue(
                ErrorCode.FORBIDDEN,
                "요청 확인 헤더가 없어 거부했습니다. 페이지를 새로고침한 뒤 다시 시도하세요.",
                stage=Stage.API,
                reason="csrf_header_missing",
            )
            await error_response(403, issue)(scope, receive, send)
            return

        limit = (
            self.settings.max_upload_bytes + MULTIPART_OVERHEAD
            if path == UPLOAD_PATH
            else self.settings.max_json_body_bytes
        )
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await error_response(413, self._too_large_issue(path))(scope, receive, send)
            return

        new_cookie: str | None = None
        if not path.startswith(_OWNER_EXEMPT):
            token = cookies.get(OWNER_COOKIE, "")
            if not _TOKEN_RE.match(token):
                token = new_owner_token()
                new_cookie = cookie_header(
                    OWNER_COOKIE,
                    token,
                    secure=self.settings.cookie_secure,
                    max_age=OWNER_COOKIE_MAX_AGE,
                )
            state: dict[str, Any] = scope.setdefault("state", {})
            state["vf_owner_key"] = owner_key_for(token)

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(self._too_large_issue(path))
            return message

        started = False

        async def send_with_cookie(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                if new_cookie is not None:
                    raw = list(message.get("headers", []))
                    raw.append((b"set-cookie", new_cookie.encode("latin-1")))
                    message = {**message, "headers": raw}
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_cookie)
        except BodyTooLarge as exc:  # normally answered by the app's handler; fallback only
            if not started:
                await error_response(413, exc.issue)(scope, receive, send)

    def _too_large_issue(self, path: str) -> Issue:
        if path == UPLOAD_PATH:
            return make_issue(
                ErrorCode.UPLOAD_TOO_LARGE,
                "업로드 파일이 허용 크기를 넘었습니다.",
                stage=Stage.API,
                max_bytes=self.settings.max_upload_bytes,
            )
        return make_issue(ErrorCode.INVALID_REQUEST, "요청 본문이 너무 큽니다.", stage=Stage.API)


__all__ = [
    "ACCESS_COOKIE",
    "API_PREFIX",
    "CSRF_HEADER",
    "OWNER_COOKIE",
    "SecurityMiddleware",
    "access_cookie_value",
    "cookie_header",
    "is_authenticated",
    "new_owner_token",
    "owner_key_for",
    "parse_cookies",
    "token_matches",
]
