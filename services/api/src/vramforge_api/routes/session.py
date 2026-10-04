"""Optional access-token session (`VRAMFORGE_ACCESS_TOKEN`).

EventSource cannot send headers, so the browser exchanges the token for an httpOnly, SameSite=Strict
`vf_access` cookie (docs/research/stack-compat.md W9). API clients may use `Authorization: Bearer`.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, Request, Response

from vramforge_estimator.schemas import ErrorCode

from ..deps import StateDep
from ..errors import api_error
from ..http_models import SessionLogin, SessionStatus
from ..security import (
    ACCESS_COOKIE,
    access_cookie_value,
    cookie_header,
    is_authenticated,
    parse_cookies,
    token_matches,
)
from ._contract import ERROR_RESPONSES

router = APIRouter(prefix="/session", tags=["session"], responses=ERROR_RESPONSES)

FAILED_LOGIN_DELAY_S = 0.5


@router.get("", response_model=SessionStatus)
def session_status(request: Request, state: StateDep) -> SessionStatus:
    token = state.settings.access_token_value()
    if token is None:
        return SessionStatus(auth_required=False, authenticated=True)
    ok = is_authenticated(request.headers, parse_cookies(request.headers), token)
    return SessionStatus(auth_required=True, authenticated=ok)


@router.post("", response_model=SessionStatus)
async def login(body: SessionLogin, response: Response, state: StateDep) -> SessionStatus:
    token = state.settings.access_token_value()
    if token is None:
        return SessionStatus(auth_required=False, authenticated=True)
    if not token_matches(body.token, token):
        await anyio.sleep(FAILED_LOGIN_DELAY_S)  # slows down guessing
        raise api_error(401, ErrorCode.UNAUTHORIZED, "접근 토큰이 올바르지 않습니다.")
    response.headers.append(
        "set-cookie",
        cookie_header(
            ACCESS_COOKIE,
            access_cookie_value(token),
            secure=state.settings.cookie_secure,
            max_age=30 * 86_400,
        ),
    )
    return SessionStatus(auth_required=True, authenticated=True)


@router.delete("", response_model=SessionStatus)
def logout(response: Response, state: StateDep) -> SessionStatus:
    response.headers.append(
        "set-cookie",
        cookie_header(ACCESS_COOKIE, "", secure=state.settings.cookie_secure, max_age=0),
    )
    required = state.settings.access_token_value() is not None
    return SessionStatus(auth_required=required, authenticated=not required)
