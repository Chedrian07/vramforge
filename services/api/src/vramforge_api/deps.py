"""FastAPI dependencies: app state, DB session and the request's owner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from vramforge_estimator.schemas import ErrorCode

from .errors import api_error
from .state import AppState


def app_state(request: Request) -> AppState:
    state: AppState = request.app.state.vf
    return state


def db_session(request: Request) -> Iterator[Session]:
    session = app_state(request).sessions()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def owner_key(request: Request) -> str:
    """Owner id (sha256 of the vf_owner cookie) set by `SecurityMiddleware`."""
    key = request.scope.get("state", {}).get("vf_owner_key")
    if not key:
        raise api_error(403, ErrorCode.FORBIDDEN, "소유자 정보를 확인할 수 없습니다.")
    return str(key)


StateDep = Annotated[AppState, Depends(app_state)]
DbDep = Annotated[Session, Depends(db_session)]
OwnerDep = Annotated[str, Depends(owner_key)]

__all__ = ["DbDep", "OwnerDep", "StateDep", "app_state", "db_session", "owner_key"]
