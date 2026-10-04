"""Health: always HTTP 200 (compose liveness); component states say what is degraded.

Dependency checks run concurrently with a short budget so the probe answers quickly even when
PostgreSQL or Redis is unreachable. `hf_token` says how the server's Hugging Face token is used
(`not_configured`, `configured_not_shared`, `shared`), never its value; with an access token set
it is shown to authenticated callers only (the probe itself needs no token).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from fastapi import APIRouter, Request
from sqlalchemy import text

from vramforge_api import __version__
from vramforge_estimator.schemas import HealthResponse

from ..deps import StateDep
from ..jobs import worker_heartbeat_state
from ..security import is_authenticated, parse_cookies
from ..state import AppState

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

CHECK_BUDGET_S = 2.5
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health")


def _db(state: AppState) -> str:
    with state.engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return "ok"


def _redis_and_worker(state: AppState) -> tuple[str, str]:
    state.redis.ping()
    return "ok", worker_heartbeat_state(state.redis)


def _may_see_configuration(request: Request, state: AppState) -> bool:
    token = state.settings.access_token_value()
    return token is None or is_authenticated(request.headers, parse_cookies(request.headers), token)


@router.get("/health", response_model=HealthResponse)
def health(request: Request, state: StateDep) -> HealthResponse:
    components = {"api": "ok", "gpu_worker": "not_connected"}
    if _may_see_configuration(request, state):
        components["hf_token"] = state.settings.hf_token_mode
    db_future = _POOL.submit(_db, state)
    redis_future = _POOL.submit(_redis_and_worker, state)
    try:
        components["db"] = db_future.result(timeout=CHECK_BUDGET_S)
    except FuturesTimeout:
        components["db"] = "timeout"
    except Exception as exc:  # report, never raise: this endpoint is the liveness probe
        log.warning("health: database unavailable (%s)", type(exc).__name__)
        components["db"] = "error"
    try:
        components["redis"], components["worker"] = redis_future.result(timeout=CHECK_BUDGET_S)
    except FuturesTimeout:
        components["redis"], components["worker"] = "timeout", "unknown"
    except Exception as exc:
        log.warning("health: redis unavailable (%s)", type(exc).__name__)
        components["redis"], components["worker"] = "error", "unknown"
    healthy = all(components[k] == "ok" for k in ("db", "redis", "worker"))
    return HealthResponse(
        status="ok" if healthy else "degraded", version=__version__, components=components
    )
