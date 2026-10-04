"""Health: always HTTP 200 (compose liveness); component states say what is degraded."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from redis.exceptions import RedisError
from sqlalchemy import text

from vramforge_api import __version__
from vramforge_estimator.schemas import HealthResponse

from ..deps import StateDep
from ..jobs import worker_heartbeat_state

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: StateDep) -> HealthResponse:
    components = {"api": "ok", "gpu_worker": "not_connected"}
    try:
        with state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        components["db"] = "ok"
    except Exception as exc:  # report, never raise: this endpoint is the liveness probe
        log.warning("health: database unavailable (%s)", type(exc).__name__)
        components["db"] = "error"
    try:
        state.redis.ping()
        components["redis"] = "ok"
        components["worker"] = worker_heartbeat_state(state.redis)
    except RedisError as exc:
        log.warning("health: redis unavailable (%s)", type(exc).__name__)
        components["redis"] = "error"
        components["worker"] = "unknown"
    healthy = all(components[k] == "ok" for k in ("db", "redis", "worker"))
    return HealthResponse(
        status="ok" if healthy else "degraded", version=__version__, components=components
    )
