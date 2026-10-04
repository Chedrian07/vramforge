"""Liveness/readiness."""

from __future__ import annotations

from fastapi import APIRouter

from vramforge_api import __version__
from vramforge_estimator.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, components={"api": "ok"})
