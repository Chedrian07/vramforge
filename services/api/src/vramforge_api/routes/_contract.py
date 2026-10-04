"""Shared response declarations for the route contract."""

from __future__ import annotations

from typing import Any

from vramforge_estimator.schemas import ErrorResponse

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid request"},
    401: {"model": ErrorResponse, "description": "Access token required"},
    403: {
        "model": ErrorResponse,
        "description": "Forbidden (owner mismatch or CSRF header missing)",
    },
    404: {"model": ErrorResponse, "description": "Not found"},
    409: {"model": ErrorResponse, "description": "Conflict"},
    422: {"model": ErrorResponse, "description": "Validation or compatibility error"},
    429: {"model": ErrorResponse, "description": "Concurrency limit"},
}
