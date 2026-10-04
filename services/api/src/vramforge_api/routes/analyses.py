"""Analysis jobs: create, status, SSE events, cancel, scenarios, export, delete, GPU profile."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Response
from fastapi.responses import StreamingResponse

from vramforge_estimator.schemas import (
    AnalysisCreated,
    AnalysisRequest,
    AnalysisStatus,
    ErrorResponse,
    ScenarioRequest,
    ScenarioResponse,
)

from ._contract import ERROR_RESPONSES, not_implemented

router = APIRouter(prefix="/analyses", tags=["analyses"], responses=ERROR_RESPONSES)

ExportFormat = Literal["json", "yaml", "md", "trainer-config"]


@router.post("", response_model=AnalysisCreated, status_code=202)
def create_analysis(
    body: AnalysisRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
) -> AnalysisCreated:
    """Validate, enqueue a full-dataset analysis and return its id (202)."""
    raise not_implemented()


@router.get("/{analysis_id}", response_model=AnalysisStatus)
def get_analysis(analysis_id: str) -> AnalysisStatus:
    raise not_implemented()


@router.get(
    "/{analysis_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream"}},
)
def analysis_events(
    analysis_id: str,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Server-sent events (`AnalysisEvent` JSON in `data:`), resumable with Last-Event-ID."""
    raise not_implemented()


@router.post("/{analysis_id}/cancel", response_model=AnalysisStatus)
def cancel_analysis(analysis_id: str) -> AnalysisStatus:
    raise not_implemented()


@router.post("/{analysis_id}/scenarios", response_model=ScenarioResponse)
def scenarios(analysis_id: str, body: ScenarioRequest) -> ScenarioResponse:
    """Recompute batch plan and memory from cached artifacts, or report re-analysis is needed."""
    raise not_implemented()


@router.get(
    "/{analysis_id}/export",
    responses={
        200: {"content": {"application/json": {}, "application/yaml": {}, "text/markdown": {}}}
    },
)
def export_analysis(
    analysis_id: str, format: Annotated[ExportFormat, Query()] = "json"
) -> Response:
    raise not_implemented()


@router.delete("/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str) -> Response:
    raise not_implemented()


@router.post(
    "/{analysis_id}/profile",
    status_code=503,
    responses={503: {"model": ErrorResponse, "description": "GPU worker not connected"}},
)
def start_gpu_profile(analysis_id: str) -> Response:
    """Opt-in GPU validation (M5). Not available in this release: always GPU_WORKER_UNAVAILABLE."""
    raise not_implemented()
