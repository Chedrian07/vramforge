"""Source inspection, uploads, local roots and backend profiles."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile

from vramforge_estimator.schemas import (
    BackendProfilesResponse,
    InspectRequest,
    InspectResponse,
    LocalRootsResponse,
    UploadResponse,
)

from ._contract import ERROR_RESPONSES, not_implemented

router = APIRouter(tags=["sources"], responses=ERROR_RESPONSES)


@router.post("/sources/inspect", response_model=InspectResponse)
def inspect_sources(body: InspectRequest) -> InspectResponse:
    """Metadata-only inspection of a model and/or dataset (no weights, no full scan)."""
    raise not_implemented()


@router.post("/uploads", response_model=UploadResponse, status_code=201)
async def upload_dataset(file: UploadFile) -> UploadResponse:
    """Upload a dataset file (JSON/JSONL/Parquet/Arrow/CSV) for analysis."""
    raise not_implemented()


@router.get("/local-roots", response_model=LocalRootsResponse)
def local_roots() -> LocalRootsResponse:
    """Read-only server-side roots usable as `local:<name>/<path>` references."""
    raise not_implemented()


@router.get("/backend-profiles", response_model=BackendProfilesResponse)
def backend_profiles() -> BackendProfilesResponse:
    """Supported objective × strategy combinations, environments and GPU presets."""
    raise not_implemented()
