"""Source inspection, uploads, local roots and backend profiles."""

from __future__ import annotations

import logging
import shutil
from datetime import timedelta
from pathlib import Path

import anyio
from fastapi import APIRouter, Request
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from vramforge_estimator import compatibility
from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    BackendProfilesResponse,
    DatasetSourceRef,
    ErrorCode,
    ErrorResponse,
    InspectRequest,
    InspectResponse,
    Issue,
    LocalRoot,
    LocalRootsResponse,
    ModelSourceRef,
    SourceType,
    Stage,
    UploadResponse,
)

from .. import store
from ..access import source_access
from ..db import session_scope, utcnow
from ..deps import OwnerDep, StateDep
from ..inspect_service import inspect_sources as run_inspection
from ..models import Upload
from ..settings import Settings
from ..uploads import SavedUpload, receive_upload
from ._contract import ERROR_RESPONSES

log = logging.getLogger(__name__)

router = APIRouter(tags=["sources"], responses=ERROR_RESPONSES)

UPLOAD_BODY_SCHEMA = "Body_upload_dataset_api_v1_uploads_post"


def upload_reference_issue(
    settings: Settings,
    db_factory: sessionmaker[Session],
    owner: str,
    ref: ModelSourceRef | DatasetSourceRef,
) -> Issue | None:
    """An issue when `ref` points at an upload or local root the caller may not use."""
    reference = ref.reference.strip()
    if ref.source_type is SourceType.UPLOAD or reference.startswith(store.UPLOAD_PREFIX):
        upload_id = store.parse_upload_id(reference)
        with session_scope(db_factory) as db:
            found = upload_id is not None and store.get_owned_upload(db, owner, upload_id)
        if not found:
            return make_issue(
                ErrorCode.SOURCE_NOT_FOUND,
                "업로드한 파일을 찾을 수 없습니다. "
                "만료되었거나 삭제되었을 수 있으니 다시 업로드하세요.",
                stage=Stage.REQUEST,
                field="reference",
            )
    if ref.source_type is SourceType.LOCAL or reference.startswith("local:"):
        root = reference.removeprefix("local:").split("/", 1)[0]
        if root not in settings.local_root_map:
            return make_issue(
                ErrorCode.LOCAL_PATH_NOT_ALLOWED,
                "등록되지 않은 로컬 경로입니다. 허용된 로컬 root 아래의 경로만 사용할 수 있습니다.",
                stage=Stage.REQUEST,
                field="reference",
            )
    return None


@router.post("/sources/inspect", response_model=InspectResponse)
async def inspect_sources(
    body: InspectRequest, state: StateDep, owner: OwnerDep
) -> InspectResponse:
    """Metadata-only inspection of a model and/or dataset (no weights, no full scan)."""
    settings = state.settings
    model_issue = dataset_issue = None
    if body.model is not None:
        model_issue = await run_in_threadpool(
            upload_reference_issue, settings, state.sessions, owner, body.model
        )
    if body.dataset is not None:
        dataset_issue = await run_in_threadpool(
            upload_reference_issue, settings, state.sessions, owner, body.dataset
        )
    limiter = state.extras.get("inspect_limiter")
    if limiter is None:
        limiter = state.extras.setdefault("inspect_limiter", anyio.CapacityLimiter(4))
    access = source_access(
        settings, owner, http_timeout_s=min(30.0, float(settings.inspect_timeout_s))
    )
    return await run_inspection(
        body,
        access,
        timeout_s=settings.inspect_timeout_s,
        model_issue=model_issue,
        dataset_issue=dataset_issue,
        limiter=limiter,
    )


def _record_upload(
    settings: Settings,
    db_factory: sessionmaker[Session],
    owner: str,
    upload_id: str,
    saved: SavedUpload,
) -> Upload:
    now = utcnow()
    with session_scope(db_factory) as db:
        store.ensure_owner(db, owner)
        row = Upload(
            id=upload_id,
            owner_id=owner,
            filename=saved.filename,
            format=saved.format,
            size_bytes=saved.size_bytes,
            sha256=saved.sha256,
            path=str(saved.path.relative_to(settings.data_dir)),
            created_at=now,
            expires_at=now + timedelta(days=settings.retention_days),
        )
        db.add(row)
    return row


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=201,
    responses={
        413: {"model": ErrorResponse, "description": "Upload too large"},
        415: {"model": ErrorResponse, "description": "File type not allowed"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {"$ref": f"#/components/schemas/{UPLOAD_BODY_SCHEMA}"}
                }
            },
            "required": True,
        }
    },
)
async def upload_dataset(request: Request, state: StateDep, owner: OwnerDep) -> UploadResponse:
    """Upload a dataset file (JSON/JSONL/Parquet/Arrow/CSV) for analysis.

    Streams the single `file` part to disk with the `VRAMFORGE_MAX_UPLOAD_BYTES` cap; the
    extension must match the content. Use the returned `reference` as the dataset reference.
    """
    settings = state.settings
    upload_id = store.new_id()
    target_dir = store.owner_uploads_dir(settings, owner) / upload_id
    try:
        saved = await receive_upload(
            request.headers.get("content-type"),
            request.stream(),
            target_dir,
            settings.max_upload_bytes,
        )
    except BaseException:
        await run_in_threadpool(_remove_dir_quietly, target_dir)
        raise
    try:
        row = await run_in_threadpool(
            _record_upload, settings, state.sessions, owner, upload_id, saved
        )
    except BaseException:
        await run_in_threadpool(_remove_dir_quietly, target_dir)
        raise
    return UploadResponse(
        upload_id=upload_id,
        reference=f"{store.UPLOAD_PREFIX}{upload_id}",
        filename=row.filename,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        expires_at=row.expires_at,
    )


def _remove_dir_quietly(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


@router.get("/local-roots", response_model=LocalRootsResponse)
def local_roots(state: StateDep) -> LocalRootsResponse:
    """Read-only server-side roots usable as `local:<name>/<path>` references."""
    roots = []
    for name, path in sorted(state.settings.local_root_map.items()):
        if not path.is_dir():
            log.info("local root %r is configured but not mounted", name)
            continue
        roots.append(
            LocalRoot(
                name=name,
                reference_prefix=f"local:{name}/",
                description=(
                    "서버에 읽기 전용으로 연결된 로컬 경로입니다. "
                    f"local:{name}/<상대 경로> 형식으로 지정하세요."
                ),
                read_only=True,
            )
        )
    return LocalRootsResponse(roots=roots)


@router.get("/backend-profiles", response_model=BackendProfilesResponse)
def backend_profiles() -> BackendProfilesResponse:
    """Supported objective × strategy combinations, environments and GPU presets."""
    profiles = compatibility.backend_profiles()
    # GPU validation is opt-in and not connected in this release (plan.md §17.1).
    return profiles.model_copy(update={"gpu_worker_connected": False})
