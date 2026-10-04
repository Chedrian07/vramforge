"""Database operations shared by the API and the worker.

Every lookup of an analysis, upload or cache entry is scoped to the owner: another owner's id
behaves exactly like an unknown id (plan.md §18, §19.4 "Unauthorized access").
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    TERMINAL_JOB_STATUSES,
    AnalysisEvent,
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    ErrorCode,
    EventType,
    Issue,
    JobProgress,
    JobStatus,
    Severity,
    SourceType,
    Stage,
)

from .db import utcnow
from .models import Analysis, AnalysisEventRow, Owner, ScanCacheEntry, Upload
from .settings import Settings

log = logging.getLogger(__name__)

ACTIVE_STATUSES = frozenset(s.value for s in JobStatus if s not in TERMINAL_JOB_STATUSES)
TERMINAL_STATUSES = frozenset(s.value for s in TERMINAL_JOB_STATUSES)
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_PREFIX = "upload:"

# Terminal job status -> SSE event type. PARTIAL (stopped early with a partial result) is
# reported as `failed`; the event's `status` field still says PARTIAL.
TERMINAL_EVENT_FOR_STATUS: dict[JobStatus, EventType] = {
    JobStatus.COMPLETED: EventType.COMPLETED,
    JobStatus.PARTIAL: EventType.FAILED,
    JobStatus.FAILED: EventType.FAILED,
    JobStatus.CANCELLED: EventType.CANCELLED,
    JobStatus.NEEDS_INPUT: EventType.NEEDS_INPUT,
}


def is_valid_id(value: str) -> bool:
    return bool(_HEX32.match(value))


def new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------- paths


def artifact_rel_dir(owner_key: str, analysis_id: str) -> str:
    return f"artifacts/{owner_key}/{analysis_id}"


def owner_uploads_dir(settings: Settings, owner_key: str) -> Path:
    return settings.uploads_dir / owner_key


def resolve_data_path(settings: Settings, rel: str) -> Path | None:
    """Absolute path of a data-dir-relative path, or None if it escapes the data dir."""
    base = settings.data_dir.resolve()
    target = (base / rel).resolve()
    if target == base or not target.is_relative_to(base):
        return None
    return target


def remove_data_path(settings: Settings, rel: str) -> None:
    """Delete a file or directory under the data dir (never follows a symlink out of it)."""
    target = resolve_data_path(settings, rel)
    if target is None:
        log.warning("refusing to delete a path outside the data directory")
        return
    raw = settings.data_dir / rel
    try:
        if raw.is_symlink():
            raw.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not delete artifact path: %s", type(exc).__name__)


# ---------------------------------------------------------------- owners


def ensure_owner(db: Session, owner_key: str) -> None:
    owner = db.get(Owner, owner_key)
    now = utcnow()
    if owner is None:
        db.add(Owner(id=owner_key, created_at=now, last_seen_at=now))
        db.flush()
    elif now - owner.last_seen_at > timedelta(hours=1):
        owner.last_seen_at = now


def lock_owner(db: Session, owner_key: str) -> None:
    """Serialize per-owner admission checks (FOR UPDATE on PostgreSQL)."""
    db.execute(select(Owner.id).where(Owner.id == owner_key).with_for_update())


# ---------------------------------------------------------------- analyses


def get_owned_analysis(db: Session, owner_key: str, analysis_id: str) -> Analysis | None:
    if not is_valid_id(analysis_id):
        return None
    return db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == owner_key)
    )


def find_by_idempotency_key(db: Session, owner_key: str, key: str) -> Analysis | None:
    return db.scalar(
        select(Analysis).where(Analysis.owner_id == owner_key, Analysis.idempotency_key == key)
    )


def count_active(db: Session, owner_key: str) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(Analysis)
            .where(Analysis.owner_id == owner_key, Analysis.status.in_(ACTIVE_STATUSES))
        )
        or 0
    )


def create_analysis(
    db: Session,
    *,
    owner_key: str,
    request: AnalysisRequest,
    fingerprint: str,
    idempotency_key: str | None,
) -> Analysis:
    analysis_id = new_id()
    now = utcnow()
    row = Analysis(
        id=analysis_id,
        owner_id=owner_key,
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        request=request.model_dump(mode="json"),
        status=JobStatus.QUEUED.value,
        progress=JobProgress(stage=JobStatus.QUEUED).model_dump(mode="json"),
        attempt=1,
        cancel_requested=False,
        artifact_dir=artifact_rel_dir(owner_key, analysis_id),
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.flush()
    return row


def is_terminal(analysis: Analysis) -> bool:
    return analysis.status in TERMINAL_STATUSES


# ---------------------------------------------------------------- events


def lock_analysis(db: Session, analysis_id: str) -> None:
    """Row lock that serializes event inserts of one analysis (monotonic SSE ids)."""
    db.execute(select(Analysis.id).where(Analysis.id == analysis_id).with_for_update())


def append_event(
    db: Session,
    *,
    analysis_id: str,
    fingerprint: str,
    event_type: EventType,
    status: JobStatus,
    progress: JobProgress | None = None,
    issue: Issue | None = None,
    partial: dict[str, Any] | None = None,
    lock: bool = True,
) -> int:
    if lock:
        lock_analysis(db, analysis_id)
    payload = {
        "type": event_type.value,
        "status": status.value,
        "progress": progress.model_dump(mode="json") if progress else None,
        "fingerprint": fingerprint,
        "issue": issue.model_dump(mode="json") if issue else None,
        "partial": partial,
        "timestamp": utcnow().isoformat(),
    }
    row = AnalysisEventRow(analysis_id=analysis_id, type=event_type.value, payload=payload)
    db.add(row)
    db.flush()
    return int(row.id)


def event_from_row(row: AnalysisEventRow) -> AnalysisEvent:
    p = row.payload
    return AnalysisEvent(
        event_id=int(row.id),
        analysis_id=row.analysis_id,
        type=EventType(p["type"]),
        status=JobStatus(p["status"]),
        progress=JobProgress.model_validate(p["progress"]) if p.get("progress") else None,
        fingerprint=p.get("fingerprint", ""),
        issue=Issue.model_validate(p["issue"]) if p.get("issue") else None,
        partial=p.get("partial"),
        timestamp=datetime.fromisoformat(p["timestamp"]),
    )


def events_after(
    db: Session, analysis_id: str, after_id: int, limit: int = 200
) -> list[AnalysisEvent]:
    rows = db.scalars(
        select(AnalysisEventRow)
        .where(AnalysisEventRow.analysis_id == analysis_id, AnalysisEventRow.id > after_id)
        .order_by(AnalysisEventRow.id)
        .limit(limit)
    ).all()
    return [event_from_row(r) for r in rows]


def last_event_id(db: Session, analysis_id: str) -> int | None:
    value = db.scalar(
        select(func.max(AnalysisEventRow.id)).where(AnalysisEventRow.analysis_id == analysis_id)
    )
    return int(value) if value is not None else None


# ---------------------------------------------------------------- status


def internal_error_issue(message: str = "저장된 데이터를 읽을 수 없습니다.") -> Issue:
    return make_issue(ErrorCode.INTERNAL_ERROR, message, stage=Stage.API)


def load_result(analysis: Analysis) -> AnalysisResult | None:
    if not analysis.result:
        return None
    try:
        return AnalysisResult.model_validate(analysis.result)
    except ValidationError:
        log.error("stored result of %s does not match the current schema", analysis.id)
        return None


def load_request(analysis: Analysis) -> AnalysisRequest:
    return AnalysisRequest.model_validate(analysis.request)


def expires_at(analysis: Analysis, retention_days: int) -> datetime | None:
    """When retention deletes the analysis (the worker's cleanup uses the same rule): a finished
    analysis `retention_days` after it finished; None while it is still running."""
    if not is_terminal(analysis):
        return None
    return (analysis.finished_at or analysis.created_at) + timedelta(days=retention_days)


def to_status(db: Session, analysis: Analysis, *, retention_days: int) -> AnalysisStatus:
    error = Issue.model_validate(analysis.error) if analysis.error else None
    result = load_result(analysis)
    if analysis.result and result is None and error is None:
        error = internal_error_issue("저장된 분석 결과를 현재 형식으로 읽을 수 없습니다.")
    try:
        request: AnalysisRequest | None = load_request(analysis)
    except ValidationError:
        log.error("stored request of %s does not match the current schema", analysis.id)
        request = None
    return AnalysisStatus(
        analysis_id=analysis.id,
        status=JobStatus(analysis.status),
        fingerprint=analysis.fingerprint,
        progress=JobProgress.model_validate(analysis.progress) if analysis.progress else None,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        finished_at=analysis.finished_at,
        expires_at=expires_at(analysis, retention_days),
        last_event_id=last_event_id(db, analysis.id),
        request=request,
        result=result,
        error=error,
    )


def finish(
    db: Session,
    analysis: Analysis,
    status: JobStatus,
    *,
    issue: Issue | None = None,
    result: AnalysisResult | None = None,
    message: str | None = None,
) -> None:
    """Move to a terminal status and append the matching terminal event (caller commits)."""
    now = utcnow()
    analysis.status = status.value
    analysis.finished_at = now
    analysis.updated_at = now
    analysis.lease_owner = None
    analysis.lease_expires_at = None
    if issue is not None:
        analysis.error = issue.model_dump(mode="json")
    if result is not None:
        analysis.result = result.model_dump(mode="json")
    progress = JobProgress(stage=status, message=message)
    analysis.progress = progress.model_dump(mode="json")
    append_event(
        db,
        analysis_id=analysis.id,
        fingerprint=analysis.fingerprint,
        event_type=TERMINAL_EVENT_FOR_STATUS[status],
        status=status,
        progress=progress,
        issue=issue,
    )


def request_cancel(db: Session, analysis: Analysis) -> None:
    now = utcnow()
    analysis.cancel_requested = True
    analysis.cancel_requested_at = analysis.cancel_requested_at or now
    analysis.status = JobStatus.CANCEL_REQUESTED.value
    analysis.updated_at = now
    progress = JobProgress(
        stage=JobStatus.CANCEL_REQUESTED,
        message="취소를 요청했습니다. 현재 단계를 정리하는 중입니다.",
    )
    analysis.progress = progress.model_dump(mode="json")
    append_event(
        db,
        analysis_id=analysis.id,
        fingerprint=analysis.fingerprint,
        event_type=EventType.PROGRESS,
        status=JobStatus.CANCEL_REQUESTED,
        progress=progress,
    )


def cancelled_issue() -> Issue:
    return make_issue(
        ErrorCode.CANCELLED, "사용자 요청으로 작업이 취소되었습니다.", severity=Severity.WARNING
    )


# ---------------------------------------------------------------- uploads


def parse_upload_id(reference: str) -> str | None:
    ref = reference.strip()
    if ref.startswith(UPLOAD_PREFIX):
        ref = ref[len(UPLOAD_PREFIX) :]
    return ref if is_valid_id(ref) else None


def get_owned_upload(db: Session, owner_key: str, upload_id: str) -> Upload | None:
    if not is_valid_id(upload_id):
        return None
    upload = db.scalar(select(Upload).where(Upload.id == upload_id, Upload.owner_id == owner_key))
    if upload is None or upload.expires_at <= utcnow():
        return None
    return upload


def referenced_upload_ids(request: AnalysisRequest) -> list[str]:
    ids: list[str] = []
    for ref in (request.model, request.dataset):
        if ref.source_type is SourceType.UPLOAD or ref.reference.startswith(UPLOAD_PREFIX):
            upload_id = parse_upload_id(ref.reference)
            if upload_id:
                ids.append(upload_id)
    return ids


# ---------------------------------------------------------------- deletion


@dataclass
class PendingDeletion:
    """Filesystem paths (data-dir relative) to remove after the DB transaction commits."""

    paths: list[str] = field(default_factory=list)

    def apply(self, settings: Settings) -> None:
        for rel in self.paths:
            remove_data_path(settings, rel)


def delete_analysis(db: Session, analysis: Analysis) -> PendingDeletion:
    """Delete the analysis, its events, its cache entries and uploads used only by it."""
    pending = PendingDeletion(paths=[analysis.artifact_dir])
    try:
        upload_ids = referenced_upload_ids(load_request(analysis))
    except ValidationError:
        upload_ids = []
    for entry in db.scalars(
        select(ScanCacheEntry).where(
            ScanCacheEntry.owner_id == analysis.owner_id,
            ScanCacheEntry.analysis_id == analysis.id,
        )
    ).all():
        db.delete(entry)
    owner = analysis.owner_id
    db.delete(analysis)
    db.flush()
    for upload_id in upload_ids:
        upload = db.scalar(select(Upload).where(Upload.id == upload_id, Upload.owner_id == owner))
        if upload is None or _upload_still_referenced(db, owner, upload_id):
            continue
        pending.paths.append(str(Path(upload.path).parent))
        db.delete(upload)
    return pending


def _upload_still_referenced(db: Session, owner_key: str, upload_id: str) -> bool:
    for request_json in db.scalars(select(Analysis.request).where(Analysis.owner_id == owner_key)):
        try:
            request = AnalysisRequest.model_validate(request_json)
        except ValidationError:
            continue
        if upload_id in referenced_upload_ids(request):
            return True
    return False


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_EVENT_FOR_STATUS",
    "TERMINAL_STATUSES",
    "PendingDeletion",
    "append_event",
    "artifact_rel_dir",
    "cancelled_issue",
    "count_active",
    "create_analysis",
    "delete_analysis",
    "ensure_owner",
    "event_from_row",
    "events_after",
    "expires_at",
    "find_by_idempotency_key",
    "finish",
    "get_owned_analysis",
    "get_owned_upload",
    "is_terminal",
    "is_valid_id",
    "last_event_id",
    "load_request",
    "load_result",
    "lock_analysis",
    "lock_owner",
    "new_id",
    "owner_uploads_dir",
    "parse_upload_id",
    "referenced_upload_ids",
    "remove_data_path",
    "request_cancel",
    "resolve_data_path",
    "to_status",
]
