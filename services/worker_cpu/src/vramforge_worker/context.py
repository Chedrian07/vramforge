"""The worker's `JobContext` (vramforge_estimator.pipeline) backed by PostgreSQL.

Besides the contract (stage, progress, warnings, cancel flag, checkpoints, limits, source access)
it implements the pipeline's optional extensions: an owner-scoped scan cache and publication of
the partial result after each stage. The core never sees the database.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api import store
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis, ScanCacheEntry
from vramforge_api.settings import Settings
from vramforge_estimator.pipeline import CachedScan
from vramforge_estimator.scan import ScanLimits
from vramforge_estimator.schemas import (
    AnalysisResult,
    DatasetScanResult,
    EventType,
    Issue,
    JobProgress,
    JobStatus,
)
from vramforge_estimator.sources import SourceAccess

from .lease import LeaseState

log = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"
MAX_PARTIAL_KEYS = 32
MAX_PARTIAL_STR = 200

STAGE_MESSAGES: dict[JobStatus, str] = {
    JobStatus.RESOLVING: "모델과 데이터셋 출처를 확인하고 revision을 고정하는 중입니다.",
    JobStatus.INSPECTING: "모델 구조, tokenizer, 데이터셋 컬럼을 확인하는 중입니다.",
    JobStatus.TOKENIZING: "실제 tokenizer와 학습 전처리로 전체 데이터를 토큰화하는 중입니다.",
    JobStatus.VALIDATING_DATA: "데이터 보존과 context 상한을 검사하는 중입니다.",
    JobStatus.PLANNING_BATCHES: "최악 batch 계획을 세우는 중입니다.",
    JobStatus.ESTIMATING: "단계별 메모리를 산정하는 중입니다.",
}


def scan_limits(settings: Settings) -> ScanLimits:
    """Stop the scan (PARTIAL, never silently) well before the RQ hard timeout so the job can
    still persist what it verified."""
    budget = max(60.0, settings.job_timeout_s * 0.9 - 30.0)
    return ScanLimits(max_seconds=budget)


def _clean_partial(partial: dict[str, Any] | None) -> dict[str, int | float | str | None] | None:
    if not partial:
        return None
    cleaned: dict[str, int | float | str | None] = {}
    for key, value in list(partial.items())[:MAX_PARTIAL_KEYS]:
        if isinstance(value, bool):
            cleaned[str(key)[:64]] = int(value)
        elif isinstance(value, int | float) or value is None:
            cleaned[str(key)[:64]] = value
        elif isinstance(value, str):
            cleaned[str(key)[:64]] = value[:MAX_PARTIAL_STR]
    return cleaned or None


class WorkerJobContext:
    def __init__(
        self,
        *,
        settings: Settings,
        sessions: sessionmaker[Session],
        analysis_id: str,
        owner_key: str,
        fingerprint: str,
        lease_owner: str,
        artifact_dir: Path,
        access: SourceAccess,
        state: LeaseState,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.analysis_id = analysis_id
        self.owner_key = owner_key
        self.fingerprint = fingerprint
        self.lease_owner = lease_owner
        self.artifact_dir = artifact_dir
        self.access = access
        self.state = state
        self.limits = scan_limits(settings)
        self.current_status = JobStatus.QUEUED
        self._last_event = float("-inf")
        self._pending: tuple[JobProgress, dict[str, Any] | None] | None = None

    # ---------------------------------------------------------------- helpers
    def _owned(self) -> Any:
        return (Analysis.id == self.analysis_id) & (Analysis.lease_owner == self.lease_owner)

    def _event(
        self,
        db: Session,
        event_type: EventType,
        *,
        progress: JobProgress | None = None,
        issue: Issue | None = None,
        partial: dict[str, Any] | None = None,
        lock: bool = True,
    ) -> None:
        # While a cancel is pending the job status stays CANCEL_REQUESTED (progress.stage still
        # says where the pipeline is).
        status = JobStatus.CANCEL_REQUESTED if self.state.cancel_requested else self.current_status
        store.append_event(
            db,
            analysis_id=self.analysis_id,
            fingerprint=self.fingerprint,
            event_type=event_type,
            status=status,
            progress=progress,
            issue=issue,
            partial=_clean_partial(partial),
            lock=lock,
        )

    def _still_leased(self, db: Session) -> bool:
        lease = db.scalar(select(Analysis.lease_owner).where(Analysis.id == self.analysis_id))
        if lease != self.lease_owner:
            self.state.lease_lost = True
            return False
        return True

    def flush_progress(self) -> None:
        if self._pending is None:
            return
        progress, partial = self._pending
        self._pending = None
        with session_scope(self.sessions) as db:
            # Lock before checking the lease: a terminal event committed by another process
            # (reaper, on_stopped) must never be followed by one of ours.
            store.lock_analysis(db, self.analysis_id)
            if self._still_leased(db):
                self._event(db, EventType.PROGRESS, progress=progress, partial=partial, lock=False)
        self._last_event = time.monotonic()

    # ---------------------------------------------------------------- JobContext
    def set_stage(self, stage: JobStatus) -> None:
        self.flush_progress()
        self.current_status = stage
        progress = JobProgress(stage=stage, message=STAGE_MESSAGES.get(stage))
        with session_scope(self.sessions) as db:
            store.lock_analysis(db, self.analysis_id)
            values: dict[str, Any] = {
                "progress": progress.model_dump(mode="json"),
                "updated_at": utcnow(),
            }
            # A pending cancel keeps the CANCEL_REQUESTED status visible.
            changed = db.execute(
                update(Analysis)
                .where(self._owned(), Analysis.cancel_requested.is_(False))
                .values(status=stage.value, **values)
            ).rowcount  # type: ignore[attr-defined]
            if not changed:
                self.state.cancel_requested = True
                db.execute(update(Analysis).where(self._owned()).values(**values))
            if self._still_leased(db):
                self._event(db, EventType.PROGRESS, progress=progress, lock=False)
        self._last_event = time.monotonic()

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        with session_scope(self.sessions) as db:
            db.execute(
                update(Analysis)
                .where(self._owned())
                .values(progress=progress.model_dump(mode="json"), updated_at=utcnow())
            )
        self._pending = (progress, partial)
        final = (progress.message_code or "").endswith("final") or (
            progress.total_rows is not None and progress.processed_rows == progress.total_rows
        )
        interval = self.settings.progress_event_interval_s
        if final or time.monotonic() - self._last_event >= interval:
            self.flush_progress()

    def warn(self, issue: Issue) -> None:
        self.flush_progress()
        with session_scope(self.sessions) as db:
            store.lock_analysis(db, self.analysis_id)
            if self._still_leased(db):
                self._event(db, EventType.WARNING, issue=issue, lock=False)

    def cancelled(self) -> bool:
        return self.state.cancel_requested or self.state.lease_lost

    def load_checkpoint(self) -> dict[str, Any] | None:
        path = self.artifact_dir / CHECKPOINT_FILE
        if not path.is_file():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("ignoring an unreadable checkpoint for %s", self.analysis_id)
            return None
        return data

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / CHECKPOINT_FILE
        tmp = path.with_name(f".{CHECKPOINT_FILE}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    # ---------------------------------------------------------------- extensions
    def publish_partial(self, result: AnalysisResult) -> None:
        with session_scope(self.sessions) as db:
            changed = db.execute(
                update(Analysis)
                .where(self._owned())
                .values(result=result.model_dump(mode="json"), updated_at=utcnow())
            ).rowcount  # type: ignore[attr-defined]
            if changed:
                self._event(db, EventType.PARTIAL_RESULT)

    def lookup_scan_cache(self, preprocess_key: str) -> CachedScan | None:
        with session_scope(self.sessions) as db:
            entry = db.scalar(
                select(ScanCacheEntry).where(
                    ScanCacheEntry.owner_id == self.owner_key,
                    ScanCacheEntry.preprocess_key == preprocess_key,
                )
            )
            if entry is None:
                return None
            path = store.resolve_data_path(self.settings, entry.artifact_path)
            if path is None or not path.exists():
                db.delete(entry)
                return None
            try:
                summary = entry.summary
                cached = CachedScan(
                    result=DatasetScanResult.model_validate(summary["result"]),
                    artifact_path=path,
                    template_content_loss_rows=int(summary.get("template_content_loss_rows", 0)),
                    issues=tuple(Issue.model_validate(i) for i in summary.get("issues", [])),
                )
            except (KeyError, TypeError, ValidationError):
                db.delete(entry)
                return None
            entry.last_used_at = utcnow()
            return cached

    def store_scan_cache(self, preprocess_key: str, scan: CachedScan) -> None:
        try:
            rel = str(scan.artifact_path.resolve().relative_to(self.settings.data_dir.resolve()))
        except ValueError:
            log.warning("scan artifact outside the data directory is not cached")
            return
        summary = {
            "result": scan.result.model_dump(mode="json"),
            "template_content_loss_rows": scan.template_content_loss_rows,
            "issues": [i.model_dump(mode="json") for i in scan.issues],
        }
        with session_scope(self.sessions) as db:
            entry = db.scalar(
                select(ScanCacheEntry).where(
                    ScanCacheEntry.owner_id == self.owner_key,
                    ScanCacheEntry.preprocess_key == preprocess_key,
                )
            )
            now = utcnow()
            if entry is None:
                db.add(
                    ScanCacheEntry(
                        owner_id=self.owner_key,
                        preprocess_key=preprocess_key,
                        artifact_path=rel,
                        summary=summary,
                        analysis_id=self.analysis_id,
                        created_at=now,
                        last_used_at=now,
                    )
                )
            else:
                entry.artifact_path = rel
                entry.summary = summary
                entry.analysis_id = self.analysis_id
                entry.last_used_at = now


__all__ = ["STAGE_MESSAGES", "WorkerJobContext", "scan_limits"]
