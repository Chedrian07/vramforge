"""Retention cleanup (plan.md §16.4, docs/privacy-and-retention.md).

Finished analyses older than `VRAMFORGE_RETENTION_DAYS` are deleted with their events, artifacts,
cache references and uploads used only by them; expired uploads are deleted unless a running
analysis still needs them. Orphaned directories (no database row) are removed after a day. Only
paths with the service's own layout under the data directory are ever touched; the shared HF
cache is left alone.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api import store
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis, ScanCacheEntry, Upload
from vramforge_api.settings import Settings
from vramforge_estimator.schemas import AnalysisRequest

log = logging.getLogger(__name__)

_OWNER_DIR = re.compile(r"^[0-9a-f]{64}$")
_ITEM_DIR = re.compile(r"^[0-9a-f]{32}$")
ORPHAN_GRACE_S = 86_400


@dataclass
class CleanupReport:
    analyses: list[str] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    cache_entries: int = 0
    orphan_dirs: int = 0


def cleanup_expired(settings: Settings, sessions: sessionmaker[Session]) -> CleanupReport:
    report = CleanupReport()
    now = utcnow()
    cutoff = now - timedelta(days=settings.retention_days)

    with session_scope(sessions) as db:
        old = db.scalars(
            select(Analysis.id).where(
                Analysis.status.in_(store.TERMINAL_STATUSES),
                func.coalesce(Analysis.finished_at, Analysis.created_at) < cutoff,
            )
        ).all()
    for analysis_id in old:
        with session_scope(sessions) as db:
            row = db.get(Analysis, analysis_id)
            if row is None:
                continue
            pending = store.delete_analysis(db, row)
        pending.apply(settings)
        report.analyses.append(analysis_id)

    with session_scope(sessions) as db:
        active_requests = db.scalars(
            select(Analysis.request).where(Analysis.status.in_(store.ACTIVE_STATUSES))
        ).all()
        in_use = {uid for req in active_requests for uid in _upload_ids(req)}
        expired = db.scalars(select(Upload).where(Upload.expires_at < now)).all()
        doomed = []
        for upload in expired:
            if upload.id in in_use:
                continue
            doomed.append(str(Path(upload.path).parent))
            report.uploads.append(upload.id)
            db.delete(upload)
    for rel in doomed:
        store.remove_data_path(settings, rel)

    with session_scope(sessions) as db:
        for entry in db.scalars(select(ScanCacheEntry)).all():
            path = store.resolve_data_path(settings, entry.artifact_path)
            if entry.last_used_at < cutoff or path is None or not path.exists():
                db.delete(entry)
                report.cache_entries += 1

    report.orphan_dirs = _remove_orphans(settings, sessions)
    if report.analyses or report.uploads or report.cache_entries or report.orphan_dirs:
        log.info(
            "retention: %d analyses, %d uploads, %d cache entries, %d orphan dirs removed",
            len(report.analyses),
            len(report.uploads),
            report.cache_entries,
            report.orphan_dirs,
        )
    return report


def _upload_ids(request_json: object) -> list[str]:
    try:
        return store.referenced_upload_ids(AnalysisRequest.model_validate(request_json))
    except ValidationError:
        return []


def _remove_orphans(settings: Settings, sessions: sessionmaker[Session]) -> int:
    removed = 0
    horizon = time.time() - ORPHAN_GRACE_S
    for base, model in ((settings.artifacts_dir, Analysis), (settings.uploads_dir, Upload)):
        if not base.is_dir():
            continue
        for owner_dir in base.iterdir():
            if not owner_dir.is_dir() or owner_dir.is_symlink():
                continue
            if not _OWNER_DIR.match(owner_dir.name):
                continue
            for item in owner_dir.iterdir():
                if not _ITEM_DIR.match(item.name) or item.is_symlink():
                    continue
                if item.stat().st_mtime > horizon:
                    continue
                with session_scope(sessions) as db:
                    exists = db.get(model, item.name) is not None
                if not exists:
                    store.remove_data_path(settings, str(item.relative_to(settings.data_dir)))
                    removed += 1
    return removed


__all__ = ["CleanupReport", "cleanup_expired"]
