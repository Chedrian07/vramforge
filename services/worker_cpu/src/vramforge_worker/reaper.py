"""Job recovery (plan.md §16.1–16.2, docs/research/stack-compat.md R3/R4).

- Expired leases (a work-horse died, e.g. OOM-killed: RQ runs no `on_failure` then) are re-queued
  as a new attempt `<id>-a<n+1>` up to `max_attempts`, after which the analysis is FAILED. The
  attempt number is part of the lease condition, so a stale runner can never write.
- QUEUED analyses whose RQ job vanished (Redis restart) are re-enqueued.
- Pending cancels are escalated with `send_stop_job_command` after the grace period; analyses
  without a runner are cancelled directly.

Every transition is a conditional update, so several workers may run the reaper concurrently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api import jobs, store
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis
from vramforge_api.settings import Settings
from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import ErrorCode, EventType, JobProgress, JobStatus, Stage

log = logging.getLogger(__name__)

_ENDED_RQ_STATES = frozenset({"finished", "failed", "stopped", "canceled"})


@dataclass
class ReapReport:
    requeued: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    reenqueued: list[str] = field(default_factory=list)
    stop_sent: list[str] = field(default_factory=list)


def _retry(
    db: Session, row: Analysis, settings: Settings, report: ReapReport, reason: str
) -> int | None:
    """Move to the next attempt (QUEUED) or fail after the last one. Returns the new attempt."""
    if row.cancel_requested:
        store.finish(
            db, row, JobStatus.CANCELLED, issue=store.cancelled_issue(), message="취소되었습니다."
        )
        report.cancelled.append(row.id)
        return None
    if row.attempt >= settings.max_attempts:
        store.finish(
            db,
            row,
            JobStatus.FAILED,
            issue=make_issue(
                ErrorCode.INTERNAL_ERROR,
                f"작업 실행이 {row.attempt}번 중단되어 더 시도하지 않습니다. 작업 메모리나 "
                "시간 한도를 확인하세요.",
                stage=Stage.API,
                retryable=True,
                reason=reason,
                attempts=row.attempt,
            ),
        )
        report.failed.append(row.id)
        return None
    new_attempt = row.attempt + 1
    changed = db.execute(
        update(Analysis)
        .where(Analysis.id == row.id, Analysis.attempt == row.attempt)
        .values(
            attempt=new_attempt,
            status=JobStatus.QUEUED.value,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=utcnow(),
            progress=JobProgress(
                stage=JobStatus.QUEUED,
                message=f"작업을 다시 시도합니다 ({new_attempt}/{settings.max_attempts}).",
            ).model_dump(mode="json"),
        )
    ).rowcount  # type: ignore[attr-defined]
    if not changed:
        return None
    store.append_event(
        db,
        analysis_id=row.id,
        fingerprint=row.fingerprint,
        event_type=EventType.PROGRESS,
        status=JobStatus.QUEUED,
        progress=JobProgress(
            stage=JobStatus.QUEUED,
            message=(
                f"이전 실행이 중단되어 다시 시도합니다 ({new_attempt}/{settings.max_attempts})."
            ),
        ),
        lock=False,
    )
    report.requeued.append(row.id)
    return new_attempt


def _enqueue(queue: Queue, settings: Settings, analysis_id: str, attempt: int) -> None:
    try:
        jobs.enqueue_analysis(queue, analysis_id, attempt, settings.job_timeout_s)
    except RedisError as exc:
        log.warning("re-enqueue of %s failed (%s); retrying later", analysis_id, type(exc).__name__)


def reap(
    settings: Settings, sessions: sessionmaker[Session], redis: Redis, queue: Queue
) -> ReapReport:
    report = ReapReport()
    now = utcnow()
    to_enqueue: list[tuple[str, int]] = []

    # 1. expired leases
    with session_scope(sessions) as db:
        expired = db.scalars(
            select(Analysis.id).where(
                Analysis.status.in_(store.ACTIVE_STATUSES),
                Analysis.lease_owner.is_not(None),
                Analysis.lease_expires_at < now,
            )
        ).all()
    for analysis_id in expired:
        with session_scope(sessions) as db:
            store.lock_analysis(db, analysis_id)
            row = db.get(Analysis, analysis_id, populate_existing=True)
            if (
                row is None
                or store.is_terminal(row)
                or row.lease_owner is None
                or row.lease_expires_at is None
                or row.lease_expires_at >= now
            ):
                continue
            log.warning("analysis %s: lease expired (attempt %d)", analysis_id, row.attempt)
            attempt = _retry(db, row, settings, report, "lease_expired")
            if attempt is not None:
                to_enqueue.append((analysis_id, attempt))

    # 2. queued analyses whose RQ job disappeared or ended without running
    stale_before = now - timedelta(seconds=settings.queued_requeue_after_s)
    with session_scope(sessions) as db:
        queued = db.execute(
            select(Analysis.id, Analysis.attempt).where(
                Analysis.status == JobStatus.QUEUED.value,
                Analysis.lease_owner.is_(None),
                Analysis.updated_at < stale_before,
            )
        ).all()
    for analysis_id, attempt in queued:
        try:
            rq_state = jobs.job_status(redis, analysis_id, attempt)
        except RedisError:
            break
        if rq_state is None:
            to_enqueue.append((analysis_id, attempt))
            report.reenqueued.append(analysis_id)
        elif rq_state in _ENDED_RQ_STATES:
            with session_scope(sessions) as db:
                store.lock_analysis(db, analysis_id)
                row = db.get(Analysis, analysis_id, populate_existing=True)
                if row is None or row.status != JobStatus.QUEUED.value or row.attempt != attempt:
                    continue
                new_attempt = _retry(db, row, settings, report, f"rq_{rq_state}")
                if new_attempt is not None:
                    to_enqueue.append((analysis_id, new_attempt))

    # 3. cancel escalation
    grace_before = now - timedelta(seconds=settings.cancel_grace_s)
    with session_scope(sessions) as db:
        pending = db.execute(
            select(Analysis.id, Analysis.attempt, Analysis.lease_owner).where(
                Analysis.cancel_requested.is_(True),
                Analysis.status.in_(store.ACTIVE_STATUSES),
                Analysis.cancel_requested_at < grace_before,
            )
        ).all()
    for analysis_id, attempt, lease_owner in pending:
        if lease_owner is None:
            with session_scope(sessions) as db:
                store.lock_analysis(db, analysis_id)
                row = db.get(Analysis, analysis_id, populate_existing=True)
                if row is not None and not store.is_terminal(row) and row.lease_owner is None:
                    store.finish(
                        db,
                        row,
                        JobStatus.CANCELLED,
                        issue=store.cancelled_issue(),
                        message="사용자 요청으로 취소되었습니다.",
                    )
                    report.cancelled.append(analysis_id)
        elif jobs.stop_running_job(redis, analysis_id, attempt):
            report.stop_sent.append(analysis_id)

    for analysis_id, attempt in to_enqueue:
        _enqueue(queue, settings, analysis_id, attempt)
    return report


__all__ = ["ReapReport", "reap"]
