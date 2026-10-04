"""RQ task entry points (referenced by import path from the API: `vramforge_worker.tasks.*`).

`run_analysis(analysis_id, attempt)` takes the lease, runs `pipeline.analyze` with a DB-backed
context and persists the result with the terminal status and event. Unexpected exceptions become
FAILED/INTERNAL_ERROR (details are logged server-side only, with redaction). A runner that lost
its lease (deleted analysis, or the reaper handed the job to a newer attempt) writes nothing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rq.timeouts import JobTimeoutException
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api import store
from vramforge_api.access import source_access
from vramforge_api.db import get_engine, session_factory, session_scope
from vramforge_api.models import Analysis
from vramforge_api.settings import Settings, get_settings
from vramforge_estimator import pipeline
from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    AnalysisRequest,
    AnalysisResult,
    ErrorCode,
    Issue,
    JobStatus,
    Stage,
)

from .context import WorkerJobContext
from .lease import Heartbeat, LeaseState, acquire_lease, new_lease_owner

log = logging.getLogger(__name__)

FINAL_MESSAGES: dict[JobStatus, str] = {
    JobStatus.COMPLETED: "분석을 마쳤습니다.",
    JobStatus.PARTIAL: "일부 단계까지만 분석했습니다. 확인된 정보와 중단 사유를 확인하세요.",
    JobStatus.FAILED: "분석을 완료하지 못했습니다.",
    JobStatus.CANCELLED: "사용자 요청으로 취소되었습니다.",
    JobStatus.NEEDS_INPUT: "분석을 계속하려면 데이터셋 설정을 선택해야 합니다.",
}


def prepare_environment(settings: Settings) -> None:
    """Library caches under the data volume; no telemetry."""
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _sessions(settings: Settings) -> sessionmaker[Session]:
    return session_factory(get_engine(settings.database_url))


def _terminal_issue(result: AnalysisResult, status: JobStatus) -> Issue | None:
    if status is JobStatus.COMPLETED:
        return None
    halted = pipeline.halting_issue(result)
    if halted is not None:
        return halted
    return result.errors[0] if result.errors else None


def persist_outcome(
    settings: Settings,
    sessions: sessionmaker[Session],
    analysis_id: str,
    lease_owner: str,
    status: JobStatus,
    *,
    result: AnalysisResult | None,
    issue: Issue | None,
) -> bool:
    """Write the terminal state if this runner still holds the lease (False otherwise: the
    analysis was deleted or another attempt owns it)."""
    with session_scope(sessions) as db:
        store.lock_analysis(db, analysis_id)
        row = db.get(Analysis, analysis_id, populate_existing=True)
        if row is None:
            log.info("analysis %s was deleted while running", analysis_id)
            return False
        if row.lease_owner != lease_owner:
            log.warning("analysis %s: lease lost before completion; result discarded", analysis_id)
            return False
        if result is not None and result.dataset_scan is not None:
            row.preprocess_key = result.dataset_scan.preprocess_key
        store.finish(
            db, row, status, issue=issue, result=result, message=FINAL_MESSAGES.get(status)
        )
        return True


def _owner_artifacts(settings: Settings, analysis: Analysis) -> Path:
    path = store.resolve_data_path(settings, analysis.artifact_dir)
    if path is None:  # pragma: no cover - ids are validated hex, the path is always inside
        raise RuntimeError("artifact directory escapes the data directory")
    return path


def run_analysis(analysis_id: str, attempt: int) -> str:
    """RQ job: run one attempt of an analysis. Returns the final status (or "skipped")."""
    settings = get_settings()
    prepare_environment(settings)
    sessions = _sessions(settings)
    lease_owner = new_lease_owner()
    if not acquire_lease(sessions, analysis_id, int(attempt), lease_owner, settings.lease_ttl_s):
        log.info("analysis %s attempt %s: not runnable or leased elsewhere", analysis_id, attempt)
        return "skipped"

    with session_scope(sessions) as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            return "skipped"
        owner_key, fingerprint = analysis.owner_id, analysis.fingerprint
        artifact_dir = _owner_artifacts(settings, analysis)
        request_json: dict[str, Any] = analysis.request
    artifact_dir.mkdir(parents=True, exist_ok=True)

    state = LeaseState()
    heartbeat = Heartbeat(
        sessions,
        analysis_id,
        lease_owner,
        state,
        ttl_s=settings.lease_ttl_s,
        poll_s=settings.cancel_poll_s,
    )
    heartbeat.start()
    ctx = WorkerJobContext(
        settings=settings,
        sessions=sessions,
        analysis_id=analysis_id,
        owner_key=owner_key,
        fingerprint=fingerprint,
        lease_owner=lease_owner,
        artifact_dir=artifact_dir,
        access=source_access(settings, owner_key),
        state=state,
    )
    result: AnalysisResult | None = None
    issue: Issue | None
    try:
        request = AnalysisRequest.model_validate(request_json)
        result = pipeline.analyze(request, ctx)
        status = pipeline.terminal_status(result)
        issue = _terminal_issue(result, status)
    except ValidationError:
        log.error("analysis %s: stored request does not match the current schema", analysis_id)
        status = JobStatus.FAILED
        issue = make_issue(
            ErrorCode.INVALID_REQUEST,
            "저장된 요청을 현재 형식으로 읽을 수 없습니다. 분석을 다시 요청하세요.",
            stage=Stage.REQUEST,
        )
    except JobTimeoutException:
        log.error("analysis %s: job timeout", analysis_id)
        status = JobStatus.FAILED
        issue = make_issue(
            ErrorCode.JOB_TIMEOUT,
            "분석이 작업 시간 제한을 넘어 중단되었습니다.",
            stage=Stage.API,
            retryable=True,
        )
    except Exception:
        log.exception("analysis %s: unexpected error", analysis_id)
        status = JobStatus.FAILED
        issue = make_issue(
            ErrorCode.INTERNAL_ERROR,
            "분석 중 내부 오류가 발생했습니다. 잠시 후 다시 시도하세요.",
            stage=Stage.API,
            retryable=True,
        )
    finally:
        heartbeat.stop()
    try:
        ctx.flush_progress()
    except Exception:  # progress is display-only
        log.warning("analysis %s: final progress flush failed", analysis_id)
    if state.lease_lost and status is not JobStatus.CANCELLED:
        log.warning("analysis %s: lease lost; not persisting", analysis_id)
    written = persist_outcome(
        settings, sessions, analysis_id, lease_owner, status, result=result, issue=issue
    )
    if not written:
        with session_scope(sessions) as db:
            if db.get(Analysis, analysis_id) is None:
                store.remove_data_path(settings, str(artifact_dir.relative_to(settings.data_dir)))
        return "discarded"
    return status.value


def on_analysis_stopped(job: Any, connection: Any, *args: Any, **kwargs: Any) -> None:
    """RQ `on_stopped` callback (runs in the worker parent after `send_stop_job_command`)."""
    try:
        analysis_id, attempt = str(job.args[0]), int(job.args[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return
    settings = get_settings()
    sessions = _sessions(settings)
    with session_scope(sessions) as db:
        store.lock_analysis(db, analysis_id)
        row = db.get(Analysis, analysis_id, populate_existing=True)
        if row is None or store.is_terminal(row) or row.attempt != attempt:
            return
        if row.cancel_requested:
            store.finish(
                db,
                row,
                JobStatus.CANCELLED,
                issue=store.cancelled_issue(),
                message="사용자 요청으로 작업을 중지했습니다.",
            )
        else:
            store.finish(
                db,
                row,
                JobStatus.FAILED,
                issue=make_issue(
                    ErrorCode.INTERNAL_ERROR,
                    "작업이 강제로 중지되었습니다.",
                    stage=Stage.API,
                    retryable=True,
                ),
            )


__all__ = ["on_analysis_stopped", "persist_outcome", "prepare_environment", "run_analysis"]
