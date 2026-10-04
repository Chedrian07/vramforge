"""Analysis jobs: create, status, SSE events, cancel, scenarios, export, delete, GPU profile.

Every handler resolves the analysis through the caller's owner id first; an id that belongs to
someone else is indistinguishable from an unknown id (404).
"""

from __future__ import annotations

import logging
import threading
from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from vramforge_estimator import compatibility, pipeline
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.exports import (
    export_json,
    export_plan_yaml,
    export_report_md,
    export_trainer_config,
)
from vramforge_estimator.keys import request_fingerprint
from vramforge_estimator.schemas import (
    AnalysisCreated,
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    ErrorCode,
    ErrorResponse,
    EventType,
    Issue,
    JobProgress,
    JobStatus,
    ScenarioRequest,
    ScenarioResponse,
    Severity,
    Stage,
)

from .. import jobs, store
from ..db import session_scope
from ..deps import DbDep, OwnerDep, StateDep
from ..errors import ApiError, api_error, not_found
from ..models import Analysis
from ..sse import SSE_HEADERS, event_stream, parse_last_event_id
from ..state import AppState
from ._contract import ERROR_RESPONSES
from .sources import upload_reference_issue

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analyses", tags=["analyses"], responses=ERROR_RESPONSES)

ExportFormat = Literal["json", "yaml", "md", "trainer-config"]

EXPORTS = {
    "json": (export_json, "application/json", "analysis.json"),
    "yaml": (export_plan_yaml, "application/yaml; charset=utf-8", "resolved-plan.yaml"),
    "md": (export_report_md, "text/markdown; charset=utf-8", "report.md"),
    "trainer-config": (
        export_trainer_config,
        "application/yaml; charset=utf-8",
        "trainer-config.yaml",
    ),
}


def validate_request_or_422(request: AnalysisRequest) -> list[Issue]:
    """Server-side capability validation (plan.md §15.1); blocking issues → 422."""
    try:
        issues = compatibility.validate_request(request)
    except NotImplementedError:
        log.warning("compatibility.validate_request is not implemented; skipping pre-validation")
        return []
    errors = [i for i in issues if i.severity is Severity.ERROR]
    if errors:
        raise ApiError(422, errors[0], issues)
    return issues


def _check_references(state: AppState, owner: str, request: AnalysisRequest) -> None:
    for ref in (request.model, request.dataset):
        issue = upload_reference_issue(state.settings, state.sessions, owner, ref)
        if issue is not None:
            raise ApiError(422, issue)


def _owned(db: DbDep, owner: str, analysis_id: str) -> Analysis:
    analysis = store.get_owned_analysis(db, owner, analysis_id)
    if analysis is None:
        raise not_found()
    return analysis


def _reused(existing: Analysis, fingerprint: str) -> AnalysisCreated:
    if existing.fingerprint != fingerprint:
        raise api_error(
            409,
            ErrorCode.INVALID_REQUEST,
            "같은 Idempotency-Key가 다른 요청 내용에 이미 사용되었습니다.",
            stage=Stage.REQUEST,
        )
    return AnalysisCreated(
        analysis_id=existing.id,
        status=JobStatus(existing.status),
        fingerprint=existing.fingerprint,
        created_at=existing.created_at,
        reused=True,
    )


@router.post(
    "",
    response_model=AnalysisCreated,
    status_code=202,
    responses={503: {"model": ErrorResponse, "description": "Job queue unavailable"}},
)
def create_analysis(
    body: AnalysisRequest,
    state: StateDep,
    db: DbDep,
    owner: OwnerDep,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
) -> AnalysisCreated:
    """Validate, enqueue a full-dataset analysis and return its id (202)."""
    settings = state.settings
    fingerprint = request_fingerprint(body)
    key = idempotency_key.strip() if idempotency_key else None
    if key:
        existing = store.find_by_idempotency_key(db, owner, key)
        if existing is not None:
            return _reused(existing, fingerprint)
    _check_references(state, owner, body)
    validate_request_or_422(body)

    store.ensure_owner(db, owner)
    store.lock_owner(db, owner)
    if store.count_active(db, owner) >= settings.max_concurrent_jobs_per_owner:
        raise api_error(
            429,
            ErrorCode.CONCURRENCY_LIMIT,
            f"동시에 실행할 수 있는 분석은 {settings.max_concurrent_jobs_per_owner}개입니다. "
            "진행 중인 분석이 끝나거나 취소한 뒤 다시 시도하세요.",
            retryable=True,
            limit=settings.max_concurrent_jobs_per_owner,
        )
    analysis = store.create_analysis(
        db, owner_key=owner, request=body, fingerprint=fingerprint, idempotency_key=key
    )
    store.append_event(
        db,
        analysis_id=analysis.id,
        fingerprint=fingerprint,
        event_type=EventType.PROGRESS,
        status=JobStatus.QUEUED,
        progress=JobProgress(stage=JobStatus.QUEUED, message="분석 대기열에 등록되었습니다."),
        lock=False,
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = store.find_by_idempotency_key(db, owner, key) if key else None
        if existing is not None:
            return _reused(existing, fingerprint)
        raise

    try:
        jobs.enqueue_analysis(state.queue(), analysis.id, analysis.attempt, settings.job_timeout_s)
    except RedisError:
        log.error("could not enqueue analysis %s", analysis.id)
        failed = db.get(Analysis, analysis.id)
        issue = make_issue(
            ErrorCode.INTERNAL_ERROR,
            "작업 큐에 연결할 수 없습니다. 잠시 후 다시 시도하세요.",
            stage=Stage.API,
            retryable=True,
        )
        if failed is not None and not store.is_terminal(failed):
            store.finish(db, failed, JobStatus.FAILED, issue=issue)
            db.commit()
        raise ApiError(503, issue) from None

    db.refresh(analysis)  # a synchronous queue (tests) may already have run the job
    return AnalysisCreated(
        analysis_id=analysis.id,
        status=JobStatus(analysis.status),
        fingerprint=fingerprint,
        created_at=analysis.created_at,
        reused=False,
    )


@router.get("/{analysis_id}", response_model=AnalysisStatus)
def get_analysis(analysis_id: str, db: DbDep, owner: OwnerDep) -> AnalysisStatus:
    return store.to_status(db, _owned(db, owner, analysis_id))


def _stream_precheck(
    state: AppState, owner: str, analysis_id: str, after_id: int
) -> tuple[bool, bool]:
    """(exists for this owner, nothing left to replay for a finished analysis)."""
    with session_scope(state.sessions) as db:
        analysis = store.get_owned_analysis(db, owner, analysis_id)
        if analysis is None:
            return False, False
        last = store.last_event_id(db, analysis_id) or 0
        return True, store.is_terminal(analysis) and after_id >= last


@router.get(
    "/{analysis_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
        204: {"description": "Finished analysis with no events after Last-Event-ID"},
    },
)
async def analysis_events(
    analysis_id: str,
    request: Request,
    state: StateDep,
    owner: OwnerDep,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[int | None, Query(ge=0, description="Resume after this event id")] = None,
) -> Response:
    """Server-sent events (`AnalysisEvent` JSON in `data:`), resumable with Last-Event-ID.

    Frames are `id: <event_id>`, `event: <type>`, `data: <AnalysisEvent JSON>`. Types:
    `progress` (stage changes and coalesced scan progress), `partial_result` (refetch the
    analysis for the result so far), `warning`, and the terminal types `completed`, `failed`
    (status FAILED, or PARTIAL when the pipeline stopped with a partial result), `cancelled`
    and `needs_input`. The stream ends after a terminal event; reconnecting with the last id
    of a finished analysis returns 204. `after` resumes without the header (first connect).
    """
    after_id = parse_last_event_id(last_event_id, after)
    exists, finished = await run_in_threadpool(
        _stream_precheck, state, owner, analysis_id, after_id
    )
    if not exists:
        raise not_found()
    if finished:
        return Response(status_code=204)  # stops EventSource reconnects
    return StreamingResponse(
        event_stream(state, analysis_id, after_id, request.is_disconnected),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _stop_if_still_running(state: AppState, analysis_id: str, attempt: int) -> None:
    try:
        with session_scope(state.sessions) as db:
            analysis = db.get(Analysis, analysis_id)
            if (
                analysis is None
                or store.is_terminal(analysis)
                or analysis.attempt != attempt
                or not analysis.cancel_requested
            ):
                return
        jobs.stop_running_job(state.redis, analysis_id, attempt)
    except Exception as exc:  # best effort; the worker reaper retries the escalation
        log.warning("cancel escalation for %s failed: %s", analysis_id, type(exc).__name__)


def schedule_stop(state: AppState, analysis_id: str, attempt: int) -> None:
    """Force-stop after the cooperative grace period (docs/research/stack-compat.md R3)."""
    grace = state.settings.cancel_grace_s
    if grace <= 0:
        _stop_if_still_running(state, analysis_id, attempt)
        return
    timer = threading.Timer(grace, _stop_if_still_running, args=(state, analysis_id, attempt))
    timer.daemon = True
    timer.start()


@router.post("/{analysis_id}/cancel", response_model=AnalysisStatus)
def cancel_analysis(
    analysis_id: str, state: StateDep, db: DbDep, owner: OwnerDep
) -> AnalysisStatus:
    analysis = _owned(db, owner, analysis_id)
    db.refresh(analysis, with_for_update=True)
    if store.is_terminal(analysis):
        return store.to_status(db, analysis)
    if analysis.status == JobStatus.QUEUED.value and analysis.lease_owner is None:
        try:
            jobs.cancel_queued_job(state.redis, analysis.id, analysis.attempt)
        except RedisError:
            log.warning("could not reach redis to cancel %s; the worker will skip it", analysis.id)
        store.finish(
            db,
            analysis,
            JobStatus.CANCELLED,
            issue=store.cancelled_issue(),
            message="사용자 요청으로 취소되었습니다.",
        )
        db.commit()
    elif not analysis.cancel_requested:
        store.request_cancel(db, analysis)
        db.commit()
        schedule_stop(state, analysis.id, analysis.attempt)
    return store.to_status(db, analysis)


@router.post("/{analysis_id}/scenarios", response_model=ScenarioResponse)
def scenarios(
    analysis_id: str, body: ScenarioRequest, state: StateDep, db: DbDep, owner: OwnerDep
) -> ScenarioResponse:
    """Recompute batch plan and memory from cached artifacts, or report re-analysis is needed."""
    analysis = _owned(db, owner, analysis_id)
    if not store.is_terminal(analysis):
        raise api_error(
            409, ErrorCode.INVALID_REQUEST, "분석이 끝난 뒤에 조건을 바꿔 재계산할 수 있습니다."
        )
    validate_request_or_422(body.request)
    base = store.load_result(analysis)
    if base is None:
        return ScenarioResponse(
            fingerprint=request_fingerprint(body.request),
            client_fingerprint=body.client_fingerprint,
            requires_reanalysis=True,
            reanalysis_reasons=["저장된 분석 결과가 없어 데이터 재분석이 필요합니다."],
        )
    artifact_dir = store.resolve_data_path(state.settings, analysis.artifact_dir)
    if artifact_dir is None:
        raise not_found()
    response = pipeline.recompute(base, body.request, artifact_dir)
    return response.model_copy(update={"client_fingerprint": body.client_fingerprint})


@router.get(
    "/{analysis_id}/export",
    responses={
        200: {"content": {"application/json": {}, "application/yaml": {}, "text/markdown": {}}}
    },
)
def export_analysis(
    analysis_id: str,
    db: DbDep,
    owner: OwnerDep,
    format: Annotated[ExportFormat, Query()] = "json",
) -> Response:
    analysis = _owned(db, owner, analysis_id)
    result: AnalysisResult | None = store.load_result(analysis)
    if result is None:
        raise api_error(409, ErrorCode.INVALID_REQUEST, "내보낼 분석 결과가 아직 없습니다.")
    exporter, media_type, filename = EXPORTS[format]
    try:
        content = exporter(result)
    except EstimatorError as exc:
        # trainer-config is only produced for a `ready` result; say why it is not.
        raise ApiError(409, exc.issue) from None
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete("/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, state: StateDep, db: DbDep, owner: OwnerDep) -> Response:
    """Delete the analysis, its events and artifacts, and uploads only it referenced."""
    analysis = _owned(db, owner, analysis_id)
    if not store.is_terminal(analysis):
        try:
            if not jobs.cancel_queued_job(state.redis, analysis.id, analysis.attempt):
                jobs.stop_running_job(state.redis, analysis.id, analysis.attempt)
        except RedisError:
            log.warning("could not reach redis while deleting %s", analysis.id)
    pending = store.delete_analysis(db, analysis)
    db.commit()
    pending.apply(state.settings)
    return Response(status_code=204)


@router.post(
    "/{analysis_id}/profile",
    status_code=503,
    responses={503: {"model": ErrorResponse, "description": "GPU worker not connected"}},
)
def start_gpu_profile(analysis_id: str, db: DbDep, owner: OwnerDep) -> Response:
    """Opt-in GPU validation (M5). Not available in this release: always GPU_WORKER_UNAVAILABLE."""
    _owned(db, owner, analysis_id)
    raise api_error(
        503,
        ErrorCode.GPU_WORKER_UNAVAILABLE,
        "GPU 검증 워커가 연결되어 있지 않습니다(GPU 검증 미연결). "
        "정적 분석 결과는 그대로 사용할 수 있습니다.",
    )
