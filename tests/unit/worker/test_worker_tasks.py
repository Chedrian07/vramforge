"""run_analysis: lease, context, persistence, cancellation and failure handling."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker
from worker_testkit import (
    OTHER_OWNER,
    OWNER,
    create_analysis,
    event_types,
    get,
    lease_values,
    load_pipeline_fakes,
)

from vramforge_api.db import session_scope
from vramforge_api.models import Analysis, ScanCacheEntry
from vramforge_api.settings import Settings
from vramforge_estimator import pipeline
from vramforge_estimator.errors import CancelledError
from vramforge_estimator.schemas import (
    AnalysisResult,
    ErrorCode,
    Issue,
    JobProgress,
    JobStatus,
    Stage,
)
from vramforge_worker import tasks

fakes = load_pipeline_fakes()


def test_completed_run_persists_result_events_and_cache(
    settings: Settings, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes.FakeModules().install(monkeypatch)
    aid = create_analysis(sessions)
    assert tasks.run_analysis(aid, 1) == "COMPLETED"

    row = get(sessions, aid)
    assert row.status == "COMPLETED"
    assert row.lease_owner is None and row.finished_at is not None and row.started_at
    assert row.result["analysis_id"] == aid
    assert row.preprocess_key and row.preprocess_key.startswith("pre_")
    assert row.error is None
    types = event_types(sessions, aid)
    stages = [s for t, s in types if t == "progress"]
    assert stages[:6] == [
        "RESOLVING",
        "INSPECTING",
        "TOKENIZING",
        "VALIDATING_DATA",
        "PLANNING_BATCHES",
        "ESTIMATING",
    ]
    assert ("partial_result", "INSPECTING") in types
    assert types[-1] == ("completed", "COMPLETED")
    # artifacts written by the pipeline live under the owner's artifact directory
    artifacts = settings.data_dir / row.artifact_dir
    assert (artifacts / pipeline.INVENTORY_FILE).is_file()
    assert (artifacts / "lengths").is_dir()
    with session_scope(sessions) as db:
        (entry,) = db.query(ScanCacheEntry).all()
        assert entry.owner_id == OWNER and entry.analysis_id == aid
        assert entry.artifact_path == f"{row.artifact_dir}/lengths"


def test_scan_cache_is_owner_scoped(
    settings: Settings, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = fakes.FakeModules().install(monkeypatch)
    first = create_analysis(sessions)
    tasks.run_analysis(first, 1)
    assert modules.calls.count("full_scan") == 1

    same_owner = create_analysis(sessions)
    assert tasks.run_analysis(same_owner, 1) == "COMPLETED"
    assert modules.calls.count("full_scan") == 1  # reused, not re-tokenized

    other_owner = create_analysis(sessions, owner=OTHER_OWNER)
    assert tasks.run_analysis(other_owner, 1) == "COMPLETED"
    assert modules.calls.count("full_scan") == 2  # never shared across owners


def test_live_lease_elsewhere_is_skipped(sessions: sessionmaker[Session], monkeypatch) -> None:
    modules = fakes.FakeModules().install(monkeypatch)
    aid = create_analysis(sessions, status=JobStatus.TOKENIZING.value, **lease_values())
    assert tasks.run_analysis(aid, 1) == "skipped"
    assert get(sessions, aid).lease_owner == "other-worker"
    assert modules.calls == []


def test_stale_attempt_and_cancelled_jobs_are_skipped(
    sessions: sessionmaker[Session], monkeypatch
) -> None:
    fakes.FakeModules().install(monkeypatch)
    newer = create_analysis(sessions, attempt=2)
    assert tasks.run_analysis(newer, 1) == "skipped"
    cancelled = create_analysis(sessions, cancel_requested=True)
    assert tasks.run_analysis(cancelled, 1) == "skipped"
    finished = create_analysis(sessions, status="COMPLETED")
    assert tasks.run_analysis(finished, 1) == "skipped"
    assert tasks.run_analysis("f" * 32, 1) == "skipped"


def test_unexpected_errors_fail_without_leaking_details(
    sessions: sessionmaker[Session], monkeypatch
) -> None:
    def boom(request, ctx):
        raise RuntimeError("/srv/private/path exploded")

    monkeypatch.setattr(pipeline, "analyze", boom)
    aid = create_analysis(sessions)
    assert tasks.run_analysis(aid, 1) == "FAILED"
    row = get(sessions, aid)
    assert row.error["code"] == "INTERNAL_ERROR"
    assert "/srv/private" not in str(row.error)
    assert event_types(sessions, aid)[-1] == ("failed", "FAILED")


def test_job_timeout_is_reported(sessions: sessionmaker[Session], monkeypatch) -> None:
    from rq.timeouts import JobTimeoutException

    def slow(request, ctx):
        raise JobTimeoutException("timeout")

    monkeypatch.setattr(pipeline, "analyze", slow)
    aid = create_analysis(sessions)
    assert tasks.run_analysis(aid, 1) == "FAILED"
    assert get(sessions, aid).error["code"] == "JOB_TIMEOUT"


def test_cooperative_cancel_while_scanning(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    aid = create_analysis(sessions)

    def waiting_scan(stream, adapter, ctx, **kwargs):
        # what the API does on POST /cancel for a running job
        with session_scope(sessions) as db:
            db.execute(
                update(Analysis)
                .where(Analysis.id == aid)
                .values(cancel_requested=True, status=JobStatus.CANCEL_REQUESTED.value)
            )
        deadline = time.monotonic() + 5
        while not ctx.cancelled():
            assert time.monotonic() < deadline, "heartbeat never saw the cancel flag"
            time.sleep(0.01)
        raise CancelledError(Stage.TOKENIZING)

    fakes.FakeModules(full_scan=waiting_scan).install(monkeypatch)
    assert tasks.run_analysis(aid, 1) == "CANCELLED"
    row = get(sessions, aid)
    assert row.status == "CANCELLED"
    assert row.result["status"]["scan_coverage"] == "partial"
    assert event_types(sessions, aid)[-1] == ("cancelled", "CANCELLED")


def test_set_stage_keeps_a_pending_cancel_visible(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    aid = create_analysis(sessions)
    seen: list[str] = []

    def analyze(request, ctx):
        with session_scope(sessions) as db:
            db.execute(
                update(Analysis)
                .where(Analysis.id == aid)
                .values(cancel_requested=True, status="CANCEL_REQUESTED")
            )
        ctx.set_stage(JobStatus.TOKENIZING)
        seen.append(get(sessions, aid).status)
        raise RuntimeError("stop here")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    tasks.run_analysis(aid, 1)
    assert seen == ["CANCEL_REQUESTED"]
    tokenizing = [s for t, s in event_types(sessions, aid) if t == "progress"]
    assert tokenizing[-1] == "CANCEL_REQUESTED"  # events carry the job status


def _empty_result(request, ctx) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=ctx.analysis_id,
        created_at=datetime.now(UTC),
        analysis_fingerprint="req_x",
        estimator_version="test",
        requested_config=request,
    )


def test_deleted_while_running_discards_and_cleans_up(
    settings: Settings, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    aid = create_analysis(sessions)

    def analyze(request, ctx):
        (ctx.artifact_dir / "lengths").mkdir(parents=True)
        with session_scope(sessions) as db:  # what DELETE /analyses/{id} does meanwhile
            db.delete(db.get(Analysis, aid))
        return _empty_result(request, ctx)

    monkeypatch.setattr(pipeline, "analyze", analyze)
    rel = f"artifacts/{OWNER}/{aid}"
    assert tasks.run_analysis(aid, 1) == "discarded"
    assert get(sessions, aid) is None
    assert not (settings.data_dir / rel).exists()


def test_lost_lease_is_never_overwritten(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    aid = create_analysis(sessions)

    def analyze(request, ctx):
        # the reaper handed the job to attempt 2 meanwhile
        with session_scope(sessions) as db:
            db.execute(
                update(Analysis)
                .where(Analysis.id == aid)
                .values(attempt=2, lease_owner=None, status="QUEUED")
            )
        return _empty_result(request, ctx)

    monkeypatch.setattr(pipeline, "analyze", analyze)
    assert tasks.run_analysis(aid, 1) == "discarded"
    row = get(sessions, aid)
    assert row.status == "QUEUED" and row.attempt == 2 and row.result is None


def test_context_rate_limits_progress_and_sanitizes_partials(
    settings_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_factory(progress_event_interval_s=3600)
    from vramforge_api.db import get_engine, session_factory

    sessions = session_factory(get_engine(settings.database_url))
    aid = create_analysis(sessions)

    def analyze(request, ctx):
        ctx.set_stage(JobStatus.TOKENIZING)
        for i in range(5):
            ctx.report(
                JobProgress(stage=JobStatus.TOKENIZING, processed_rows=i, total_rows=10),
                {"max_prompt": i, "text": "x" * 500, "nested": {"a": 1}, "flag": True},
            )
        ctx.report(
            JobProgress(
                stage=JobStatus.TOKENIZING,
                processed_rows=10,
                total_rows=10,
                message_code="scan_final",
            ),
            None,
        )
        ctx.warn(Issue(code=ErrorCode.SCAN_FAILED_ROWS, severity="warning", user_message="x"))
        ctx.save_checkpoint({"rows": 10})
        assert ctx.load_checkpoint() == {"rows": 10}
        raise RuntimeError("stop")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    tasks.run_analysis(aid, 1)
    with session_scope(sessions) as db:
        from vramforge_api import store

        events = store.events_after(db, aid, 0)
    progress = [
        e
        for e in events
        if e.type.value == "progress" and e.progress and e.progress.processed_rows is not None
    ]
    # throttled: only the final report produced an event
    assert [e.progress.processed_rows for e in progress] == [10]
    assert any(e.type.value == "warning" for e in events)
    assert get(sessions, aid).progress["stage"] == "FAILED"


def test_partial_values_are_sanitized() -> None:
    from vramforge_worker.context import _clean_partial

    cleaned = _clean_partial({"a": 1, "b": "x" * 500, "c": {"nested": 1}, "d": True, "e": None})
    assert cleaned == {"a": 1, "b": "x" * 200, "d": 1, "e": None}


def test_on_stopped_marks_cancelled_or_failed(sessions: sessionmaker[Session]) -> None:
    class FakeJob:
        def __init__(self, aid: str, attempt: int) -> None:
            self.args = [aid, attempt]

    requested = create_analysis(
        sessions, cancel_requested=True, status="CANCEL_REQUESTED", **lease_values()
    )
    tasks.on_analysis_stopped(FakeJob(requested, 1), None)
    assert get(sessions, requested).status == "CANCELLED"

    killed = create_analysis(sessions, status="TOKENIZING", **lease_values())
    tasks.on_analysis_stopped(FakeJob(killed, 1), None)
    row = get(sessions, killed)
    assert row.status == "FAILED" and row.error["code"] == "INTERNAL_ERROR"

    stale = create_analysis(sessions, status="TOKENIZING", attempt=2)
    tasks.on_analysis_stopped(FakeJob(stale, 1), None)
    assert get(sessions, stale).status == "TOKENIZING"


def test_heartbeat_extends_the_lease(sessions: sessionmaker[Session]) -> None:
    from vramforge_worker.lease import Heartbeat, LeaseState, acquire_lease

    aid = create_analysis(sessions)
    assert acquire_lease(sessions, aid, 1, "me", ttl_s=30)
    assert not acquire_lease(sessions, aid, 1, "someone-else", ttl_s=30)
    before = get(sessions, aid).lease_expires_at
    state = LeaseState()
    beat = Heartbeat(sessions, aid, "me", state, ttl_s=3, poll_s=0.02)
    time.sleep(0.01)
    beat.beat(extend=True)
    assert get(sessions, aid).lease_expires_at != before
    with session_scope(sessions) as db:
        db.execute(update(Analysis).where(Analysis.id == aid).values(lease_owner="thief"))
    beat.beat(extend=True)
    assert state.lease_lost is True


def test_runner_threads_do_not_leak(sessions: sessionmaker[Session], monkeypatch) -> None:
    fakes.FakeModules().install(monkeypatch)
    before = threading.active_count()
    tasks.run_analysis(create_analysis(sessions), 1)
    assert threading.active_count() <= before


def test_artifact_dir_is_owner_scoped(settings: Settings, sessions, monkeypatch) -> None:
    captured: list[Path] = []

    def analyze(request, ctx):
        captured.append(ctx.artifact_dir)
        assert ctx.access.uploads_dir == settings.uploads_dir / OWNER
        raise RuntimeError("stop")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    aid = create_analysis(sessions)
    tasks.run_analysis(aid, 1)
    assert captured == [settings.data_dir / "artifacts" / OWNER / aid]
