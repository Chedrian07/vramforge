"""Reaper (expired leases, lost jobs, cancel escalation), retention and the healthcheck."""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import fakeredis
import pytest
from rq import Worker
from rq.job import Job
from rq.serializers import JSONSerializer
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker
from worker_testkit import (
    OWNER,
    create_analysis,
    event_types,
    get,
    lease_values,
    load_pipeline_fakes,
)

from vramforge_api import jobs
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis, ScanCacheEntry, Upload
from vramforge_api.settings import Settings
from vramforge_worker import healthcheck, main
from vramforge_worker.lease import Heartbeat, LeaseState, acquire_lease
from vramforge_worker.reaper import reap
from vramforge_worker.retention import cleanup_expired


def _queue(redis: fakeredis.FakeRedis):
    return jobs.make_queue(redis, "analyses")


def _expire(sessions: sessionmaker[Session], aid: str) -> None:
    with session_scope(sessions) as db:
        db.execute(
            update(Analysis)
            .where(Analysis.id == aid)
            .values(lease_expires_at=utcnow() - timedelta(seconds=1))
        )


def test_expired_lease_is_requeued_as_a_new_attempt(
    settings: Settings, sessions: sessionmaker[Session], fake_redis: fakeredis.FakeRedis
) -> None:
    aid = create_analysis(sessions, status="TOKENIZING", **lease_values("dead-worker"))
    _expire(sessions, aid)
    report = reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert report.requeued == [aid]
    row = get(sessions, aid)
    assert (row.status, row.attempt, row.lease_owner) == ("QUEUED", 2, None)
    job = Job.fetch(f"{aid}-a2", connection=fake_redis, serializer=JSONSerializer)
    assert job.args == [aid, 2]
    assert job.timeout == settings.job_timeout_s
    assert event_types(sessions, aid)[-1] == ("progress", "QUEUED")


def test_never_two_runners(sessions: sessionmaker[Session], settings, fake_redis) -> None:
    aid = create_analysis(sessions)
    assert acquire_lease(sessions, aid, 1, "runner-1", ttl_s=30)
    state = LeaseState()
    old = Heartbeat(sessions, aid, "runner-1", state, ttl_s=30, poll_s=1)
    _expire(sessions, aid)
    reap(settings, sessions, fake_redis, _queue(fake_redis))
    # the stale runner notices on its next heartbeat and cannot win the new attempt
    old.beat(extend=True)
    assert state.lease_lost is True
    assert not acquire_lease(sessions, aid, 1, "runner-1", ttl_s=30)
    assert acquire_lease(sessions, aid, 2, "runner-2", ttl_s=30)
    assert not acquire_lease(sessions, aid, 2, "runner-3", ttl_s=30)


def test_last_attempt_fails_instead_of_retrying(
    settings: Settings, sessions: sessionmaker[Session], fake_redis
) -> None:
    aid = create_analysis(sessions, status="TOKENIZING", attempt=3, **lease_values())
    _expire(sessions, aid)
    report = reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert report.failed == [aid]
    row = get(sessions, aid)
    assert row.status == "FAILED" and row.error["details"]["attempts"] == 3
    assert not Job.exists(f"{aid}-a4", connection=fake_redis)


def test_expired_lease_with_cancel_request_is_cancelled(
    settings: Settings, sessions: sessionmaker[Session], fake_redis
) -> None:
    aid = create_analysis(
        sessions, status="CANCEL_REQUESTED", cancel_requested=True, **lease_values()
    )
    _expire(sessions, aid)
    reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert get(sessions, aid).status == "CANCELLED"


def test_lost_queued_job_is_reenqueued(settings_factory, fake_redis: fakeredis.FakeRedis) -> None:
    settings = settings_factory(queued_requeue_after_s=1)
    from vramforge_api.db import get_engine, session_factory

    sessions = session_factory(get_engine(settings.database_url))
    old = utcnow() - timedelta(minutes=5)
    missing = create_analysis(sessions, updated_at=old)
    report = reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert report.reenqueued == [missing]
    assert Job.exists(f"{missing}-a1", connection=fake_redis)

    # an RQ job that ended without running the analysis moves to the next attempt
    ended = create_analysis(sessions, updated_at=old)
    jobs.enqueue_analysis(_queue(fake_redis), ended, 1, 60)
    jobs.cancel_queued_job(fake_redis, ended, 1)
    report = reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert report.requeued == [ended]
    assert get(sessions, ended).attempt == 2

    fresh = create_analysis(sessions)  # recently queued: left alone
    reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert not Job.exists(f"{fresh}-a1", connection=fake_redis)


def test_cancel_escalation_and_orphaned_cancels(
    settings: Settings, sessions: sessionmaker[Session], fake_redis, monkeypatch
) -> None:
    stops: list[str] = []
    monkeypatch.setattr(jobs, "stop_running_job", lambda r, a, n: stops.append(a) or True)
    long_ago = utcnow() - timedelta(minutes=5)
    running = create_analysis(
        sessions,
        status="CANCEL_REQUESTED",
        cancel_requested=True,
        cancel_requested_at=long_ago,
        **lease_values(),
    )
    orphan = create_analysis(
        sessions, status="CANCEL_REQUESTED", cancel_requested=True, cancel_requested_at=long_ago
    )
    report = reap(settings, sessions, fake_redis, _queue(fake_redis))
    assert report.stop_sent == [running] and stops == [running]
    assert report.cancelled == [orphan]
    assert get(sessions, orphan).status == "CANCELLED"


def test_retention_deletes_old_analyses_uploads_cache_and_orphans(
    settings: Settings, sessions: sessionmaker[Session]
) -> None:
    long_ago = utcnow() - timedelta(days=settings.retention_days + 1)
    old = create_analysis(sessions, status="COMPLETED", finished_at=long_ago)
    recent = create_analysis(sessions, status="COMPLETED", finished_at=utcnow())
    for aid in (old, recent):
        path = settings.data_dir / "artifacts" / OWNER / aid / "lengths"
        path.mkdir(parents=True)
        (path / "part-00000.parquet").write_bytes(b"PAR1")

    upload_dir = settings.uploads_dir / OWNER / ("c" * 32)
    upload_dir.mkdir(parents=True)
    (upload_dir / "d.jsonl").write_text("{}\n")
    in_use_dir = settings.uploads_dir / OWNER / ("d" * 32)
    in_use_dir.mkdir(parents=True)
    (in_use_dir / "d.jsonl").write_text("{}\n")
    running_request = load_pipeline_fakes().example_request(
        **{"dataset.source_type": "upload", "dataset.reference": "upload:" + "d" * 32}
    )
    create_analysis(sessions, status="TOKENIZING", request=running_request)
    with session_scope(sessions) as db:
        for uid in ("c" * 32, "d" * 32):
            db.add(
                Upload(
                    id=uid,
                    owner_id=OWNER,
                    filename="d.jsonl",
                    format="jsonl",
                    size_bytes=3,
                    sha256="0" * 64,
                    path=f"uploads/{OWNER}/{uid}/d.jsonl",
                    expires_at=utcnow() - timedelta(seconds=1),
                )
            )
        db.add(
            ScanCacheEntry(
                owner_id=OWNER,
                preprocess_key="pre_gone",
                artifact_path=f"artifacts/{OWNER}/{'e' * 32}/lengths",
                summary={},
            )
        )
    orphan = settings.data_dir / "artifacts" / OWNER / ("9" * 32)
    orphan.mkdir(parents=True)
    stamp = time.time() - 3 * 86_400
    os.utime(orphan, (stamp, stamp))
    unrelated = settings.data_dir / "artifacts" / "not-an-owner"
    unrelated.mkdir(parents=True)

    report = cleanup_expired(settings, sessions)
    assert report.analyses == [old]
    assert get(sessions, old) is None and get(sessions, recent) is not None
    assert not (settings.data_dir / "artifacts" / OWNER / old).exists()
    assert (settings.data_dir / "artifacts" / OWNER / recent).exists()
    assert report.uploads == ["c" * 32]
    assert not upload_dir.exists() and in_use_dir.exists()
    assert report.cache_entries == 1
    assert report.orphan_dirs == 1 and not orphan.exists()
    assert unrelated.exists()  # only the service's own layout is ever deleted


def test_healthcheck_requires_a_fresh_local_worker(
    settings: Settings, fake_redis: fakeredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert healthcheck.check(settings, redis=fake_redis) is False
    worker = Worker(["analyses"], connection=fake_redis, serializer=JSONSerializer)
    worker.register_birth()
    worker.heartbeat()
    assert healthcheck.check(settings, redis=fake_redis, hostname=socket.gethostname()) is True
    assert healthcheck.check(settings, redis=fake_redis, hostname="another-host") is False

    monkeypatch.setattr(healthcheck, "check", lambda s: True)
    with pytest.raises(SystemExit) as ok:
        main.main(["healthcheck"])
    assert ok.value.code == 0
    monkeypatch.setattr(healthcheck, "check", lambda s: False)
    with pytest.raises(SystemExit) as bad:
        main.main(["healthcheck"])
    assert bad.value.code == 1


HEAVY_MODULES = (
    "sqlalchemy",
    "alembic",
    "vramforge_api.db",
    "vramforge_estimator.pipeline",
    "vramforge_estimator.inspection",
    "vramforge_worker.tasks",
    "transformers",
    "tokenizers",
    "datasets",
    "pyarrow",
    "torch",
)


def _loaded_after(code: str) -> set[str]:
    import json
    import subprocess
    import sys

    script = f"import json, sys\n{code}\nprint(json.dumps(sorted(sys.modules)))\n"
    env = {**os.environ, "VRAMFORGE_REDIS_URL": "redis://127.0.0.1:1/0"}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
    )
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_healthcheck_imports_only_what_it_needs() -> None:
    """The compose healthcheck runs the CLI every few seconds: no database layer, estimator or
    tokenizer stack may be imported for it (lazy imports in main.py)."""
    cli = _loaded_after("import vramforge_worker.main")
    probe = _loaded_after(
        "from vramforge_worker import main\n"
        "try:\n    main.main(['healthcheck'])\nexcept SystemExit as exc:\n"
        "    assert exc.code == 1, exc.code"
    )
    for loaded in (cli, probe):
        heavy = sorted(
            m for m in loaded if any(m == h or m.startswith(f"{h}.") for h in HEAVY_MODULES)
        )
        assert heavy == [], heavy
    assert "redis" not in cli  # nothing beyond argparse until a command runs
    assert {"redis", "vramforge_api.settings"} <= probe


def test_worker_burst_runs_queued_jobs_with_json_serializer(
    settings: Settings, sessions: sessionmaker[Session], fake_redis, monkeypatch
) -> None:
    from vramforge_worker.worker import VramforgeWorker

    ran: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "vramforge_worker.tasks.run_analysis", lambda a, n: ran.append((a, n)) or "COMPLETED"
    )
    aid = create_analysis(sessions)
    jobs.enqueue_analysis(_queue(fake_redis), aid, 1, 60)
    worker = VramforgeWorker(
        [_queue(fake_redis)], connection=fake_redis, serializer=JSONSerializer, worker_ttl=60
    )
    worker.vf_settings, worker.vf_sessions = settings, sessions
    from rq.worker import SimpleWorker

    # fork-free execution for the test; production uses the forking VramforgeWorker
    simple = SimpleWorker([_queue(fake_redis)], connection=fake_redis, serializer=JSONSerializer)
    simple.work(burst=True)
    assert ran == [(aid, 1)]
    worker.run_maintenance_tasks()  # reaper + retention hooks run without errors


def test_status_expiry_matches_what_retention_deletes(settings, sessions) -> None:
    """`AnalysisStatus.expires_at` (store.expires_at) is the retention rule the cleanup applies."""
    from vramforge_api import store

    now = utcnow()
    old = create_analysis(
        sessions, status="COMPLETED", finished_at=now - timedelta(days=7, hours=1)
    )
    fresh = create_analysis(sessions, status="FAILED", finished_at=now - timedelta(days=6))
    running = create_analysis(sessions, status="TOKENIZING", created_at=now - timedelta(days=30))
    with session_scope(sessions) as db:
        expiry = {
            aid: store.expires_at(db.get(Analysis, aid), settings.retention_days)
            for aid in (old, fresh, running)
        }
    assert expiry[old] < now < expiry[fresh]
    assert expiry[running] is None
    report = cleanup_expired(settings, sessions)
    assert report.analyses == [old]
    assert get(sessions, fresh) is not None and get(sessions, running) is not None
