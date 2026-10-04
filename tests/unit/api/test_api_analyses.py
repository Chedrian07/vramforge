"""Analysis job API: creation, idempotency, limits, owner isolation, cancel, delete, SSE."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta

import fakeredis
import pytest
from api_testkit import example_request
from fastapi.testclient import TestClient
from rq.job import Job
from rq.serializers import JSONSerializer

from vramforge_api import jobs, store
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis
from vramforge_api.settings import Settings
from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import ErrorCode, EventType, JobProgress, JobStatus, Severity


def _create(client: TestClient, key: str | None = None, **overrides) -> dict:
    headers = {"Idempotency-Key": key} if key else {}
    resp = client.post("/api/v1/analyses", json=example_request(**overrides), headers=headers)
    assert resp.status_code == 202, resp.text
    return resp.json()


def _sessions(client: TestClient):
    return client.app.state.vf.sessions  # type: ignore[attr-defined]


def test_create_enqueues_a_json_serialized_job(
    client: TestClient, settings: Settings, fake_redis: fakeredis.FakeRedis
) -> None:
    created = _create(client)
    assert created["status"] == "QUEUED"
    assert created["reused"] is False
    assert created["fingerprint"].startswith("req_")
    job = Job.fetch(
        f"{created['analysis_id']}-a1", connection=fake_redis, serializer=JSONSerializer
    )
    assert job.func_name == jobs.TASK_PATH
    assert job.args == [created["analysis_id"], 1]
    assert job.timeout == settings.job_timeout_s

    status = client.get(f"/api/v1/analyses/{created['analysis_id']}").json()
    assert status["status"] == "QUEUED"
    assert status["last_event_id"] is not None
    assert status["result"] is None


def test_idempotency_key_reuses_the_analysis(client: TestClient) -> None:
    first = _create(client, key="k-1")
    second = _create(client, key="k-1")
    assert second["analysis_id"] == first["analysis_id"]
    assert second["reused"] is True
    other = client.post(
        "/api/v1/analyses",
        json=example_request(**{"training.gradient_accumulation_steps": 8}),
        headers={"Idempotency-Key": "k-1"},
    )
    assert other.status_code == 409


def test_concurrency_limit_is_per_owner(
    settings: Settings, client_factory: Callable[..., TestClient]
) -> None:
    alice = client_factory(settings)
    bob = client_factory(settings)
    _create(alice)
    _create(alice)
    third = alice.post("/api/v1/analyses", json=example_request())
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "CONCURRENCY_LIMIT"
    _create(bob)


def test_compatibility_errors_block_creation(client: TestClient, monkeypatch) -> None:
    from vramforge_estimator import compatibility

    issue = make_issue(
        ErrorCode.CONFLICTING_OPTIONS, "4-bit과 전체 미세조정은 함께 쓸 수 없습니다."
    )
    warn = make_issue(ErrorCode.GRPO_REWARD_UNSPECIFIED, "reward 미지정", severity=Severity.WARNING)
    monkeypatch.setattr(compatibility, "validate_request", lambda r: [warn, issue])
    resp = client.post("/api/v1/analyses", json=example_request())
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "CONFLICTING_OPTIONS"
    assert len(body["issues"]) == 2


def test_enqueue_failure_marks_the_analysis_failed(client: TestClient, monkeypatch) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    def down(*args, **kwargs):
        raise RedisConnectionError("down")

    monkeypatch.setattr(jobs, "enqueue_analysis", down)
    resp = client.post("/api/v1/analyses", json=example_request())
    assert resp.status_code == 503
    assert resp.json()["error"]["retryable"] is True
    with session_scope(_sessions(client)) as db:
        (row,) = db.query(Analysis).all()
        assert row.status == "FAILED"


def test_other_owners_get_404_everywhere(
    settings: Settings, client_factory: Callable[..., TestClient]
) -> None:
    alice = client_factory(settings)
    mallory = client_factory(settings)
    aid = _create(alice)["analysis_id"]
    for method, url, body in (
        ("GET", f"/api/v1/analyses/{aid}", None),
        ("GET", f"/api/v1/analyses/{aid}/events", None),
        ("GET", f"/api/v1/analyses/{aid}/export?format=json", None),
        ("POST", f"/api/v1/analyses/{aid}/cancel", None),
        ("POST", f"/api/v1/analyses/{aid}/scenarios", {"request": example_request()}),
        ("POST", f"/api/v1/analyses/{aid}/scenarios/export", {"request": example_request()}),
        ("POST", f"/api/v1/analyses/{aid}/profile", None),
        ("DELETE", f"/api/v1/analyses/{aid}", None),
    ):
        resp = mallory.request(method, url, json=body)
        assert resp.status_code == 404, (method, url, resp.text)
        assert resp.json()["error"]["code"] == "NOT_FOUND"
    assert alice.get(f"/api/v1/analyses/{aid}").json()["status"] == "QUEUED"


def test_malformed_ids_are_not_found(client: TestClient) -> None:
    assert client.get("/api/v1/analyses/../../etc").status_code == 404
    assert client.get("/api/v1/analyses/XYZ").status_code == 404


def test_cancel_queued_job(client: TestClient, fake_redis: fakeredis.FakeRedis) -> None:
    aid = _create(client)["analysis_id"]
    resp = client.post(f"/api/v1/analyses/{aid}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "CANCELLED"
    assert body["error"]["code"] == "CANCELLED"
    job = Job.fetch(f"{aid}-a1", connection=fake_redis, serializer=JSONSerializer)
    assert job.get_status() == "canceled"
    with session_scope(_sessions(client)) as db:
        events = store.events_after(db, aid, 0)
    assert events[-1].type is EventType.CANCELLED
    # cancelling again is a no-op on a terminal analysis
    assert client.post(f"/api/v1/analyses/{aid}/cancel").json()["status"] == "CANCELLED"


def test_cancel_running_job_requests_cooperative_stop(client: TestClient, monkeypatch) -> None:
    stops: list[tuple[str, int]] = []
    monkeypatch.setattr(jobs, "stop_running_job", lambda r, a, n: stops.append((a, n)) or True)
    aid = _create(client)["analysis_id"]
    with session_scope(_sessions(client)) as db:
        row = db.get(Analysis, aid)
        row.status = JobStatus.TOKENIZING.value
        row.lease_owner = "worker-1"
        row.lease_expires_at = utcnow() + timedelta(seconds=60)
    body = client.post(f"/api/v1/analyses/{aid}/cancel").json()
    assert body["status"] == "CANCEL_REQUESTED"
    with session_scope(_sessions(client)) as db:
        row = db.get(Analysis, aid)
        assert row.cancel_requested is True and row.cancel_requested_at is not None
    assert stops == [(aid, 1)]  # grace period is 0 in tests


def test_delete_removes_rows_and_artifacts(client: TestClient, settings: Settings) -> None:
    aid = _create(client)["analysis_id"]
    with session_scope(_sessions(client)) as db:
        rel = db.get(Analysis, aid).artifact_dir
    artifact_dir = settings.data_dir / rel
    (artifact_dir / "lengths").mkdir(parents=True)
    (artifact_dir / "lengths" / "x.parquet").write_bytes(b"PAR1")
    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 204
    assert not artifact_dir.exists()
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404
    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 404


def test_gpu_profile_is_unavailable(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    resp = client.post(f"/api/v1/analyses/{aid}/profile")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "GPU_WORKER_UNAVAILABLE"
    assert client.post("/api/v1/analyses/" + "f" * 32 + "/profile").status_code == 404


def test_scenarios_require_a_finished_analysis(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    resp = client.post(f"/api/v1/analyses/{aid}/scenarios", json={"request": example_request()})
    assert resp.status_code == 409
    client.post(f"/api/v1/analyses/{aid}/cancel")
    resp = client.post(
        f"/api/v1/analyses/{aid}/scenarios",
        json={"request": example_request(), "client_fingerprint": "ui-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_reanalysis"] is True
    assert body["client_fingerprint"] == "ui-1"
    assert body["reanalysis_reasons"]


def test_export_without_result_is_a_conflict(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    resp = client.get(f"/api/v1/analyses/{aid}/export?format=md")
    assert resp.status_code == 409
    assert client.get(f"/api/v1/analyses/{aid}/export?format=pdf").status_code == 422


def _add_events(client: TestClient, aid: str, terminal: bool) -> list[int]:
    ids = []
    with session_scope(_sessions(client)) as db:
        row = db.get(Analysis, aid)
        for stage in (JobStatus.RESOLVING, JobStatus.INSPECTING):
            ids.append(
                store.append_event(
                    db,
                    analysis_id=aid,
                    fingerprint=row.fingerprint,
                    event_type=EventType.PROGRESS,
                    status=stage,
                    progress=JobProgress(stage=stage),
                )
            )
        if terminal:
            store.finish(db, row, JobStatus.COMPLETED)
    with session_scope(_sessions(client)) as db:
        ids.append(store.last_event_id(db, aid))
    return ids


def _read_sse(resp) -> list[dict]:
    events, current = [], {}
    for line in resp.iter_lines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(": ")
        current[key] = value
    if current:
        events.append(current)
    return [e for e in events if "id" in e]


def test_sse_replays_after_last_event_id_and_closes_on_terminal(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    ids = _add_events(client, aid, terminal=True)
    with client.stream(
        "GET", f"/api/v1/analyses/{aid}/events", headers={"Last-Event-ID": str(ids[0])}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"].startswith("no-cache")
        events = _read_sse(resp)
    assert [int(e["id"]) for e in events] == ids[1:]
    assert [e["event"] for e in events] == ["progress", "completed"]
    payload = json.loads(events[-1]["data"])
    assert payload["analysis_id"] == aid and payload["status"] == "COMPLETED"

    # query-parameter resume (EventSource cannot set headers on the first connect)
    with client.stream("GET", f"/api/v1/analyses/{aid}/events?after={ids[1]}") as resp:
        assert [e["event"] for e in _read_sse(resp)] == ["completed"]

    done = client.get(f"/api/v1/analyses/{aid}/events", headers={"Last-Event-ID": str(ids[-1])})
    assert done.status_code == 204


def test_sse_full_replay_from_start(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    _add_events(client, aid, terminal=True)
    with client.stream("GET", f"/api/v1/analyses/{aid}/events") as resp:
        events = _read_sse(resp)
    assert events[0]["event"] == "progress"
    assert json.loads(events[0]["data"])["status"] == "QUEUED"
    assert events[-1]["event"] == "completed"


def test_sse_unknown_id_is_a_plain_404(client: TestClient) -> None:
    resp = client.get("/api/v1/analyses/" + "e" * 32 + "/events")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("ref", ["upload:" + "a" * 32, "upload:../../x"])
def test_foreign_or_missing_upload_reference_is_rejected(client: TestClient, ref: str) -> None:
    resp = client.post(
        "/api/v1/analyses",
        json=example_request(**{"dataset.source_type": "upload", "dataset.reference": ref}),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "SOURCE_NOT_FOUND"


def test_unregistered_local_root_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/analyses",
        json=example_request(
            **{"model.source_type": "local", "model.reference": "local:other/models/x"}
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "LOCAL_PATH_NOT_ALLOWED"


def test_sse_of_a_finished_analysis_never_waits_for_more(client: TestClient) -> None:
    """A finished analysis whose cursor is past its terminal event (e.g. a late event landed after
    it) must close the stream instead of polling until sse_max_duration_s."""
    import time

    aid = _create(client)["analysis_id"]
    terminal_id = _add_events(client, aid, terminal=True)[-1]
    with session_scope(_sessions(client)) as db:
        late = store.append_event(
            db,
            analysis_id=aid,
            fingerprint="req_x",
            event_type=EventType.WARNING,
            status=JobStatus.TOKENIZING,
        )
    started = time.monotonic()
    with client.stream(
        "GET", f"/api/v1/analyses/{aid}/events", headers={"Last-Event-ID": str(terminal_id)}
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse(resp)
    assert [int(e["id"]) for e in events] == [late]
    assert time.monotonic() - started < 2  # sse_max_duration_s is 5 s in these tests


def test_status_echoes_the_request_and_the_retention_deadline(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    from vramforge_estimator.schemas import AnalysisRequest

    client = client_factory(settings_factory(retention_days=3))
    body = example_request(**{"training.lora.r": 32})
    aid = _create(client, **{"training.lora.r": 32})["analysis_id"]
    queued = client.get(f"/api/v1/analyses/{aid}").json()
    assert queued["expires_at"] is None  # still running: retention does not apply yet
    # the stored request, so the UI can restore the form after a reload
    assert queued["request"] == AnalysisRequest.model_validate(body).model_dump(mode="json")

    cancelled = client.post(f"/api/v1/analyses/{aid}/cancel").json()
    finished = datetime.fromisoformat(cancelled["finished_at"])
    assert datetime.fromisoformat(cancelled["expires_at"]) == finished + timedelta(days=3)
    again = client.get(f"/api/v1/analyses/{aid}").json()
    assert again["expires_at"] == cancelled["expires_at"]
    assert again["request"]["training"]["lora"]["r"] == 32


def test_status_survives_a_stored_request_from_an_older_schema(client: TestClient) -> None:
    aid = _create(client)["analysis_id"]
    with session_scope(_sessions(client)) as db:
        db.get(Analysis, aid).request = {"schema_version": "0.1", "legacy": True}
    resp = client.get(f"/api/v1/analyses/{aid}")
    assert resp.status_code == 200
    assert resp.json()["request"] is None
