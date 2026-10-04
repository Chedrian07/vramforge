"""Health is a liveness probe (always 200) whose components say what is degraded."""

from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from rq import Worker
from rq.serializers import JSONSerializer

from vramforge_api.jobs import worker_heartbeat_state


def test_health_reports_components(client: TestClient, fake_redis: fakeredis.FakeRedis) -> None:
    body = client.get("/api/v1/health").json()
    assert body["components"]["db"] == "ok"
    assert body["components"]["redis"] == "ok"
    assert body["components"]["worker"] == "none"
    assert body["components"]["gpu_worker"] == "not_connected"
    assert body["status"] == "degraded"

    worker = Worker(["analyses"], connection=fake_redis, serializer=JSONSerializer)
    worker.register_birth()
    worker.heartbeat()
    assert worker_heartbeat_state(fake_redis) == "ok"
    assert worker_heartbeat_state(fake_redis, hostname="some-other-host") == "none"
    body = client.get("/api/v1/health").json()
    assert body["components"]["worker"] == "ok"
    assert body["status"] == "ok"


def test_health_survives_unreachable_dependencies(tmp_path) -> None:
    from vramforge_api.app import create_app
    from vramforge_api.settings import Settings

    broken = Settings(
        database_url="postgresql+psycopg://x:y@127.0.0.1:1/none",
        redis_url="redis://127.0.0.1:1/0",
        data_dir=tmp_path,
    )
    with TestClient(create_app(broken)) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["components"]["db"] in ("error", "timeout")
    assert body["components"]["redis"] in ("error", "timeout")
    assert body["components"]["worker"] == "unknown"
