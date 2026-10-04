"""API test fixtures: SQLite database, fakeredis, and owner-scoped TestClients.

The settings are also exported to the environment so code that reads `get_settings()` (the worker
task when the queue runs synchronously) sees the same database and data directory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from api_testkit import CSRF
from fastapi.testclient import TestClient

from vramforge_api.app import create_app
from vramforge_api.db import Base, dispose_engines, get_engine
from vramforge_api.settings import Settings, reset_settings_cache


@pytest.fixture
def settings_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Settings]]:
    def make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "database_url": f"sqlite:///{tmp_path / 'api.db'}",
            "redis_url": "redis://fakeredis:6379/0",
            "data_dir": tmp_path / "data",
            "local_roots": f"local={tmp_path / 'local'}",
            "cancel_grace_s": 0,
            "sse_poll_interval_s": 0.01,
            "sse_keepalive_s": 0.2,
            "sse_max_duration_s": 5,
            "progress_event_interval_s": 0,
        }
        values.update(overrides)
        settings = Settings(**values)
        for key, value in values.items():
            if value is None:
                continue
            monkeypatch.setenv(f"VRAMFORGE_{key.upper()}", str(value))
        reset_settings_cache()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(get_engine(settings.database_url))
        return settings

    yield make
    reset_settings_cache()
    dispose_engines()


@pytest.fixture
def settings(settings_factory: Callable[..., Settings]) -> Settings:
    return settings_factory()


@pytest.fixture
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


@pytest.fixture
def client_factory(
    fake_redis: fakeredis.FakeRedis,
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(
        settings: Settings, *, queue_is_async: bool = True, csrf: bool = True, **kwargs: Any
    ) -> TestClient:
        app = create_app(settings, redis=fake_redis, queue_is_async=queue_is_async)
        client = TestClient(app, headers=CSRF if csrf else None, **kwargs)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(settings: Settings, client_factory: Callable[..., TestClient]) -> TestClient:
    return client_factory(settings)


@pytest.fixture(autouse=True)
def _no_validation_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """compatibility.validate_request is owned by another module; default to 'no issues'."""
    from vramforge_estimator import compatibility

    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
