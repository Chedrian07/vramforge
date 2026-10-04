"""Security test fixtures (same stack as the integration tests)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from security_testkit import kit

from vramforge_api.app import create_app
from vramforge_api.db import dispose_engines
from vramforge_api.settings import Settings, reset_settings_cache


@pytest.fixture
def settings_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Settings]]:
    def make(**overrides: Any) -> Settings:
        settings: Settings = kit.make_settings(tmp_path, monkeypatch, **overrides)
        return settings

    yield make
    reset_settings_cache()
    dispose_engines()


@pytest.fixture
def settings(settings_factory: Callable[..., Settings]) -> Settings:
    return settings_factory()


@pytest.fixture
def client_factory() -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []
    redis = fakeredis.FakeRedis()

    def make(settings: Settings, *, queue_is_async: bool = False, **kwargs: Any) -> TestClient:
        app = create_app(settings, redis=redis, queue_is_async=queue_is_async)
        client = TestClient(app, headers=kit.CSRF, **kwargs)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    fakes = kit.load_pipeline_fakes()
    from vramforge_estimator import compatibility

    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    return fakes.FakeModules().install(monkeypatch)
