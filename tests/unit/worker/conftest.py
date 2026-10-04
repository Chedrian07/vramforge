"""Worker test fixtures: SQLite database, data directory and settings mirrored to the env."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api.db import Base, dispose_engines, get_engine, session_factory
from vramforge_api.settings import Settings, reset_settings_cache


@pytest.fixture
def settings_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Settings]]:
    def make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "database_url": f"sqlite:///{tmp_path / 'worker.db'}",
            "redis_url": "redis://fakeredis:6379/0",
            "data_dir": tmp_path / "data",
            "cancel_poll_s": 0.02,
            "cancel_grace_s": 0,
            "lease_ttl_s": 30,
            "progress_event_interval_s": 0,
        }
        values.update(overrides)
        for key, value in values.items():
            monkeypatch.setenv(f"VRAMFORGE_{key.upper()}", str(value))
        reset_settings_cache()
        settings = Settings(**values)
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
def sessions(settings: Settings) -> sessionmaker[Session]:
    return session_factory(get_engine(settings.database_url))


@pytest.fixture
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()
