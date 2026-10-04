"""SQLAlchemy engine/session helpers shared by the API and the worker.

PostgreSQL is the source of truth for job state (plan.md §16.1). Unit tests run on SQLite, so the
column types used here have SQLite-compatible variants. Engines are cached per process id: the RQ
worker forks a work-horse per job and a pooled connection must never cross a fork.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Engine, Integer, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

JsonType = JSON().with_variant(JSONB(), "postgresql")
# SQLite only auto-increments INTEGER PRIMARY KEY columns.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(
            url, connect_args={"check_same_thread": False, "timeout": 30}, future=True
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        return engine
    connect_args: dict[str, Any] = {}
    if url.startswith("postgresql"):
        connect_args["connect_timeout"] = 5
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args, future=True)


_ENGINES: dict[tuple[int, str], Engine] = {}
_LOCK = threading.Lock()


def get_engine(url: str) -> Engine:
    """Engine for the current process (a forked child never reuses the parent's pool)."""
    key = (os.getpid(), url)
    with _LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = make_engine(url)
            _ENGINES[key] = engine
        return engine


def dispose_engines() -> None:
    with _LOCK:
        for (pid, _url), engine in list(_ENGINES.items()):
            if pid == os.getpid():
                engine.dispose()
        _ENGINES.clear()


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit on success, roll back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "Base",
    "BigIntPK",
    "JsonType",
    "UTCDateTime",
    "dispose_engines",
    "get_engine",
    "make_engine",
    "session_factory",
    "session_scope",
    "utcnow",
]
