"""Per-application runtime state: settings, DB sessions, Redis and the RQ queue.

Everything is created lazily so importing the app (e.g. to export the OpenAPI document) never
connects to PostgreSQL or Redis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from redis import Redis
from rq import Queue
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from .db import get_engine, session_factory
from .jobs import make_queue
from .settings import Settings


@dataclass
class AppState:
    settings: Settings
    redis_client: Redis | None = None
    # False runs jobs synchronously at enqueue time (tests only).
    queue_is_async: bool = True
    _sessions: sessionmaker[Session] | None = field(default=None, repr=False)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def engine(self) -> Engine:
        return get_engine(self.settings.database_url)

    @property
    def sessions(self) -> sessionmaker[Session]:
        if self._sessions is None or self._sessions.kw.get("bind") is not self.engine:
            self._sessions = session_factory(self.engine)
        return self._sessions

    @property
    def redis(self) -> Redis:
        if self.redis_client is None:
            self.redis_client = Redis.from_url(
                self.settings.redis_url, socket_timeout=5, socket_connect_timeout=3
            )
        return self.redis_client

    def queue(self) -> Queue:
        return make_queue(self.redis, self.settings.queue_name, is_async=self.queue_is_async)


__all__ = ["AppState"]
