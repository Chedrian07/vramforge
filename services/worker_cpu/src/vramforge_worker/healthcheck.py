"""`vramforge-worker healthcheck`: exit 0 iff a worker on this host heartbeated recently.

Used as the compose healthcheck; the worker runs with `worker_ttl=60` so an idle worker
heartbeats at least every ~45 s (docs/research/stack-compat.md §10.8, E16).
"""

from __future__ import annotations

import socket

from redis import Redis
from redis.exceptions import RedisError

from vramforge_api.jobs import worker_heartbeat_state
from vramforge_api.settings import Settings


def check(settings: Settings, *, redis: Redis | None = None, hostname: str | None = None) -> bool:
    client = redis or Redis.from_url(settings.redis_url, socket_timeout=3, socket_connect_timeout=3)
    try:
        return worker_heartbeat_state(client, hostname=hostname or socket.gethostname()) == "ok"
    except RedisError:
        return False


__all__ = ["check"]
