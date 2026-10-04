"""`vramforge-worker` command line.

- `vramforge-worker` / `vramforge-worker run`: forking RQ worker on the "analyses" queue with the
  JSON serializer, worker TTL 60 s, startup and periodic reaper + retention cleanup.
- `vramforge-worker healthcheck`: exit 0 iff this host has a live worker heartbeat in Redis.
- `vramforge-worker maintenance`: run the reaper and retention once (operations).
"""

from __future__ import annotations

import argparse
import logging
import sys

from redis import Redis
from rq.serializers import JSONSerializer

from vramforge_api.db import get_engine, session_factory
from vramforge_api.jobs import make_queue
from vramforge_api.logging_setup import configure_logging
from vramforge_api.settings import Settings, get_settings

from .healthcheck import check
from .tasks import prepare_environment
from .worker import VramforgeWorker, horse_killed_handler, run_maintenance

log = logging.getLogger("vramforge_worker")


def _redis(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url, socket_connect_timeout=5)


def run_worker(settings: Settings, *, burst: bool = False) -> int:
    configure_logging(settings.log_level)
    prepare_environment(settings)
    redis = _redis(settings)
    queue = make_queue(redis, settings.queue_name)
    sessions = session_factory(get_engine(settings.database_url))
    run_maintenance(settings, sessions, queue)
    worker = VramforgeWorker(
        [queue],
        connection=redis,
        serializer=JSONSerializer,
        worker_ttl=settings.worker_ttl_s,
        maintenance_interval=settings.maintenance_interval_s,
        work_horse_killed_handler=horse_killed_handler(sessions),
    )
    worker.vf_settings = settings
    worker.vf_sessions = sessions
    worker.work(burst=burst, logging_level=settings.log_level.upper())
    return 0


def run_maintenance_once(settings: Settings) -> int:
    configure_logging(settings.log_level)
    redis = _redis(settings)
    queue = make_queue(redis, settings.queue_name)
    run_maintenance(settings, session_factory(get_engine(settings.database_url)), queue)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vramforge-worker", description="VRAMForge CPU worker")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="process analysis jobs (default)")
    run.add_argument("--burst", action="store_true", help="exit when the queue is empty")
    sub.add_parser("healthcheck", help="exit 0 iff a live worker heartbeat exists on this host")
    sub.add_parser("maintenance", help="run the reaper and retention cleanup once")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "healthcheck":
        sys.exit(0 if check(settings) else 1)
    if args.command == "maintenance":
        sys.exit(run_maintenance_once(settings))
    sys.exit(run_worker(settings, burst=getattr(args, "burst", False)))


if __name__ == "__main__":
    main()
