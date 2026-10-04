"""RQ queue helpers (docs/research/stack-compat.md §6, R1–R5).

- JSONSerializer everywhere (no pickle; plan.md §18).
- Explicit `job_timeout` (the RQ default of 180 s is far too short for a full scan).
- One RQ job per attempt: `<analysis_id>-a<attempt>` (`:` is not allowed in job ids, and a
  finished job hash would block re-enqueueing the same id). No `Retry`: an OOM-killed horse would
  only repeat the OOM; expired leases are re-run by the reaper instead.
- `job.cancel()` only for queued jobs; running jobs get `send_stop_job_command` after the
  cooperative cancel grace period.
- The task is referenced by its import path, so the API never imports the worker package.
"""

from __future__ import annotations

import logging
import socket
from datetime import UTC, datetime

from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Callback, Job, JobStatus
from rq.serializers import JSONSerializer
from rq.worker import Worker

log = logging.getLogger(__name__)

TASK_PATH = "vramforge_worker.tasks.run_analysis"
STOPPED_CALLBACK_PATH = "vramforge_worker.tasks.on_analysis_stopped"
RESULT_TTL_S = 3_600
FAILURE_TTL_S = 86_400
HEARTBEAT_MAX_AGE_S = 120.0


def job_id_for(analysis_id: str, attempt: int) -> str:
    return f"{analysis_id}-a{attempt}"


def make_queue(redis: Redis, name: str, *, is_async: bool = True) -> Queue:
    return Queue(name, connection=redis, serializer=JSONSerializer, is_async=is_async)


def enqueue_analysis(queue: Queue, analysis_id: str, attempt: int, job_timeout_s: int) -> Job:
    return queue.enqueue(
        TASK_PATH,
        analysis_id,
        attempt,
        job_id=job_id_for(analysis_id, attempt),
        job_timeout=job_timeout_s,
        result_ttl=RESULT_TTL_S,
        failure_ttl=FAILURE_TTL_S,
        on_stopped=Callback(STOPPED_CALLBACK_PATH),
        description=f"analysis {analysis_id} attempt {attempt}",
    )


def fetch_job(redis: Redis, analysis_id: str, attempt: int) -> Job | None:
    try:
        return Job.fetch(
            job_id_for(analysis_id, attempt), connection=redis, serializer=JSONSerializer
        )
    except NoSuchJobError:
        return None


def job_status(redis: Redis, analysis_id: str, attempt: int) -> str | None:
    """RQ status of the attempt's job, or None when Redis no longer knows it."""
    job = fetch_job(redis, analysis_id, attempt)
    if job is None:
        return None
    status = job.get_status(refresh=True)
    return str(status.value if isinstance(status, JobStatus) else status) if status else None


def cancel_queued_job(redis: Redis, analysis_id: str, attempt: int) -> bool:
    """Remove a not-yet-started job from the queue. Returns True when it was cancelled."""
    try:
        job = fetch_job(redis, analysis_id, attempt)
        if job is None:
            return False
        if job.get_status(refresh=True) in (JobStatus.QUEUED, JobStatus.DEFERRED):
            job.cancel()
            return True
    except (RedisError, InvalidJobOperation) as exc:
        log.warning("could not cancel queued job %s: %s", analysis_id, type(exc).__name__)
    return False


def stop_running_job(redis: Redis, analysis_id: str, attempt: int) -> bool:
    """Ask the executing worker to kill the work-horse (`on_stopped` updates the DB)."""
    try:
        send_stop_job_command(redis, job_id_for(analysis_id, attempt), serializer=JSONSerializer)
        return True
    except (NoSuchJobError, InvalidJobOperation):
        return False
    except RedisError as exc:
        log.warning("could not stop job %s: %s", analysis_id, type(exc).__name__)
        return False


def worker_heartbeat_state(
    redis: Redis, *, hostname: str | None = None, max_age_s: float = HEARTBEAT_MAX_AGE_S
) -> str:
    """ "ok" when a registered worker (optionally on `hostname`) heartbeated recently,
    "stale" when workers exist but none is fresh, "none" when no worker is registered."""
    workers = Worker.all(connection=redis, serializer=JSONSerializer)
    if hostname is not None:
        workers = [w for w in workers if w.hostname == hostname]
    if not workers:
        return "none"
    now = datetime.now(UTC)
    for worker in workers:
        beat = worker.last_heartbeat
        if beat is None:
            continue
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=UTC)
        if (now - beat).total_seconds() < max_age_s:
            return "ok"
    return "stale"


def local_hostname() -> str:
    return socket.gethostname()


__all__ = [
    "STOPPED_CALLBACK_PATH",
    "TASK_PATH",
    "cancel_queued_job",
    "enqueue_analysis",
    "fetch_job",
    "job_id_for",
    "job_status",
    "local_hostname",
    "make_queue",
    "stop_running_job",
    "worker_heartbeat_state",
]
