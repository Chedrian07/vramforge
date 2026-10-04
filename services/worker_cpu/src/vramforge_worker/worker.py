"""The forking RQ worker with VRAMForge maintenance (reaper + retention).

A forking `rq.Worker` is required: `SimpleWorker` cannot be stopped by
`send_stop_job_command` (docs/research/stack-compat.md §6). Maintenance runs in the parent
process between jobs (`maintenance_interval`) and once at startup.
"""

from __future__ import annotations

import logging
from typing import Any

from rq import Queue, Worker
from rq.job import Job
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api.settings import Settings

from .lease import expire_lease
from .reaper import reap
from .retention import cleanup_expired

log = logging.getLogger(__name__)


def run_maintenance(settings: Settings, sessions: sessionmaker[Session], queue: Queue) -> None:
    """Reaper and retention; failures are logged and retried at the next interval."""
    try:
        report = reap(settings, sessions, queue.connection, queue)
        if report.requeued or report.failed or report.cancelled or report.reenqueued:
            log.info(
                "reaper: requeued=%d failed=%d cancelled=%d reenqueued=%d stop_sent=%d",
                len(report.requeued),
                len(report.failed),
                len(report.cancelled),
                len(report.reenqueued),
                len(report.stop_sent),
            )
    except Exception:
        log.exception("reaper failed")
    try:
        cleanup_expired(settings, sessions)
    except Exception:
        log.exception("retention cleanup failed")


class VramforgeWorker(Worker):
    """`rq.Worker` that also runs the reaper and retention cleanup as maintenance tasks."""

    vf_settings: Settings | None = None
    vf_sessions: sessionmaker[Session] | None = None

    def run_maintenance_tasks(self) -> None:
        super().run_maintenance_tasks()
        if self.vf_settings is not None and self.vf_sessions is not None and self.queues:
            run_maintenance(self.vf_settings, self.vf_sessions, self.queues[0])


def horse_killed_handler(
    sessions: sessionmaker[Session],
) -> Any:
    """Expire the lease of a job whose work-horse died without a stop command (e.g. OOM), so the
    reaper re-queues it promptly; RQ runs no on_failure callback in that case."""

    def handler(job: Job, retpid: int, ret_val: int, rusage: Any) -> None:
        try:
            analysis_id, attempt = str(job.args[0]), int(job.args[1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        log.error("work-horse for analysis %s died (status %s)", analysis_id, ret_val)
        try:
            expire_lease(sessions, analysis_id, attempt)
        except Exception:
            log.exception("could not expire the lease of %s", analysis_id)

    return handler


__all__ = ["VramforgeWorker", "horse_killed_handler", "run_maintenance"]
