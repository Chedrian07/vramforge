"""Job leases (plan.md §16.1–16.2): at most one runner per analysis attempt.

A runner owns the analysis while `lease_owner` is its id and `lease_expires_at` is in the future.
The heartbeat thread extends the lease and mirrors the cancel flag; if the row disappears or
another runner took over (the reaper re-queued an expired lease), the runner stops without
writing anything.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis
from vramforge_api.store import ACTIVE_STATUSES

log = logging.getLogger(__name__)


def new_lease_owner() -> str:
    return f"{socket.gethostname()[:64]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@dataclass
class LeaseState:
    """Flags shared between the heartbeat thread and the job (plain bools are atomic enough)."""

    cancel_requested: bool = False
    lease_lost: bool = False


def acquire_lease(
    sessions: sessionmaker[Session], analysis_id: str, attempt: int, owner: str, ttl_s: int
) -> bool:
    """Atomically take the lease for this attempt; False when another live lease exists, the
    attempt is stale, the analysis is finished/cancelled or was deleted."""
    now = utcnow()
    with session_scope(sessions) as db:
        result = db.execute(
            update(Analysis)
            .where(
                Analysis.id == analysis_id,
                Analysis.attempt == attempt,
                Analysis.status.in_(ACTIVE_STATUSES),
                Analysis.cancel_requested.is_(False),
                or_(Analysis.lease_owner.is_(None), Analysis.lease_expires_at < now),
            )
            .values(
                lease_owner=owner,
                lease_expires_at=now + timedelta(seconds=ttl_s),
                started_at=func.coalesce(Analysis.started_at, now),
                updated_at=now,
            )
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


def expire_lease(sessions: sessionmaker[Session], analysis_id: str, attempt: int) -> None:
    """Mark a dead runner's lease as expired now (the reaper re-queues it)."""
    with session_scope(sessions) as db:
        db.execute(
            update(Analysis)
            .where(
                Analysis.id == analysis_id,
                Analysis.attempt == attempt,
                Analysis.lease_owner.is_not(None),
            )
            .values(lease_expires_at=utcnow())
        )


class Heartbeat(threading.Thread):
    """Extends the lease every ttl/3 and refreshes the cancel flag every `poll_s`."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        analysis_id: str,
        owner: str,
        state: LeaseState,
        *,
        ttl_s: int,
        poll_s: float,
    ) -> None:
        super().__init__(name=f"lease-{analysis_id[:8]}", daemon=True)
        self.sessions = sessions
        self.analysis_id = analysis_id
        self.owner = owner
        self.state = state
        self.ttl_s = ttl_s
        self.poll_s = poll_s
        self._stop_event = threading.Event()

    def beat(self, *, extend: bool) -> None:
        with session_scope(self.sessions) as db:
            row = db.execute(
                select(Analysis.lease_owner, Analysis.cancel_requested).where(
                    Analysis.id == self.analysis_id
                )
            ).first()
            if row is None or row.lease_owner != self.owner:
                self.state.lease_lost = True
                return
            self.state.cancel_requested = bool(row.cancel_requested)
            if extend:
                changed = db.execute(
                    update(Analysis)
                    .where(Analysis.id == self.analysis_id, Analysis.lease_owner == self.owner)
                    .values(lease_expires_at=utcnow() + timedelta(seconds=self.ttl_s))
                ).rowcount  # type: ignore[attr-defined]
                if not changed:
                    self.state.lease_lost = True

    def run(self) -> None:
        last_extend = time.monotonic()
        while not self._stop_event.wait(self.poll_s):
            extend = time.monotonic() - last_extend >= self.ttl_s / 3
            try:
                self.beat(extend=extend)
            except Exception as exc:  # transient DB trouble: keep trying until the lease expires
                log.warning("lease heartbeat failed: %s", type(exc).__name__)
                continue
            if extend:
                last_extend = time.monotonic()
            if self.state.lease_lost:
                log.warning("lease for analysis %s lost; stopping", self.analysis_id)
                return

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)


__all__ = ["Heartbeat", "LeaseState", "acquire_lease", "expire_lease", "new_lease_owner"]
