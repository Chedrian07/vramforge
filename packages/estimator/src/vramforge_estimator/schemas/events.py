"""Job progress and SSE event payloads (plan.md §15.3).

Events never carry raw rows, tokens or credentials. Event ids are monotonic per analysis so a
reconnecting browser can resume with `Last-Event-ID`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from .common import Issue, JobStatus, VFModel


class EventType(StrEnum):
    PROGRESS = "progress"
    PARTIAL_RESULT = "partial_result"
    WARNING = "warning"
    NEEDS_INPUT = "needs_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_EVENT_TYPES = frozenset(
    {EventType.NEEDS_INPUT, EventType.COMPLETED, EventType.FAILED, EventType.CANCELLED}
)


class ShardProgress(VFModel):
    completed: int
    total: int | None = None  # None when the shard count is not known in advance


class JobProgress(VFModel):
    stage: JobStatus
    processed_rows: int | None = None
    total_rows: int | None = None  # None => UI shows counts, never an invented percentage
    shard_progress: ShardProgress | None = None
    message_code: str | None = None
    message: str | None = None  # Korean, display-ready


class AnalysisEvent(VFModel):
    event_id: int
    analysis_id: str
    type: EventType
    status: JobStatus
    progress: JobProgress | None = None
    fingerprint: str
    issue: Issue | None = None
    # Small, display-only partial statistics while scanning (e.g. current max length so far).
    partial: dict[str, int | float | str | None] | None = None
    timestamp: datetime
