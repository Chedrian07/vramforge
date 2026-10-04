"""Full-scan contracts (plan.md §7.6, §7.7, §16.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vramforge_estimator.schemas import DatasetScanResult, Issue, JobProgress


@dataclass(frozen=True)
class ScanLimits:
    """Service resource limits. Hitting one makes the scan PARTIAL — rows are never skipped
    silently and the result is never reported as complete (plan §18)."""

    max_rows: int = 20_000_000
    max_seconds: float = 6 * 3600.0
    max_record_chars: int = 50_000_000  # a single row larger than this fails (recorded)
    progress_interval_s: float = 0.5
    checkpoint_every_rows: int = 10_000


class ScanContext(Protocol):
    """What the scanner needs from the running job (implemented by the worker)."""

    artifact_dir: Path
    limits: ScanLimits

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None: ...

    def cancelled(self) -> bool: ...

    def load_checkpoint(self) -> dict[str, Any] | None: ...

    def save_checkpoint(self, data: dict[str, Any]) -> None: ...


@dataclass
class LengthTable:
    """Column arrays of the row-length artifact used by batch planning and recompute.

    All lists have the same length (one entry per successfully processed row, in source order).
    """

    row_ids: list[str] = field(default_factory=list)
    prompt_tokens: list[int | None] = field(default_factory=list)
    completion_tokens: list[int | None] = field(default_factory=list)
    sequence_tokens: list[int | None] = field(default_factory=list)
    loss_token_count: list[int | None] = field(default_factory=list)
    chosen_total_tokens: list[int | None] = field(default_factory=list)
    rejected_total_tokens: list[int | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.row_ids)


@dataclass
class ScanOutcome:
    result: DatasetScanResult
    artifact_path: Path | None
    issues: list[Issue] = field(default_factory=list)
    # True when at least one mapped content was altered/removed by the chat template.
    template_content_loss_rows: int = 0
