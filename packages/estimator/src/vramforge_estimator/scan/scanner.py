"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from pathlib import Path

from vramforge_estimator.inspection import RowStream
from vramforge_estimator.preprocessing import PreprocessingAdapter
from vramforge_estimator.schemas import (
    Objective,
)

from .base import LengthTable, ScanContext, ScanOutcome


def full_scan(
    stream: RowStream,
    adapter: PreprocessingAdapter,
    ctx: ScanContext,
    *,
    objective: Objective,
    preprocess_key: str,
    tokenizer_fingerprint: str,
    template_fingerprint: str | None,
    context_limit: int | None,
) -> ScanOutcome:
    """Tokenize every row of the stream, write the Parquet row-length artifact under
    `ctx.artifact_dir`, compute exact counts/max and statistics, and resume from the checkpoint
    when present. Coverage is COMPLETE only when the stream is complete and no row is
    unprocessed."""
    raise NotImplementedError


def load_lengths(artifact_path: Path) -> LengthTable:
    """Read the row-length artifact written by `full_scan`."""
    raise NotImplementedError
