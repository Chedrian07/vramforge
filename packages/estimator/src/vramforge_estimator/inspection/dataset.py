"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    DatasetInspection,
    DatasetSourceRef,
    Objective,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .base import RowStream


def inspect_dataset(
    source: ResolvedSource,
    ref: DatasetSourceRef,
    access: SourceAccess,
    objective: Objective | None = None,
) -> DatasetInspection:
    """Configs, splits, columns, detected format and ranked mapping candidates (preview only)."""
    raise NotImplementedError


def open_rows(
    source: ResolvedSource,
    *,
    config: str | None,
    split: str,
    access: SourceAccess,
) -> RowStream:
    """Open a sequential reader over every row of `split` (never executes dataset scripts)."""
    raise NotImplementedError
