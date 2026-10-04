"""Full dataset scan, row-length artifact, statistics, preservation audit, context validation.

Owner: scan agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from pathlib import Path

from vramforge_estimator.inspection import RowStream
from vramforge_estimator.preprocessing import PreprocessingAdapter
from vramforge_estimator.schemas import (
    BatchPlan,
    ContextValidation,
    DatasetScanResult,
    Objective,
    PreservationAudit,
    ScopeConfig,
    TokenizerManifest,
)

from .base import LengthTable, ScanContext, ScanLimits, ScanOutcome


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


def validate_context(
    scan: DatasetScanResult,
    *,
    model_declared_max: int | None,
    tokenizer: TokenizerManifest | None,
    backend_verified_max: int | None,
    extra_tokens: int = 0,
) -> ContextValidation:
    """Compare the longest sequence (+ `extra_tokens`, e.g. a GRPO completion budget) with the
    limits, keeping model/tokenizer/backend limits separate (plan §7.5)."""
    raise NotImplementedError


def audit_preservation(
    scan: DatasetScanResult,
    *,
    context: ContextValidation,
    batch_plan: BatchPlan | None,
    packing: bool,
    scope: ScopeConfig,
    template_content_loss_rows: int,
) -> PreservationAudit:
    """Evaluate the eight no-truncation conditions of plan §7.4."""
    raise NotImplementedError


__all__ = [
    "LengthTable",
    "ScanContext",
    "ScanLimits",
    "ScanOutcome",
    "audit_preservation",
    "full_scan",
    "load_lengths",
    "validate_context",
]
