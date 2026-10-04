"""Model, tokenizer and dataset inspection.

Owner: ingest agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import (
    DatasetInspection,
    DatasetSourceRef,
    ModelInventory,
    Objective,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .base import RowStream, SourceRow, TokenizerHandle


def inspect_model(source: ResolvedSource, access: SourceAccess) -> ModelInventory:
    """Build the tensor inventory from config + safetensors headers (no weight download).

    Raises `EstimatorError` with MODEL_METADATA_UNAVAILABLE, REMOTE_CODE_REQUIRED or
    UNSUPPORTED_MODEL_FORMAT.
    """
    raise NotImplementedError


def load_tokenizer(source: ResolvedSource, access: SourceAccess) -> TokenizerHandle:
    """Load the model's own tokenizer and chat template (trust_remote_code=False).

    Raises `EstimatorError` with TOKENIZER_REQUIRED / TEMPLATE_REQUIRED / REMOTE_CODE_REQUIRED.
    Never substitutes another model's tokenizer or template.
    """
    raise NotImplementedError


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


__all__ = [
    "RowStream",
    "SourceRow",
    "TokenizerHandle",
    "inspect_dataset",
    "inspect_model",
    "load_tokenizer",
    "open_rows",
]
