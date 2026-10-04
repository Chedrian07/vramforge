"""Model, tokenizer and dataset inspection.

Owner: ingest agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import RowStream, SourceRow, TokenizerHandle
from .dataset import inspect_dataset, open_rows
from .dataset_rows import FailedSourceRow, is_failed_row
from .model import inspect_model
from .tokenizer import load_tokenizer

__all__ = [
    "FailedSourceRow",
    "RowStream",
    "SourceRow",
    "TokenizerHandle",
    "inspect_dataset",
    "inspect_model",
    "is_failed_row",
    "load_tokenizer",
    "open_rows",
]
