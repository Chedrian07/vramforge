"""Model, tokenizer and dataset inspection.

Owner: ingest agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import RowStream, SourceRow, TokenizerHandle
from .dataset import inspect_dataset, open_rows
from .model import inspect_model
from .tokenizer import load_tokenizer

__all__ = [
    "RowStream",
    "SourceRow",
    "TokenizerHandle",
    "inspect_dataset",
    "inspect_model",
    "load_tokenizer",
    "open_rows",
]
