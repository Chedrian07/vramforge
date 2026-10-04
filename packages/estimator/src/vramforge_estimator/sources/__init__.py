"""Source resolution: reference normalization, revision pinning, local roots, uploads.

Owner: ingest agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import ResolvedSource, SourceAccess
from .resolver import resolve_dataset, resolve_model

__all__ = ["ResolvedSource", "SourceAccess", "resolve_dataset", "resolve_model"]
