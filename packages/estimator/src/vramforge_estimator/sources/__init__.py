"""Source resolution: reference normalization, revision pinning, local roots, uploads.

Owner: ingest agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import DatasetSourceRef, ModelSourceRef

from .base import ResolvedSource, SourceAccess


def resolve_model(ref: ModelSourceRef, access: SourceAccess) -> ResolvedSource:
    """Normalize the reference, pin the revision to an immutable identity and list files.

    Raises `EstimatorError` with SOURCE_NOT_FOUND / SOURCE_ACCESS_DENIED / LOCAL_PATH_NOT_ALLOWED /
    SOURCE_URL_NOT_ALLOWED. Never downloads weight files.
    """
    raise NotImplementedError


def resolve_dataset(ref: DatasetSourceRef, access: SourceAccess) -> ResolvedSource:
    """Same as `resolve_model` for datasets (HF datasets, local files, uploads)."""
    raise NotImplementedError


__all__ = ["ResolvedSource", "SourceAccess", "resolve_dataset", "resolve_model"]
