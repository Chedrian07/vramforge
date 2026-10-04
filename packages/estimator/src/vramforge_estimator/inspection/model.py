"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    ModelInventory,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess


def inspect_model(source: ResolvedSource, access: SourceAccess) -> ModelInventory:
    """Build the tensor inventory from config + safetensors headers (no weight download).

    Raises `EstimatorError` with MODEL_METADATA_UNAVAILABLE, REMOTE_CODE_REQUIRED or
    UNSUPPORTED_MODEL_FORMAT.
    """
    raise NotImplementedError
