"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from typing import Any

from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import ColumnMapping, EmptySystemPolicy, Objective

from .base import PreprocessingAdapter


def get_adapter(
    objective: Objective,
    tokenizer: TokenizerHandle,
    mapping: ColumnMapping,
    *,
    template_kwargs: dict[str, Any] | None = None,
    empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT,
) -> PreprocessingAdapter:
    """Return the adapter for `objective`; raises `EstimatorError` (COLUMN_MAPPING_REQUIRED)
    when the mapping cannot produce the records the objective needs."""
    raise NotImplementedError
