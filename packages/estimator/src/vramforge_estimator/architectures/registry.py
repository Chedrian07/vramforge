"""Adapter registry: structural matching of architecture facts (plan.md §6.2, §11.4)."""

from __future__ import annotations

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ArchitectureFacts, ErrorCode, Stage

from .base import ArchitectureAdapter
from .decoder import DecoderAdapter, DenseDecoderAdapter, Qwen35HybridAdapter

# Order matters: the more specific structure (linear-attention layers) is tried first.
_ADAPTERS: dict[str, DecoderAdapter] = {
    a.adapter_id: a for a in (Qwen35HybridAdapter(), DenseDecoderAdapter())
}


def match_adapter(facts: ArchitectureFacts) -> str | None:
    """Return the adapter id that supports these facts, or None (UNSUPPORTED_ARCHITECTURE).
    Matching is structural (config fields/layer types), never by model-name substrings."""
    for adapter_id, adapter in _ADAPTERS.items():
        if adapter.supports(facts):
            return adapter_id
    return None


def get_adapter(adapter_id: str) -> ArchitectureAdapter:
    adapter = _ADAPTERS.get(adapter_id)
    if adapter is None:
        raise EstimatorError(
            make_issue(
                ErrorCode.UNSUPPORTED_ARCHITECTURE,
                "등록되지 않은 architecture adapter입니다.",
                stage=Stage.ESTIMATING,
                component="architecture",
                adapter=adapter_id,
            )
        )
    return adapter


def adapter_ids() -> list[str]:
    return list(_ADAPTERS)
