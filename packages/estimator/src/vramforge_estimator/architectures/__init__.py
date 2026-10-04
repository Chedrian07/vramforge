"""Architecture adapters: dense decoder and Qwen3.5 hybrid (plan.md §2.1, §6.4).

Owner: architecture agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import ArchitectureFacts

from .base import ArchitectureAdapter, GenerationTimepoints, StepTimepoints, TrainableGroup


def match_adapter(facts: ArchitectureFacts) -> str | None:
    """Return the adapter id that supports these facts, or None (UNSUPPORTED_ARCHITECTURE).
    Matching is structural (config fields/layer types), never by model-name substrings."""
    raise NotImplementedError


def get_adapter(adapter_id: str) -> ArchitectureAdapter:
    raise NotImplementedError


__all__ = [
    "ArchitectureAdapter",
    "GenerationTimepoints",
    "StepTimepoints",
    "TrainableGroup",
    "get_adapter",
    "match_adapter",
]
