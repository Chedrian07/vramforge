"""Architecture adapters: dense decoder and Qwen3.5 hybrid (plan.md §2.1, §6.4).

Owner: architecture agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import ArchitectureAdapter, GenerationTimepoints, StepTimepoints, TrainableGroup
from .registry import get_adapter, match_adapter

__all__ = [
    "ArchitectureAdapter",
    "GenerationTimepoints",
    "StepTimepoints",
    "TrainableGroup",
    "get_adapter",
    "match_adapter",
]
