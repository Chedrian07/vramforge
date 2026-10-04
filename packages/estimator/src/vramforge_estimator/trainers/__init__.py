"""Trainer adapters for SFT, DPO and GRPO (TRL 1.14.1 behavior).

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import TrainerAdapter, TrainingSchedule
from .registry import get_trainer

__all__ = ["TrainerAdapter", "TrainingSchedule", "get_trainer"]
