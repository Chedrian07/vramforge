"""Trainer adapters for SFT, DPO and GRPO (TRL 1.14.1 behavior).

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import Objective

from .base import TrainerAdapter, TrainingSchedule


def get_trainer(objective: Objective) -> TrainerAdapter:
    raise NotImplementedError


__all__ = ["TrainerAdapter", "TrainingSchedule", "get_trainer"]
