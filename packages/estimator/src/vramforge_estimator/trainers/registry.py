"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import Objective

from .base import TrainerAdapter


def get_trainer(objective: Objective) -> TrainerAdapter:
    raise NotImplementedError
