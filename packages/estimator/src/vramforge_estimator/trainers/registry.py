"""Trainer adapter registry: one TRL 1.14.1 adapter per objective."""

from __future__ import annotations

from vramforge_estimator.schemas import Objective

from .base import TrainerAdapter
from .dpo import DpoTrainer
from .grpo import GrpoTrainer
from .sft import SftTrainer

TRAINER_PREFIX = "trl-1.14.1"

_TRAINERS: dict[Objective, TrainerAdapter] = {
    Objective.SFT: SftTrainer(),
    Objective.DPO: DpoTrainer(),
    Objective.GRPO: GrpoTrainer(),
}


def trainer_id_for(objective: Objective) -> str:
    return f"{TRAINER_PREFIX}-{Objective(objective).value}"


def get_trainer(objective: Objective) -> TrainerAdapter:
    try:
        return _TRAINERS[Objective(objective)]
    except KeyError:
        raise NotImplementedError(f"no trainer adapter for {objective}") from None
