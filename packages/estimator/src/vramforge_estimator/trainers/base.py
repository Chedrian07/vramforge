"""TrainerAdapter contract (plan.md §9.2, §9.4, §9.6, §9.7).

A trainer adapter knows the EXECUTION PROCEDURE of the pinned trainer for one objective: the
phases of a step, the timepoints inside them, which passes run where, how long buffers live, and
the LM-head/loss path. It composes architecture-adapter ledgers with optimizer/gradient states and
rollout/reference buffers into a `TrainingSchedule` that the memory engine evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from vramforge_estimator.architectures import ArchitectureAdapter
from vramforge_estimator.schemas import (
    AllocationSpec,
    Assumption,
    BatchPlan,
    BatchShape,
    ExcludedComponent,
    Issue,
    ModelInventory,
    Objective,
    ResolvedConfig,
    ScopeConfig,
    Timepoint,
)


@dataclass
class TrainingSchedule:
    timepoints: list[Timepoint]
    allocations: list[AllocationSpec]
    excluded: list[ExcludedComponent] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


class TrainerAdapter(Protocol):
    trainer_id: str  # e.g. "trl-1.14.1-grpo"
    objective: Objective

    def build_schedule(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        arch: ArchitectureAdapter,
        shape: BatchShape,
        plan: BatchPlan,
        scope: ScopeConfig,
    ) -> TrainingSchedule: ...
