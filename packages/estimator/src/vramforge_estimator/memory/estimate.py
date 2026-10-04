"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    BatchPlan,
    HardwareConfig,
    MarginPolicy,
    MemoryEstimate,
    ModelInventory,
    ResolvedConfig,
    ScopeConfig,
    TrainingReadiness,
)


def estimate_memory(
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    plan: BatchPlan,
    *,
    scope: ScopeConfig,
    hardware: HardwareConfig,
    margin_policy: MarginPolicy,
    readiness: TrainingReadiness,
) -> MemoryEstimate:
    """One `ScenarioEstimate` per batch-plan scenario (GRPO: per completion budget)."""
    raise NotImplementedError
