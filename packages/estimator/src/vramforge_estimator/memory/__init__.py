"""Memory engine: timepoint evaluation, scenario estimates, margin, hardware fit, RAM/disk.

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import (
    AnalysisRamEstimate,
    BatchPlan,
    CapacityRecommendation,
    DeviceEstimate,
    DiskEstimate,
    HardwareConfig,
    HardwareFitResult,
    HostRamEstimate,
    MarginPolicy,
    MemoryEstimate,
    ModelInventory,
    ResolvedConfig,
    ScopeConfig,
    SourceManifest,
    TrainingReadiness,
)
from vramforge_estimator.trainers import TrainingSchedule


def evaluate(schedule: TrainingSchedule, device: str = "cuda:0") -> DeviceEstimate:
    """Sum allocations alive at each timepoint (alias groups counted once) and pick the maximum
    timepoint. floor = known resident categories; low/high = None if an unknown is alive at the
    peak-candidate timepoints (plan §9.1, §10.2)."""
    raise NotImplementedError


def recommend(
    estimate: DeviceEstimate, policy: MarginPolicy, external_reserved_bytes: int
) -> CapacityRecommendation | None:
    """planning_margin = max(min_bytes, high × fraction); None when high is unknown."""
    raise NotImplementedError


def assess_fit(
    estimate: DeviceEstimate,
    recommendation: CapacityRecommendation | None,
    hardware: HardwareConfig,
    readiness: TrainingReadiness,
) -> HardwareFitResult:
    """The six outcomes of plan §10.3 (incl. not_evaluated when no hardware is selected)."""
    raise NotImplementedError


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


def estimate_host_ram(
    inventory: ModelInventory, cfg: ResolvedConfig, model_source: SourceManifest | None
) -> HostRamEstimate:
    raise NotImplementedError


def estimate_disk(
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    model_source: SourceManifest | None,
    dataset_source: SourceManifest | None,
    artifact_bytes: int | None,
) -> DiskEstimate:
    raise NotImplementedError


def estimate_analysis_ram(rows: int | None, tokenizer_bytes: int | None) -> AnalysisRamEstimate:
    raise NotImplementedError


__all__ = [
    "assess_fit",
    "estimate_analysis_ram",
    "estimate_disk",
    "estimate_host_ram",
    "estimate_memory",
    "evaluate",
    "recommend",
]
