"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    CapacityRecommendation,
    DeviceEstimate,
    HardwareConfig,
    HardwareFitResult,
    MarginPolicy,
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
