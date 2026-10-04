"""Memory ledger and estimate contract (plan.md §9, §10).

Peak semantics: the engine sums the allocations alive at each timepoint and takes the maximum
timepoint. The peak breakdown lists exactly the allocations alive at that one timepoint, so the
stacked bar always equals the reported peak (plan §12.2). Per-phase maxima are never added.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .batching import BatchShape
from .common import (
    Assumption,
    Evidence,
    EvidenceLevel,
    ExcludedComponent,
    HardwareFit,
    Issue,
    UnknownComponent,
    VFModel,
)
from .request import MarginPolicy


class Phase(StrEnum):
    MODEL_LOAD_AND_QUANTIZE = "MODEL_LOAD_AND_QUANTIZE"
    REFERENCE_PRECOMPUTE = "REFERENCE_PRECOMPUTE"
    ROLLOUT_PREFILL_AND_DECODE = "ROLLOUT_PREFILL_AND_DECODE"
    REWARD = "REWARD"
    POLICY_FORWARD_BACKWARD = "POLICY_FORWARD_BACKWARD"
    OPTIMIZER_STEP = "OPTIMIZER_STEP"
    WEIGHT_SYNC = "WEIGHT_SYNC"
    EVALUATION = "EVALUATION"
    CHECKPOINT_SAVE_OR_CONSOLIDATE = "CHECKPOINT_SAVE_OR_CONSOLIDATE"


PHASE_ORDER: tuple[Phase, ...] = tuple(Phase)


class AllocationCategory(StrEnum):
    WEIGHTS_BASE = "weights_base"
    WEIGHTS_ADAPTER = "weights_adapter"
    WEIGHTS_OTHER_MODELS = "weights_other_models"  # reference / reward / rollout copies
    GRADIENTS = "gradients"
    OPTIMIZER_STATES = "optimizer_states"
    MASTER_WEIGHTS = "master_weights"
    SAVED_ACTIVATIONS = "saved_activations"
    RECOMPUTE_WORKING_SET = "recompute_working_set"
    LOGITS_AND_LOSS = "logits_and_loss"
    GENERATION_CACHE = "generation_cache"  # attention KV cache
    RECURRENT_STATE = "recurrent_state"  # linear-attention conv/recurrent state
    ROLLOUT_BUFFERS = "rollout_buffers"  # completions, masks, old/ref log-probs, advantages
    LOAD_TRANSIENT = "load_transient"
    WORKSPACE = "workspace"
    COMMUNICATION = "communication_buffers"
    ALLOCATOR_SLACK = "allocator_slack"
    NON_FRAMEWORK = "non_framework"  # CUDA context, library handles


# Categories counted in known_floor_bytes when their size is known (plan §10.2).
RESIDENT_CATEGORIES: frozenset[AllocationCategory] = frozenset(
    {
        AllocationCategory.WEIGHTS_BASE,
        AllocationCategory.WEIGHTS_ADAPTER,
        AllocationCategory.WEIGHTS_OTHER_MODELS,
        AllocationCategory.GRADIENTS,
        AllocationCategory.OPTIMIZER_STATES,
        AllocationCategory.MASTER_WEIGHTS,
    }
)


class Timepoint(VFModel):
    id: str  # "<PHASE>:<name>", e.g. "POLICY_FORWARD_BACKWARD:loss"
    phase: Phase
    name: str
    description: str
    order: int  # global order within one training step


class AllocationSpec(VFModel):
    """One entry of the shape ledger (plan §9.5).

    `bytes_low`/`bytes_high` are TOTAL bytes (already multiplied by `count`). Deterministic
    allocations have low == high. Unknown size => both None and `note` says why; never 0.
    `live_at` lists timepoint ids; "<PHASE>:*" means every timepoint of that phase.
    Allocations sharing a `storage_alias_group` are the same storage: at a timepoint the group
    contributes only its largest member.
    """

    name: str
    category: AllocationCategory
    device: str = "cuda:0"
    shape_expression: str | None = None
    dims: dict[str, int] = Field(default_factory=dict)
    dtype: str | None = None
    count: int = 1
    bytes_low: int | None
    bytes_high: int | None
    live_at: list[str]
    saved_for_backward: bool = False
    recompute_group: str | None = None
    storage_alias_group: str | None = None
    evidence: Evidence
    formula_ref: str | None = None  # anchor in docs/methodology.md or methodology-architectures.md
    note: str | None = None


class TimepointTotal(VFModel):
    timepoint: str
    phase: Phase
    bytes_low: int | None
    bytes_high: int | None
    known_floor_bytes: int
    unknown: list[str] = Field(default_factory=list)


class PhasePeak(VFModel):
    phase: Phase
    included: bool
    excluded_reason: str | None = None
    peak_timepoint: str | None = None
    bytes_low: int | None = None
    bytes_high: int | None = None
    known_floor_bytes: int | None = None
    unknown_components: list[str] = Field(default_factory=list)


class BreakdownItem(VFModel):
    name: str
    category: AllocationCategory
    bytes_low: int | None
    bytes_high: int | None
    evidence: Evidence
    note: str | None = None


class PeakBreakdown(VFModel):
    timepoint: str
    phase: Phase
    items: list[BreakdownItem]
    by_category_high: dict[str, int] = Field(default_factory=dict)
    total_low: int | None
    total_high: int | None


class DeviceEstimate(VFModel):
    device: str
    phases: list[PhasePeak]
    timepoints: list[TimepointTotal]
    peak_phase: Phase | None = None
    peak_timepoint: str | None = None
    known_floor_bytes: int
    scenario_low_bytes: int | None
    scenario_high_bytes: int | None
    peak_breakdown: PeakBreakdown | None = None
    unknown_components: list[UnknownComponent] = Field(default_factory=list)


class CapacityRecommendation(VFModel):
    planning_margin_bytes: int
    recommended_application_capacity_bytes: int
    external_reserved_bytes: int
    required_total_device_capacity_bytes: int
    policy: MarginPolicy


FitReason = Literal[
    "not_evaluated",
    "floor_exceeds_capacity",
    "high_exceeds_capacity",
    "margin_insufficient",
    "fits_with_margin",
    "unknown_components",
    "unsupported",
    "load_budget_insufficient",
    "scan_incomplete",
]


class HardwareFitResult(VFModel):
    status: HardwareFit
    reason: FitReason
    message: str  # Korean, display-ready
    capacity_bytes: int | None = None
    utilization_ratio: float | None = None  # (high + margin) / capacity; may exceed 1.0


class ScenarioEstimate(VFModel):
    scenario_id: str  # "default", "budget_1024", ...
    label: str
    params: dict[str, Any] = Field(default_factory=dict)
    batch_shape: BatchShape
    devices: list[DeviceEstimate]
    recommendation: CapacityRecommendation | None = None
    hardware_fit: HardwareFitResult
    excluded_components: list[ExcludedComponent] = Field(default_factory=list)
    timepoints: list[Timepoint] = Field(default_factory=list)
    allocations: list[AllocationSpec] = Field(default_factory=list)


class EstimateItem(VFModel):
    name: str
    bytes_low: int | None
    bytes_high: int | None
    evidence: Evidence
    note: str | None = None


class HostRamEstimate(VFModel):
    """Training-node RAM (plan §10.1)."""

    bytes_low: int | None
    bytes_high: int | None
    items: list[EstimateItem] = Field(default_factory=list)


class AnalysisRamEstimate(VFModel):
    """RAM used by this analysis service while scanning (plan §10.1)."""

    bytes_low: int | None
    bytes_high: int | None
    items: list[EstimateItem] = Field(default_factory=list)


class DiskEstimate(VFModel):
    items: list[EstimateItem] = Field(default_factory=list)
    total_bytes: int | None


class MemoryEstimate(VFModel):
    evidence_level: EvidenceLevel
    scenarios: list[ScenarioEstimate]
    primary_scenario_id: str | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
