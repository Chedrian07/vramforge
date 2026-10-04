"""Resolved configuration: what the calculation actually used (plan.md §11).

The compatibility resolver turns an `AnalysisRequest` + model inventory + backend profile into a
`ResolvedConfig`. Architecture and trainer adapters read ONLY this object (never profile files),
which keeps "requested vs resolved vs observed" auditable.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import EvidenceLevel, Issue, Objective, Strategy, TrainingReadiness, VFModel
from .request import EmptySystemPolicy, ReferenceStrategy, RewardKind, RolloutBackend


class ConfigResolution(VFModel):
    field: str
    requested: Any = None
    resolved: Any = None
    reason: str


class EffectiveDtypes(VFModel):
    weights_nonquantized: str  # resident dtype of modules that are not 4-bit
    compute: str  # activation / matmul dtype
    adapter: str  # LoRA parameter dtype
    gradient: str
    optimizer_state: str
    master_weights: str | None = None
    logits: str
    loss: str
    kv_cache: str
    recurrent_state: str


class QuantizationResolved(VFModel):
    enabled: bool
    method: Literal["bnb_nf4", "bnb_fp4"] | None = None
    double_quant: bool = False
    blocksize: int | None = None
    nested_blocksize: int | None = None
    quant_storage_dtype: str | None = None
    compute_dtype: str | None = None
    # Module-name patterns kept out of 4-bit conversion (e.g. lm_head, vision tower).
    skip_module_patterns: list[str] = Field(default_factory=list)


class LoraResolved(VFModel):
    r: int
    alpha: float
    dropout: float
    target_module_patterns: list[str]  # what will be emitted in the trainer config
    target_modules: list[str]  # resolved inventory module names
    exclude_modules: list[str] = Field(default_factory=list)
    modules_to_save: list[str] = Field(default_factory=list)
    bias: str = "none"
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    use_dora: bool = False
    use_rslora: bool = False


class OptimizerResolved(VFModel):
    name: str
    states_per_param: int  # e.g. 2 for Adam(W)
    state_dtype: str  # dtype of full-precision states, or "uint8" for 8-bit
    eight_bit: bool = False
    block_size: int | None = None  # 8-bit blockwise quantization block
    min_8bit_size: int | None = None  # tensors smaller than this keep 32-bit states
    paged: bool = False
    fused: bool = False


class DpoResolved(VFModel):
    reference_strategy: ReferenceStrategy
    beta: float
    loss_type: str
    precompute_batch_size: int | None = None
    sync_ref_model: bool = False


class GrpoResolved(VFModel):
    num_generations: int
    generation_batch_size: int
    steps_per_generation: int
    num_iterations: int
    completion_budgets: list[int]
    budget_explicit: bool
    beta: float
    reference_needed: bool
    reward_kind: RewardKind
    reward_model_reference: str | None = None
    rollout_backend: RolloutBackend
    live_sequences: int  # C
    update_microbatch: int  # B_update
    accumulation: int  # K


class WorkspaceAssumptions(VFModel):
    """Low/high bytes for components without a shape formula (versioned in the profile)."""

    cuda_context_bytes: tuple[int, int]
    library_workspace_bytes: tuple[int, int]
    allocator_slack_fraction: tuple[float, float]
    notes: list[str] = Field(default_factory=list)


class ResolvedConfig(VFModel):
    profile_id: str | None = None
    profile_version: str | None = None
    environment_id: str | None = None
    dependency_lock_digest: str | None = None
    architecture_adapter: str | None = None
    trainer_adapter: str | None = None
    preprocessing_adapter: str | None = None
    objective: Objective
    strategy: Strategy
    loading_scope: Literal["full_checkpoint", "text_only"]
    effective_dtypes: EffectiveDtypes
    quantization: QuantizationResolved
    upcast_to_fp32_patterns: list[str] = Field(default_factory=list)
    lora: LoraResolved | None = None
    trainable_full_patterns: list[str] = Field(default_factory=list)  # full FT / modules_to_save
    optimizer: OptimizerResolved
    microbatch: int
    accumulation: int
    pad_to_multiple_of: int | None = None
    gradient_checkpointing: bool
    checkpointing_granularity: Literal["none", "per_decoder_layer"]
    attention_path_by_layer_type: dict[str, str] = Field(default_factory=dict)
    loss_path: str
    use_cache_during_training: bool = False
    dpo: DpoResolved | None = None
    grpo: GrpoResolved | None = None
    workspace: WorkspaceAssumptions
    template_kwargs: dict[str, Any] = Field(default_factory=dict)
    empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT
    resolutions: list[ConfigResolution] = Field(default_factory=list)


class SupportEntry(VFModel):
    objective: Objective
    strategy: Strategy
    grade: EvidenceLevel | None  # None = unsupported
    readiness: TrainingReadiness
    note: str = ""


class CompatibilityReport(VFModel):
    profile_id: str | None = None
    architecture_adapter: str | None = None
    support_grade: EvidenceLevel | None = None  # None = unsupported combination
    readiness: TrainingReadiness
    support: list[SupportEntry] = Field(default_factory=list)
    blockers: list[Issue] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)
    not_effective: list[ConfigResolution] = Field(default_factory=list)
