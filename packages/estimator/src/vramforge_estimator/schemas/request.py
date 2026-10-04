"""Analysis request contract (plan.md §5, §15.2).

`None` on an optional knob means "resolve from the backend profile preset"; the value actually
used is recorded in `ResolvedConfig` together with the reason (plan §11.3: requested vs resolved).
Cross-field semantics (e.g. 4-bit + full fine-tune) are validated server-side by
`vramforge_estimator.compatibility`, not by these models, so that every problem is reported as an
`Issue` with a stable error code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from vramforge_estimator.units import GiB

from .common import DataPolicy, HardwareMode, Objective, ScanMode, SourceType, Strategy, VFModel

# ---------------------------------------------------------------- sources


class LoadingScope(StrEnum):
    AUTO_VERIFIED = "auto_verified"
    FULL_CHECKPOINT = "full_checkpoint"
    TEXT_ONLY = "text_only"


class ModelSourceRef(VFModel):
    source_type: SourceType = SourceType.HUGGINGFACE
    # HF id ("org/name"), HF URL (incl. /tree/<rev>), "local:<root>/<relative path>" or upload id.
    reference: str = Field(min_length=1, max_length=2048)
    revision: str | None = Field(default=None, max_length=256)
    loading_scope: LoadingScope = LoadingScope.AUTO_VERIFIED


class DatasetFormat(StrEnum):
    AUTO = "auto"
    PREFERENCE = "preference"  # prompt + chosen + rejected
    PROMPT_COMPLETION = "prompt_completion"
    PROMPT_ONLY = "prompt_only"
    MESSAGES = "messages"  # conversational list of {role, content}
    TEXT = "text"  # language modeling


class EmptySystemPolicy(StrEnum):
    OMIT = "omit"  # empty/blank system strings are not rendered as a system message
    KEEP = "keep"  # render a system message even when empty


class ColumnMapping(VFModel):
    """Which source columns play which role. Unmapped columns are metadata and never rendered."""

    format: DatasetFormat = DatasetFormat.AUTO
    system: str | None = None
    prompt: str | None = None
    chosen: str | None = None
    rejected: str | None = None
    completion: str | None = None
    messages: str | None = None
    text: str | None = None
    empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT


class DatasetSourceRef(VFModel):
    source_type: SourceType = SourceType.HUGGINGFACE
    reference: str = Field(min_length=1, max_length=2048)
    revision: str | None = Field(default=None, max_length=256)
    config: str | None = Field(default=None, max_length=256)
    # None = auto-select ("train" when present); the selection is reported in the result.
    split: str | None = Field(default=None, max_length=256)
    eval_split: str | None = Field(default=None, max_length=256)
    scan_mode: ScanMode = ScanMode.FULL
    sample_rows: int | None = Field(default=None, ge=1)
    # None = auto-detect; ambiguous detection ends the job with NEEDS_INPUT.
    mapping: ColumnMapping | None = None


# ---------------------------------------------------------------- training


class QuantFormat(StrEnum):
    NF4 = "nf4"
    FP4 = "fp4"


class ComputeDtype(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class QuantizationConfig(VFModel):
    """`enabled` is the UI's "Load in 4-bit" switch (bitsandbytes 4-bit)."""

    enabled: bool = False
    format: QuantFormat = QuantFormat.NF4
    double_quant: bool = True
    compute_dtype: ComputeDtype = ComputeDtype.AUTO


class LoraBias(StrEnum):
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class LoraConfig(VFModel):
    r: int = Field(default=16, ge=1, le=4096)
    alpha: float = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    # "auto_verified": the architecture's verified preset; "all-linear": PEFT semantics;
    # list: explicit module name suffixes or a single regex string.
    target_modules: Literal["auto_verified", "all-linear"] | list[str] = "auto_verified"
    exclude_modules: list[str] = Field(default_factory=list)
    modules_to_save: list[str] = Field(default_factory=list)
    bias: LoraBias = LoraBias.NONE
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    use_dora: bool = False
    use_rslora: bool = False


class Precision(StrEnum):
    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class OptimizerName(StrEnum):
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"


class AttentionBackend(StrEnum):
    AUTO = "auto"
    SDPA = "sdpa"
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"


class LinearAttentionKernel(StrEnum):
    """Kernel path of linear-attention (e.g. Gated DeltaNet) layers in the training environment.

    The path is chosen at import time by what is installed (docs/research/architecture-memory.md):
    `torch_fallback` (no flash-linear-attention / causal-conv1d) upcasts to fp32 and saves every
    chunk state; `fla` uses the fused kernels. AUTO = the environment profile's installed set.
    """

    AUTO = "auto"
    TORCH_FALLBACK = "torch_fallback"
    FLA = "fla"


class LoadDtype(StrEnum):
    """dtype passed when the trainer loads the model (TRL `model_init_kwargs["dtype"]`).

    TRL 1.14.1 loads string model ids in float32 unless a dtype is given
    (docs/research/loading-quantization-peft.md). AUTO = the profile preset, which the exported
    trainer config pins explicitly.
    """

    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class LossKernel(StrEnum):
    AUTO = "auto"
    STANDARD = "standard"
    CHUNKED = "chunked"
    LIGER = "liger"


class OffloadConfig(VFModel):
    parameters: bool = False
    optimizer: bool = False
    activations: bool = False


class TemplateOptions(VFModel):
    """Chat-template kwargs. `None` keeps the template's own default."""

    enable_thinking: bool | None = None


class TrainingConfig(VFModel):
    objective: Objective
    strategy: Strategy
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    # Units: SFT samples, DPO pairs, GRPO update completions. None = profile preset.
    microbatch_per_device: int | None = Field(default=None, ge=1, le=4096)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1, le=65536)
    gradient_checkpointing: bool = True
    precision: Precision = Precision.AUTO
    optimizer: OptimizerName = OptimizerName.ADAMW_TORCH
    load_dtype: LoadDtype = LoadDtype.AUTO
    attention_backend: AttentionBackend = AttentionBackend.AUTO
    linear_attention_kernel: LinearAttentionKernel = LinearAttentionKernel.AUTO
    loss_kernel: LossKernel = LossKernel.AUTO
    packing: bool = False
    pad_to_multiple_of: int | None = Field(default=None, ge=1, le=4096)
    offload: OffloadConfig = Field(default_factory=OffloadConfig)
    compile: bool = False
    num_devices: int = Field(default=1, ge=1, le=1024)
    data_policy: DataPolicy = DataPolicy.STRICT_NO_TRUNCATION
    backend_profile: str = Field(default="auto", max_length=256)
    template: TemplateOptions = Field(default_factory=TemplateOptions)
    seed: int = 42


# ---------------------------------------------------------------- method-specific


class ReferenceStrategy(StrEnum):
    AUTO = "auto"
    FROZEN_BASE_SWITCH = "frozen_base_switch"
    STANDALONE_MODEL = "standalone_model"
    PRECOMPUTED_LOG_PROBS = "precomputed_log_probs"


class DpoConfig(VFModel):
    reference_strategy: ReferenceStrategy = ReferenceStrategy.AUTO
    # Identity of a separate reference model (standalone); None = same checkpoint as the policy.
    reference_model: str | None = Field(default=None, max_length=2048)
    beta: float = Field(default=0.1, ge=0.0)
    loss_type: str = Field(default="sigmoid", max_length=64)
    precompute_batch_size: int | None = Field(default=None, ge=1)
    sync_ref_model: bool = False


class RewardKind(StrEnum):
    UNSPECIFIED = "unspecified"
    CPU_RULE = "cpu_rule"
    REMOTE = "remote"
    LOCAL_MODEL = "local_model"


class RewardConfig(VFModel):
    kind: RewardKind = RewardKind.UNSPECIFIED
    # For LOCAL_MODEL: HF id / local reference of the reward model (inspected like the policy).
    model_reference: str | None = Field(default=None, max_length=2048)
    on_training_gpu: bool = True


class RolloutBackend(StrEnum):
    TRANSFORMERS_SHARED_POLICY = "transformers_shared_policy"
    VLLM_COLOCATE = "vllm_colocate"
    VLLM_SERVER = "vllm_server"


DEFAULT_COMPLETION_BUDGETS: tuple[int, ...] = (1024, 2048, 4096, 8192)


class GrpoConfig(VFModel):
    num_generations: int = Field(default=4, ge=1, le=1024)
    generation_batch_size: int | None = Field(default=4, ge=1)
    steps_per_generation: int | None = Field(default=None, ge=1)
    num_iterations: int = Field(default=1, ge=1)
    # None = evaluate every candidate budget as a separate scenario (plan §5.4).
    completion_budget: int | None = Field(default=None, ge=1, le=1_048_576)
    completion_budget_candidates: list[int] = Field(
        default_factory=lambda: list(DEFAULT_COMPLETION_BUDGETS)
    )
    max_live_sequences: int | None = Field(default=None, ge=1)
    beta: float = Field(default=0.0, ge=0.0)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    rollout_backend: RolloutBackend = RolloutBackend.TRANSFORMERS_SHARED_POLICY


# ---------------------------------------------------------------- hardware, scope, margin


class HardwareConfig(VFModel):
    mode: HardwareMode = HardwareMode.CAPACITY_ONLY
    gpu_preset: str | None = Field(default=None, max_length=128)
    # Total device memory (preset or user-provided) and, optionally, what is actually free for
    # training. If `usable_bytes` already excludes other processes, keep external_reserved at 0.
    device_total_bytes: int | None = Field(default=None, ge=1)
    usable_bytes: int | None = Field(default=None, ge=1)
    external_reserved_bytes: int = Field(default=0, ge=0)
    num_gpus: int = Field(default=1, ge=1, le=1024)


class ScopeConfig(VFModel):
    include_evaluation: bool = False
    include_checkpoint_save: bool = False


class ProfilingConfig(VFModel):
    enabled: bool = False


class MarginPolicy(VFModel):
    """planning_margin = max(min_bytes, scenario_high × fraction) (plan §10.2)."""

    min_bytes: int = Field(default=2 * GiB, ge=0)
    fraction: float = Field(default=0.15, ge=0.0, le=10.0)


class AnalysisRequest(VFModel):
    schema_version: Literal["1.0"] = "1.0"
    model: ModelSourceRef
    dataset: DatasetSourceRef
    training: TrainingConfig
    dpo: DpoConfig = Field(default_factory=DpoConfig)
    grpo: GrpoConfig = Field(default_factory=GrpoConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    margin_policy: MarginPolicy = Field(default_factory=MarginPolicy)
