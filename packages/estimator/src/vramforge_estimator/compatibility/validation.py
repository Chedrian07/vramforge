"""Cross-field request checks that need no network (plan.md §5, §11.3).

Errors block job creation; warnings are reported with the result. Options the pinned backend
cannot run, or that the memory model does not cover, are rejected rather than silently ignored
(plan §2.2, §11.3); an option that runs but cannot take effect is a REQUESTED_OPTION_NOT_EFFECTIVE
warning and the estimate uses the path that actually runs.
"""

from __future__ import annotations

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    AnalysisRequest,
    AttentionBackend,
    ComputeDtype,
    ErrorCode,
    HardwareMode,
    Issue,
    LinearAttentionKernel,
    LoadDtype,
    LoraBias,
    LossKernel,
    Objective,
    Precision,
    ReferenceStrategy,
    RewardKind,
    RolloutBackend,
    ScanMode,
    Severity,
    Stage,
    Strategy,
)

from .grpo_rules import resolve_grpo_batch
from .profiles import load_registry

# Batch presets used for validation before the architecture (and its profile) is known. Both
# analytic profiles carry the same values; test_profiles checks they stay in sync.
PRESET_MICROBATCH = {Objective.SFT: 1, Objective.DPO: 1, Objective.GRPO: 1}
PRESET_ACCUMULATION = {Objective.SFT: 8, Objective.DPO: 8, Objective.GRPO: 4}
# Same bound as `GrpoConfig.completion_budget` (schemas/request.py); a budget is a finite
# max_new_tokens >= 1 (docs/research/trl-grpo.md §9).
MAX_COMPLETION_BUDGET = 1_048_576


def _err(code: ErrorCode, message: str, component: str, **details: object) -> Issue:
    issue = make_issue(code, message, stage=Stage.REQUEST, component=component)
    issue.details.update(details)
    return issue


def _warn(code: ErrorCode, message: str, component: str, **details: object) -> Issue:
    issue = make_issue(
        code, message, severity=Severity.WARNING, stage=Stage.REQUEST, component=component
    )
    issue.details.update(details)
    return issue


def _unsupported(message: str, component: str) -> Issue:
    return _err(ErrorCode.UNSUPPORTED_BACKEND_COMBINATION, message, component)


def _strategy_issues(request: AnalysisRequest) -> list[Issue]:
    t = request.training
    four_bit = t.quantization.enabled
    if t.strategy is Strategy.QLORA and not four_bit:
        return [
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "QLoRA는 4-bit 로딩이 필요합니다. Load in 4-bit를 켜거나 LoRA를 선택하세요.",
                "training.quantization.enabled",
            )
        ]
    if t.strategy is Strategy.FULL and four_bit:
        return [
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "Full fine-tuning과 4-bit 로딩은 함께 쓸 수 없습니다 (양자화 모델은 adapter 없이 "
                "학습할 수 없음).",
                "training.strategy",
            )
        ]
    if t.strategy is Strategy.LORA and four_bit:
        return [
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "4-bit 로딩을 켜면 QLoRA를 선택해야 합니다.",
                "training.strategy",
            )
        ]
    return []


def _runtime_issues(request: AnalysisRequest) -> list[Issue]:
    t = request.training
    issues: list[Issue] = []
    if t.num_devices > 1 or request.hardware.num_gpus > 1:
        issues.append(
            _err(
                ErrorCode.UNSUPPORTED_DISTRIBUTED_TOPOLOGY,
                "다중 GPU 구성은 아직 지원하지 않습니다. 총 용량을 GPU 수로 나눈 값을 대신 "
                "제공하지 않습니다.",
                "training.num_devices",
                num_devices=t.num_devices,
                num_gpus=request.hardware.num_gpus,
            )
        )
    if t.packing:
        issues.append(
            _unsupported(
                "packing은 엄격 무절단 모드에서 지원하지 않습니다 (절단·분할 또는 다른 샘플과의 "
                "연결이 발생).",
                "training.packing",
            )
        )
    if t.offload.parameters or t.offload.optimizer or t.offload.activations:
        issues.append(
            _unsupported(
                "offload는 이 backend profile에서 실행 경로나 메모리 모델이 없습니다. "
                "offload를 끄고 다시 요청하세요.",
                "training.offload",
            )
        )
    if t.compile:
        issues.append(_unsupported("torch.compile 메모리 profile이 없습니다.", "training.compile"))
    if t.loss_kernel is LossKernel.LIGER:
        issues.append(_unsupported("liger-kernel이 학습 환경에 없습니다.", "training.loss_kernel"))
    if t.loss_kernel is LossKernel.CHUNKED and t.objective is not Objective.SFT:
        issues.append(
            _unsupported(
                "DPO·GRPO의 chunked log-prob 경로는 liger-kernel이 필요해 지원하지 않습니다.",
                "training.loss_kernel",
            )
        )
    if t.attention_backend is AttentionBackend.FLASH_ATTENTION_2:
        issues.append(
            _unsupported("flash-attn이 학습 환경에 없습니다.", "training.attention_backend")
        )
    if t.precision in (Precision.FP16, Precision.FP32):
        issues.append(
            _unsupported(
                "bf16 mixed precision 외의 precision은 메모리 모델이 없습니다.",
                "training.precision",
            )
        )
    if t.load_dtype is LoadDtype.FLOAT16:
        issues.append(
            _unsupported("float16 로드는 검증된 profile이 없습니다.", "training.load_dtype")
        )
    if t.quantization.enabled and t.quantization.compute_dtype in (
        ComputeDtype.FLOAT32,
        ComputeDtype.FLOAT16,
    ):
        issues.append(
            _unsupported(
                "bitsandbytes compute dtype은 bfloat16만 지원합니다 (fp32 compute는 별도 profile).",
                "training.quantization.compute_dtype",
            )
        )
    if t.strategy is not Strategy.FULL and t.lora.use_dora:
        issues.append(
            _unsupported("DoRA의 메모리 모델이 구현되지 않았습니다.", "training.lora.use_dora")
        )
    if t.linear_attention_kernel is LinearAttentionKernel.FLA:
        issues.append(
            _warn(
                ErrorCode.REQUESTED_OPTION_NOT_EFFECTIVE,
                "학습 환경에 flash-linear-attention·causal-conv1d가 없어 linear attention은 torch "
                "fallback으로 실행됩니다. 메모리는 실제 실행 경로로 계산합니다.",
                "training.linear_attention_kernel",
            )
        )
    return issues


def _lm_head_issues(request: AnalysisRequest) -> list[Issue]:
    t = request.training
    chunked = t.objective is Objective.SFT and t.loss_kernel in (
        LossKernel.AUTO,
        LossKernel.CHUNKED,
    )
    if not chunked or t.strategy is Strategy.FULL:
        return []
    issues: list[Issue] = []
    targets = t.lora.target_modules
    if isinstance(targets, list) and any(n == "lm_head" or n.endswith(".lm_head") for n in targets):
        issues.append(
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "SFT chunked_nll loss는 lm_head에 LoRA를 붙일 수 없습니다 (TRL ValueError).",
                "training.lora.target_modules",
            )
        )
    if any(m.endswith("lm_head") for m in t.lora.modules_to_save):
        issues.append(
            _warn(
                ErrorCode.UNKNOWN_MEMORY_COMPONENT,
                "SFT chunked_nll과 modules_to_save=lm_head 조합을 TRL이 허용하는지는 확인되지 "
                "않았습니다. 학습되는 lm_head의 weight-grad 누적으로 계산합니다.",
                "training.lora.modules_to_save",
            )
        )
    return issues


def _dpo_issues(request: AnalysisRequest) -> list[Issue]:
    d = request.dpo
    strategy = request.training.strategy
    issues: list[Issue] = []
    if d.reference_strategy is ReferenceStrategy.FROZEN_BASE_SWITCH and strategy is Strategy.FULL:
        issues.append(
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "frozen_base_switch reference는 끌 수 있는 adapter가 필요합니다 "
                "(Full fine-tuning과 함께 쓸 수 없음).",
                "dpo.reference_strategy",
            )
        )
    if d.sync_ref_model and d.reference_strategy is ReferenceStrategy.PRECOMPUTED_LOG_PROBS:
        issues.append(
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "precompute reference log-prob과 sync_ref_model은 함께 쓸 수 없습니다.",
                "dpo.sync_ref_model",
            )
        )
    if d.sync_ref_model and strategy is not Strategy.FULL:
        issues.append(
            _err(
                ErrorCode.CONFLICTING_OPTIONS,
                "sync_ref_model은 PEFT(LoRA·QLoRA)와 함께 쓸 수 없습니다.",
                "dpo.sync_ref_model",
            )
        )
    if "sft" in [p.strip() for p in d.loss_type.split(",")]:
        issues.append(
            _unsupported(
                "loss_type에 sft가 포함되면 추가 cross-entropy 버퍼의 메모리 모델이 없습니다.",
                "dpo.loss_type",
            )
        )
    if d.reference_model and d.reference_strategy in (
        ReferenceStrategy.FROZEN_BASE_SWITCH,
        ReferenceStrategy.PRECOMPUTED_LOG_PROBS,
    ):
        issues.append(
            _warn(
                ErrorCode.REQUESTED_OPTION_NOT_EFFECTIVE,
                "선택한 reference 전략에서는 별도 reference 모델이 사용되지 않습니다.",
                "dpo.reference_model",
            )
        )
    return issues


def _adapter_off_reference_issues(request: AnalysisRequest) -> list[Issue]:
    """plan §5.3: an adapter-off reference (DPO frozen_base_switch, GRPO beta != 0 with PEFT) is
    the base model only if no base parameter trains. PEFT's disable_adapter() keeps trained base
    biases (peft 0.21.2 tuners/tuners_utils.py:556-565 warns the output differs from the base)."""
    t = request.training
    if t.strategy is Strategy.FULL or t.lora.bias is LoraBias.NONE:
        return []
    if t.objective is Objective.DPO:
        rs = request.dpo.reference_strategy
        adapter_off = rs is ReferenceStrategy.FROZEN_BASE_SWITCH or (
            rs is ReferenceStrategy.AUTO and not request.dpo.reference_model
        )
    else:
        adapter_off = t.objective is Objective.GRPO and request.grpo.beta != 0
    if not adapter_off:
        return []
    return [
        _warn(
            ErrorCode.CONFLICTING_OPTIONS,
            f"bias='{t.lora.bias.value}'로 base 모델의 bias를 학습하면 adapter를 끈 reference가 "
            "원래 base 모델과 달라집니다 (PEFT disable_adapter는 학습된 bias를 되돌리지 않음). "
            "원래 모델을 reference로 쓰려면 bias='none' 또는 별도 reference 모델을 선택하세요.",
            "training.lora.bias",
            bias=t.lora.bias.value,
        )
    ]


def _grpo_issues(request: AnalysisRequest) -> list[Issue]:
    g = request.grpo
    t = request.training
    issues: list[Issue] = []
    if g.rollout_backend is not RolloutBackend.TRANSFORMERS_SHARED_POLICY:
        issues.append(
            _unsupported(
                "vLLM rollout은 별도 엔진의 가중치·KV pool 메모리를 모델링하지 않아 지원하지 "
                "않습니다.",
                "grpo.rollout_backend",
            )
        )
    _, batch_issues = resolve_grpo_batch(
        num_generations=g.num_generations,
        microbatch=t.microbatch_per_device or PRESET_MICROBATCH[Objective.GRPO],
        accumulation=t.gradient_accumulation_steps or PRESET_ACCUMULATION[Objective.GRPO],
        generation_batch_size=g.generation_batch_size,
        steps_per_generation=g.steps_per_generation,
        num_iterations=g.num_iterations,
    )
    issues.extend(batch_issues)
    if g.reward.kind is RewardKind.LOCAL_MODEL and not g.reward.model_reference:
        issues.append(
            _err(
                ErrorCode.INVALID_REQUEST,
                "local reward 모델을 쓰려면 reward 모델 주소(model_reference)가 필요합니다.",
                "grpo.reward.model_reference",
            )
        )
    if g.completion_budget is None and not g.completion_budget_candidates:
        issues.append(
            _err(
                ErrorCode.GRPO_BUDGET_UNSPECIFIED,
                "completion budget 또는 후보 목록이 필요합니다 (무제한 생성의 피크는 계산할 수 "
                "없음).",
                "grpo.completion_budget",
            )
        )
    invalid = sorted(
        {b for b in g.completion_budget_candidates if not 1 <= b <= MAX_COMPLETION_BUDGET}
    )
    if g.completion_budget is None and invalid:
        # The schema bounds `completion_budget` but not the candidate items.
        issues.append(
            _err(
                ErrorCode.INVALID_REQUEST,
                f"completion budget 후보는 1 이상 {MAX_COMPLETION_BUDGET:,} 이하의 정수여야 "
                "합니다.",
                "grpo.completion_budget_candidates",
                invalid=invalid[:10],
            )
        )
    return issues


def _hardware_issues(request: AnalysisRequest) -> list[Issue]:
    hw = request.hardware
    if hw.mode is HardwareMode.GPU_PRESET:
        if not hw.gpu_preset:
            return [
                _err(ErrorCode.INVALID_REQUEST, "GPU preset을 선택하세요.", "hardware.gpu_preset")
            ]
        if load_registry().gpu(hw.gpu_preset) is None and hw.device_total_bytes is None:
            return [
                _err(
                    ErrorCode.INVALID_REQUEST,
                    "알 수 없는 GPU preset입니다.",
                    "hardware.gpu_preset",
                    gpu_preset=hw.gpu_preset,
                )
            ]
    if hw.mode is HardwareMode.CUSTOM and hw.device_total_bytes is None and hw.usable_bytes is None:
        return [
            _err(
                ErrorCode.INVALID_REQUEST,
                "직접 입력 모드에는 GPU 총 용량 또는 사용 가능 용량이 필요합니다.",
                "hardware.device_total_bytes",
            )
        ]
    if (
        hw.usable_bytes is not None
        and hw.device_total_bytes is not None
        and hw.usable_bytes > hw.device_total_bytes
    ):
        return [
            _err(
                ErrorCode.INVALID_REQUEST,
                "사용 가능 용량이 GPU 총 용량보다 클 수 없습니다.",
                "hardware.usable_bytes",
            )
        ]
    return []


def validate_request(request: AnalysisRequest) -> list[Issue]:
    """Cross-field checks that need no network (e.g. 4-bit + full fine-tune, LoRA + 4-bit,
    multi-GPU without a topology adapter, sample scan claims). Errors block job creation."""
    t = request.training
    issues = [*_strategy_issues(request), *_runtime_issues(request), *_lm_head_issues(request)]
    if t.objective is Objective.DPO:
        issues += _dpo_issues(request)
    if t.objective is Objective.GRPO:
        issues += _grpo_issues(request)
    issues += _adapter_off_reference_issues(request)
    issues += _hardware_issues(request)
    if request.dataset.scan_mode is ScanMode.SAMPLE:
        issues.append(
            _warn(
                ErrorCode.SCAN_PARTIAL,
                "샘플 분석은 전체 데이터의 최대 길이를 확인하지 않으므로 결과를 전체 분석처럼 "
                "표시하지 않습니다.",
                "dataset.scan_mode",
            )
        )
    if request.profiling.enabled:
        issues.append(
            _warn(
                ErrorCode.GPU_WORKER_UNAVAILABLE,
                "GPU 검증 worker가 연결되어 있지 않아 정적 분석만 수행합니다.",
                "profiling.enabled",
            )
        )
    if t.backend_profile != "auto" and t.backend_profile not in load_registry().analytic:
        issues.append(
            _unsupported(
                "등록되지 않은 backend profile입니다.",
                "training.backend_profile",
            )
        )
    return issues


__all__ = ["MAX_COMPLETION_BUDGET", "PRESET_ACCUMULATION", "PRESET_MICROBATCH", "validate_request"]
