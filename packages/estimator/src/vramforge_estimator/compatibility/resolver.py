"""Compatibility resolution (plan.md §11): request + inventory + profile -> ResolvedConfig.

The resolver picks the analytic profile of the structurally matched architecture adapter, resolves
every knob to the value the pinned trainer actually uses and records requested -> resolved with a
reason. It never changes quality-affecting choices (rank, beta, generation count, budget) to save
memory (plan §11.1); options that cannot run are blockers, options that cannot take effect are
reported as not effective and the estimate uses the path that runs.
"""

from __future__ import annotations

from typing import Any, Literal

from vramforge_estimator import __version__
from vramforge_estimator.architectures import ArchitectureAdapter, get_adapter, match_adapter
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import (
    AnalysisRequest,
    ArchitectureFacts,
    AttentionBackend,
    BackendProfileInfo,
    BackendProfilesResponse,
    CompatibilityReport,
    ConfigResolution,
    DpoResolved,
    EffectiveDtypes,
    EmptySystemPolicy,
    EnvironmentInfo,
    ErrorCode,
    GrpoResolved,
    Issue,
    LinearAttentionKernel,
    LinearModule,
    LoadDtype,
    LoadingScope,
    LoraResolved,
    LossKernel,
    ModelInventory,
    Objective,
    OptimizerResolved,
    QuantFormat,
    QuantizationResolved,
    ReferenceStrategy,
    ResolvedConfig,
    RewardKind,
    Severity,
    Stage,
    Strategy,
    SupportEntry,
    TokenizerManifest,
    TrainingReadiness,
    WorkspaceAssumptions,
)
from vramforge_estimator.trainers.registry import trainer_id_for
from vramforge_estimator.trainers.trainable import rank_for

from .grpo_rules import resolve_grpo_batch
from .profiles import AnalyticProfile, EnvironmentProfile, load_registry
from .validation import validate_request

LINEAR_ATTENTION = "linear_attention"


def _issue(code: ErrorCode, message: str, component: str, **details: object) -> Issue:
    issue = make_issue(code, message, stage=Stage.RESOLVING, component=component)
    issue.details.update(details)
    return issue


def _warning(code: ErrorCode, message: str, component: str, **details: object) -> Issue:
    issue = make_issue(
        code, message, severity=Severity.WARNING, stage=Stage.RESOLVING, component=component
    )
    issue.details.update(details)
    return issue


def _support(profile: AnalyticProfile) -> list[SupportEntry]:
    return [
        SupportEntry(
            objective=r.objective,
            strategy=r.strategy,
            grade=r.grade,
            readiness=r.readiness,
            note=r.note,
        )
        for r in profile.support
    ]


class _Resolution:
    """Collects requested -> resolved records, blockers and warnings."""

    def __init__(self) -> None:
        self.records: list[ConfigResolution] = []
        self.not_effective: list[ConfigResolution] = []
        self.blockers: list[Issue] = []
        self.warnings: list[Issue] = []

    def set(self, field: str, requested: Any, resolved: Any, reason: str) -> Any:
        self.records.append(
            ConfigResolution(field=field, requested=requested, resolved=resolved, reason=reason)
        )
        return resolved

    def ineffective(self, field: str, requested: Any, resolved: Any, reason: str) -> None:
        self.set(field, requested, resolved, reason)
        self.not_effective.append(self.records[-1])


# ---------------------------------------------------------------- knob resolution


def _loading_scope(
    request: AnalysisRequest, facts: ArchitectureFacts, profile: AnalyticProfile, res: _Resolution
) -> Literal["full_checkpoint", "text_only"]:
    requested = request.model.loading_scope
    if not facts.has_vision:
        return res.set(
            "model.loading_scope",
            requested.value,
            "full_checkpoint",
            "비전 모듈이 없어 checkpoint 전체가 텍스트 디코더입니다.",
        )
    if requested is LoadingScope.TEXT_ONLY:
        if profile.loading.text_only.verified:
            return res.set("model.loading_scope", requested.value, "text_only", "검증된 경로")
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                profile.loading.text_only.reason,
                "model.loading_scope",
            )
        )
        return "full_checkpoint"
    arch = facts.architectures[0] if facts.architectures else None
    rule = next((r for r in profile.loading.scope_rules if r.architecture == arch), None)
    scope = rule.scope if rule else profile.loading.default_scope
    reason = rule.reason if rule else profile.loading.default_scope_reason
    return res.set("model.loading_scope", requested.value, scope, reason)


def _load_dtype(request: AnalysisRequest, profile: AnalyticProfile, res: _Resolution) -> str:
    requested = request.training.load_dtype
    dtype: str
    if requested is LoadDtype.AUTO:
        dtype = profile.loading.default_load_dtype
        return res.set(
            "training.load_dtype",
            requested.value,
            dtype,
            profile.loading.load_dtypes.get(dtype, "profile 기본값"),
        )
    dtype = requested.value
    note = profile.loading.load_dtypes.get(dtype)
    if note is None:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                f"load dtype {dtype}은 이 profile에서 지원하지 않습니다.",
                "training.load_dtype",
            )
        )
    return res.set("training.load_dtype", requested.value, dtype, note or "지원하지 않는 dtype")


def _emit_patterns(
    target: str | list[str], modules: list[LinearModule], inventory: ModelInventory
) -> list[str]:
    """What the exported LoraConfig.target_modules will contain."""
    if isinstance(target, list):
        return list(target)
    if target == "all-linear":
        return ["all-linear"]
    kinds = sorted({m.kind for m in modules})
    chosen = {m.name for m in modules}
    by_kind = {m.name for m in inventory.linear_modules if m.kind in kinds}
    # PEFT matches list entries as name suffixes; leaf kinds are exact only if they select
    # nothing else (e.g. vision `qkv` vs text `q_proj`).
    return kinds if by_kind == chosen else sorted(chosen)


def _lora(
    request: AnalysisRequest,
    inventory: ModelInventory,
    arch: ArchitectureAdapter,
    res: _Resolution,
) -> LoraResolved | None:
    t = request.training
    if t.strategy is Strategy.FULL:
        return None
    lora = t.lora
    try:
        modules = arch.lora_target_modules(inventory, lora.target_modules, lora.exclude_modules)
    except EstimatorError as exc:
        res.blockers.append(exc.issue)
        return None
    target_repr = (
        lora.target_modules if isinstance(lora.target_modules, str) else list(lora.target_modules)
    )
    if not modules:
        res.blockers.append(
            _issue(
                ErrorCode.CONFLICTING_OPTIONS,
                "LoRA target이 모델의 어떤 Linear 모듈과도 일치하지 않습니다.",
                "training.lora.target_modules",
            )
        )
        return None
    names = [m.name for m in modules]
    patterns = _emit_patterns(lora.target_modules, modules, inventory)
    res.set(
        "training.lora.target_modules",
        target_repr,
        patterns,
        f"구조별 검증 preset/요청을 모듈 {len(names)}개로 해석했습니다.",
    )
    params = sum(
        rank_for(m.name, lora.r, lora.rank_pattern) * (m.in_features + m.out_features)
        for m in modules
    )
    res.set(
        "lora.trainable_params",
        None,
        params,
        "P = Σ r_j × (in_j + out_j) (실제 모듈 차원, alpha와 무관).",
    )
    non_text = [m.name for m in modules if m.component.value != "text"]
    if non_text:
        res.warnings.append(
            _warning(
                ErrorCode.REQUESTED_OPTION_NOT_EFFECTIVE,
                f"LoRA가 텍스트 디코더 밖의 모듈 {len(non_text)}개(비전 타워 등)에도 붙습니다. "
                "텍스트 데이터에서는 이 adapter에 gradient가 생기지 않고 가중치만 상주합니다.",
                "training.lora.target_modules",
                modules=len(non_text),
            )
        )
    chunked_sft = t.objective is Objective.SFT and t.loss_kernel in (
        LossKernel.AUTO,
        LossKernel.CHUNKED,
    )
    if chunked_sft and any(m.kind == "lm_head" or m.name.endswith("lm_head") for m in modules):
        res.blockers.append(
            _issue(
                ErrorCode.CONFLICTING_OPTIONS,
                "SFT chunked_nll loss는 lm_head에 LoRA를 붙일 수 없습니다 (TRL ValueError).",
                "training.lora.target_modules",
            )
        )
    return LoraResolved(
        r=lora.r,
        alpha=lora.alpha,
        dropout=lora.dropout,
        target_module_patterns=patterns,
        target_modules=names,
        exclude_modules=list(lora.exclude_modules),
        modules_to_save=list(lora.modules_to_save),
        bias=lora.bias.value,
        rank_pattern=dict(lora.rank_pattern),
        use_dora=lora.use_dora,
        use_rslora=lora.use_rslora,
    )


def _optimizer(
    request: AnalysisRequest, profile: AnalyticProfile, param_dtype: str, res: _Resolution
) -> OptimizerResolved:
    name = request.training.optimizer
    rule = profile.optimizer_rule(name)
    if rule is None:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                f"optimizer {name.value}는 이 profile에서 지원하지 않습니다.",
                "training.optimizer",
            )
        )
        rule = profile.optimizers[0]
    state_dtype = "uint8" if rule.state_dtype == "uint8" else param_dtype
    res.set("training.optimizer", name.value, rule.name.value, rule.note or "profile 규칙")
    return OptimizerResolved(
        name=rule.name.value,
        states_per_param=rule.states_per_param,
        state_dtype=state_dtype,
        eight_bit=rule.eight_bit,
        block_size=rule.block_size,
        min_8bit_size=rule.min_8bit_size,
        paged=rule.paged,
        fused=rule.fused,
    )


def _attention(
    request: AnalysisRequest,
    facts: ArchitectureFacts,
    profile: AnalyticProfile,
    env: EnvironmentProfile,
    res: _Resolution,
) -> dict[str, str]:
    t = request.training
    requested = t.attention_backend
    default = profile.attention.by_layer_type.get("full_attention", "sdpa")
    attn = default if requested is AttentionBackend.AUTO else requested.value
    supported = profile.attention.supported.get("full_attention", [default])
    if attn not in supported:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                f"attention backend {attn}는 이 profile에서 지원하지 않습니다.",
                "training.attention_backend",
            )
        )
    res.set("training.attention_backend", requested.value, attn, "transformers 5.18 기본 sdpa")
    paths: dict[str, str] = {}
    for layer_type in facts.layer_types or ["full_attention"]:
        if layer_type == LINEAR_ATTENTION:
            kernel = env.kernels.linear_attention
            want = t.linear_attention_kernel
            if want is not LinearAttentionKernel.AUTO and want.value != kernel:
                res.ineffective(
                    "training.linear_attention_kernel",
                    want.value,
                    kernel,
                    "학습 환경에 해당 kernel 패키지가 없어 실제로 실행되는 경로로 계산합니다.",
                )
            paths[layer_type] = kernel
        else:
            paths[layer_type] = attn
    return paths


def _loss_path(request: AnalysisRequest, profile: AnalyticProfile, res: _Resolution) -> str:
    t = request.training
    rule = profile.loss[t.objective]
    if t.objective is Objective.SFT and t.loss_kernel is LossKernel.STANDARD:
        path = "hf_ce"
    else:
        path = rule.default
    if path not in rule.supported:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                f"loss 경로 {path}는 지원하지 않습니다.",
                "training.loss_kernel",
            )
        )
    return res.set("training.loss_kernel", t.loss_kernel.value, path, rule.note or rule.default)


def _dpo(
    request: AnalysisRequest, profile: AnalyticProfile, microbatch: int, res: _Resolution
) -> DpoResolved:
    d = request.dpo
    peft = request.training.strategy is not Strategy.FULL
    strategy = d.reference_strategy
    if strategy is ReferenceStrategy.AUTO:
        if d.reference_model:
            strategy = ReferenceStrategy.STANDALONE_MODEL
            reason = "별도 reference 모델이 지정되어 standalone_model로 해석했습니다."
        else:
            strategy = profile.dpo_reference_auto["peft" if peft else "full"]
            reason = (
                "PEFT: adapter를 끈 같은 모델이 reference입니다 (TRL ref_model=None)."
                if peft
                else "Full fine-tuning: 두 번째 전체 모델을 reference로 로드합니다."
            )
    else:
        reason = "요청한 전략"
    res.set("dpo.reference_strategy", d.reference_strategy.value, strategy.value, reason)
    separate = (
        d.reference_model
        if strategy is ReferenceStrategy.STANDALONE_MODEL
        and d.reference_model
        and d.reference_model != request.model.reference
        else None
    )
    res.set(
        "dpo.reference_model",
        d.reference_model,
        separate,
        "별도 checkpoint는 inventory를 분석하기 전까지 크기를 알 수 없습니다."
        if separate
        else "정책과 같은 checkpoint",
    )
    if strategy is ReferenceStrategy.PRECOMPUTED_LOG_PROBS:
        res.set(
            "dpo.precompute_batch_size",
            d.precompute_batch_size,
            d.precompute_batch_size or microbatch,
            "TRL 기본: precompute_ref_batch_size가 없으면 per_device_train_batch_size",
        )
    return DpoResolved(
        reference_strategy=strategy,
        beta=d.beta,
        loss_type=d.loss_type,
        precompute_batch_size=d.precompute_batch_size,
        sync_ref_model=d.sync_ref_model,
    )


def _grpo(
    request: AnalysisRequest, microbatch: int, accumulation: int, res: _Resolution
) -> GrpoResolved | None:
    g = request.grpo
    batch, issues = resolve_grpo_batch(
        num_generations=g.num_generations,
        microbatch=microbatch,
        accumulation=accumulation,
        generation_batch_size=g.generation_batch_size,
        steps_per_generation=g.steps_per_generation,
        num_iterations=g.num_iterations,
    )
    if batch is None:
        res.blockers.extend(issues)
        return None
    res.set(
        "grpo.generation_batch_size",
        g.generation_batch_size,
        batch.generation_batch_size,
        "TRL GRPOConfig.__post_init__ 공식",
    )
    res.set(
        "grpo.steps_per_generation",
        g.steps_per_generation,
        batch.steps_per_generation,
        "spg = gbs / (B_update × W) 또는 gradient_accumulation_steps",
    )
    if g.max_live_sequences is not None and g.max_live_sequences != batch.live_sequences:
        res.ineffective(
            "grpo.max_live_sequences",
            g.max_live_sequences,
            batch.live_sequences,
            "Transformers 생성 경로는 device의 generation batch 전체(C)를 한 번에 생성합니다.",
        )
    budgets = (
        [g.completion_budget]
        if g.completion_budget
        else sorted(set(g.completion_budget_candidates))
    )
    res.set(
        "grpo.completion_budget",
        g.completion_budget,
        budgets,
        "지정한 budget" if g.completion_budget else "budget이 없어 후보마다 시나리오를 만듭니다.",
    )
    on_gpu = g.reward.kind is RewardKind.LOCAL_MODEL and g.reward.on_training_gpu
    res.set("grpo.reward.on_training_gpu", g.reward.on_training_gpu, on_gpu, "reward 배치")
    if batch.unique_prompts > 1:
        res.warnings.append(
            _warning(
                ErrorCode.SAMPLER_DROPS_ROWS,
                f"GRPO RepeatSampler는 epoch마다 row 수 mod {batch.unique_prompts}개의 prompt를 "
                "버립니다. 전체 coverage는 batch 계획에서 확인합니다.",
                "grpo.generation_batch_size",
                unique_prompts=batch.unique_prompts,
            )
        )
    return GrpoResolved(
        num_generations=g.num_generations,
        generation_batch_size=batch.generation_batch_size,
        steps_per_generation=batch.steps_per_generation,
        num_iterations=g.num_iterations,
        completion_budgets=budgets,
        budget_explicit=g.completion_budget is not None,
        beta=g.beta,
        reference_needed=g.beta != 0,
        reward_kind=g.reward.kind,
        reward_model_reference=g.reward.model_reference,
        rollout_backend=g.rollout_backend,
        live_sequences=batch.live_sequences,
        update_microbatch=microbatch,
        accumulation=accumulation,
    )


def _workspace(profile: AnalyticProfile) -> WorkspaceAssumptions:
    ws = profile.workspace
    return WorkspaceAssumptions(
        cuda_context_bytes=(int(ws.cuda_context_bytes.low), int(ws.cuda_context_bytes.high)),
        library_workspace_bytes=(
            int(ws.library_workspace_bytes.low),
            int(ws.library_workspace_bytes.high),
        ),
        allocator_slack_fraction=(
            ws.allocator_slack_fraction.low,
            ws.allocator_slack_fraction.high,
        ),
        notes=[
            f"cuda_context: {ws.cuda_context_bytes.source}",
            f"library_workspace: {ws.library_workspace_bytes.source}",
            f"allocator_slack: {ws.allocator_slack_fraction.source}",
        ],
    )


def _readiness(
    request: AnalysisRequest, base: TrainingReadiness, res: _Resolution
) -> TrainingReadiness:
    if res.blockers:
        return TrainingReadiness.UNSUPPORTED
    readiness = base
    if request.training.objective is Objective.GRPO:
        reward = request.grpo.reward
        if reward.kind is RewardKind.UNSPECIFIED:
            res.warnings.append(
                _warning(
                    ErrorCode.GRPO_REWARD_UNSPECIFIED,
                    "reward가 지정되지 않았습니다. policy·rollout 메모리만 계산한 조건부 결과이며 "
                    "실행용 설정은 만들 수 없습니다.",
                    "grpo.reward",
                )
            )
            readiness = TrainingReadiness.CONDITIONAL
        elif reward.kind in (RewardKind.CPU_RULE, RewardKind.REMOTE) or not reward.on_training_gpu:
            readiness = TrainingReadiness.CONDITIONAL
    if request.dataset.eval_split and not request.scope.include_evaluation:
        # Training will run an evaluation phase whose memory is outside the computed scope.
        res.warnings.append(
            _warning(
                ErrorCode.PROFILE_SCOPE_INCOMPLETE,
                "평가 split이 지정되었지만 평가 단계 메모리는 계산 범위에서 제외했습니다 "
                "(조건부 결과).",
                "scope.include_evaluation",
            )
        )
        readiness = TrainingReadiness.CONDITIONAL
    return readiness


# ---------------------------------------------------------------- public API


def resolve(
    request: AnalysisRequest,
    inventory: ModelInventory,
    tokenizer: TokenizerManifest | None,
) -> tuple[ResolvedConfig | None, CompatibilityReport]:
    """Pick the profile (data preservation → objective preservation → compatibility → memory
    efficiency, plan §11.1), resolve every knob and record requested vs resolved."""
    res = _Resolution()
    for issue in validate_request(request):
        (res.blockers if issue.severity is Severity.ERROR else res.warnings).append(issue)
    facts = inventory.facts
    adapter_id = match_adapter(facts)
    if adapter_id is None:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_ARCHITECTURE,
                "등록된 architecture adapter가 이 모델 구조를 지원하지 않습니다.",
                "model",
                model_type=facts.model_type,
            )
        )
        return None, CompatibilityReport(
            readiness=TrainingReadiness.UNSUPPORTED, blockers=res.blockers, warnings=res.warnings
        )
    registry = load_registry()
    profile = registry.profile_for_adapter(adapter_id)
    wanted = request.training.backend_profile
    if wanted != "auto":
        chosen = registry.analytic.get(wanted)
        if chosen is not None and chosen.architecture_adapter != adapter_id:
            res.blockers.append(
                _issue(
                    ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                    "선택한 backend profile은 이 모델 구조용이 아닙니다.",
                    "training.backend_profile",
                )
            )
        profile = chosen or profile
    if profile is None:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                "이 architecture adapter에 등록된 backend profile이 없습니다.",
                "training.backend_profile",
            )
        )
        return None, CompatibilityReport(
            architecture_adapter=adapter_id,
            readiness=TrainingReadiness.UNSUPPORTED,
            blockers=res.blockers,
            warnings=res.warnings,
        )
    env = registry.environment_for(profile)
    t = request.training
    rule = profile.support_rule(t.objective, t.strategy)
    if rule.grade is None:
        res.blockers.append(
            _issue(ErrorCode.UNSUPPORTED_BACKEND_COMBINATION, rule.note, "training.strategy")
        )
    if inventory.quantized_checkpoint_format:
        res.blockers.append(
            _issue(
                ErrorCode.UNSUPPORTED_MODEL_FORMAT,
                f"사전 양자화된 checkpoint({inventory.quantized_checkpoint_format})는 bitsandbytes "
                "학습 입력으로 쓰지 않습니다.",
                "model",
            )
        )
    arch = get_adapter(adapter_id)
    cfg = _build(request, inventory, tokenizer, profile, env, arch, res)
    if not res.blockers:
        # The adapter validates trainability (e.g. modules_to_save on a 4-bit module).
        try:
            arch.trainable_groups(inventory, cfg)
        except EstimatorError as exc:
            res.blockers.append(exc.issue)
    readiness = _readiness(request, rule.readiness, res)
    report = CompatibilityReport(
        profile_id=profile.id,
        architecture_adapter=adapter_id,
        support_grade=rule.grade,
        readiness=readiness,
        support=_support(profile),
        blockers=res.blockers,
        warnings=res.warnings,
        not_effective=res.not_effective,
    )
    return (None if res.blockers else cfg), report


def _build(
    request: AnalysisRequest,
    inventory: ModelInventory,
    tokenizer: TokenizerManifest | None,
    profile: AnalyticProfile,
    env: EnvironmentProfile,
    arch: ArchitectureAdapter,
    res: _Resolution,
) -> ResolvedConfig:
    t = request.training
    facts = inventory.facts
    scope = _loading_scope(request, facts, profile, res)
    load_dtype = _load_dtype(request, profile, res)
    quantized = t.strategy is Strategy.QLORA
    compute = "bfloat16"
    res.set("training.precision", t.precision.value, "bf16", "TRL 기본 bf16 mixed precision")
    if quantized:
        adapter_dtype = "bfloat16"
        adapter_reason = "TRL이 양자화 모델의 학습 파라미터를 bf16으로 변환합니다."
    elif t.strategy is Strategy.LORA:
        adapter_dtype = "float32"
        adapter_reason = "PEFT autocast_adapter_dtype=True: 비양자화 LoRA adapter는 fp32입니다."
    else:
        adapter_dtype = load_dtype
        adapter_reason = "Full fine-tuning: 학습 파라미터 = 모델 파라미터 (master copy 없음)."
    res.set("effective_dtypes.adapter", None, adapter_dtype, adapter_reason)
    quant = QuantizationResolved(enabled=False)
    if quantized:
        q = t.quantization
        fmt = q.format.value
        if fmt not in profile.quantization.formats:
            res.blockers.append(
                _issue(
                    ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                    f"양자화 형식 {fmt}은 지원하지 않습니다.",
                    "training.quantization.format",
                )
            )
        quant = QuantizationResolved(
            enabled=True,
            method="bnb_nf4" if q.format is QuantFormat.NF4 else "bnb_fp4",
            double_quant=q.double_quant,
            blocksize=profile.quantization.blocksize,
            nested_blocksize=profile.quantization.nested_blocksize if q.double_quant else None,
            quant_storage_dtype=profile.quantization.quant_storage,
            compute_dtype=profile.quantization.compute_dtype,
            skip_module_patterns=list(profile.quantization.skip_modules),
        )
        res.set(
            "training.quantization",
            q.model_dump(mode="json"),
            quant.model_dump(mode="json"),
            profile.quantization.note or "profile 4-bit preset",
        )
    lora = _lora(request, inventory, arch, res)
    optimizer = _optimizer(request, profile, adapter_dtype, res)
    preset = profile.presets[t.objective]
    microbatch = t.microbatch_per_device or preset.microbatch
    accumulation = t.gradient_accumulation_steps or preset.accumulation
    res.set(
        "training.microbatch_per_device",
        t.microbatch_per_device,
        microbatch,
        "요청값" if t.microbatch_per_device else "제품 preset",
    )
    res.set(
        "training.gradient_accumulation_steps",
        t.gradient_accumulation_steps,
        accumulation,
        "요청값" if t.gradient_accumulation_steps else "제품 preset",
    )
    res.set(
        "training.gradient_checkpointing",
        t.gradient_checkpointing,
        t.gradient_checkpointing,
        profile.checkpointing.note or "decoder layer마다 checkpoint",
    )
    paths = _attention(request, facts, profile, env, res)
    loss_path = _loss_path(request, profile, res)
    processing = "tokenizer"
    res.set(
        "training.processing_class",
        None,
        processing,
        "학습 환경에 Pillow·torchvision이 없어 AutoTokenizer를 processing_class로 명시합니다."
        if facts.has_vision
        else "tokenizer",
    )
    template_kwargs: dict[str, Any] = {}
    thinking = t.template.enable_thinking
    if thinking is not None:
        if tokenizer is not None and "enable_thinking" not in tokenizer.template_kwargs:
            res.ineffective(
                "training.template.enable_thinking",
                thinking,
                None,
                "chat template이 enable_thinking을 사용하지 않습니다.",
            )
        else:
            template_kwargs["enable_thinking"] = thinking
    dpo = _dpo(request, profile, microbatch, res) if t.objective is Objective.DPO else None
    grpo = _grpo(request, microbatch, accumulation, res) if t.objective is Objective.GRPO else None
    mapping = request.dataset.mapping
    return ResolvedConfig(
        profile_id=profile.id,
        profile_version=profile.version,
        environment_id=env.id,
        dependency_lock_digest=env.dependency_lock_digest,
        architecture_adapter=profile.architecture_adapter,
        trainer_adapter=trainer_id_for(t.objective),
        # preprocessing adapters follow the same "trl-<version>-<objective>" naming
        preprocessing_adapter=trainer_id_for(t.objective),
        objective=t.objective,
        strategy=t.strategy,
        loading_scope=scope,
        load_dtype=load_dtype,
        processing_class=processing,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=load_dtype,
            compute=compute,
            adapter=adapter_dtype,
            gradient=adapter_dtype,
            optimizer_state=optimizer.state_dtype,
            master_weights=None,
            logits="float32",
            loss="float32",
            kv_cache=load_dtype,
            recurrent_state="float32",
        ),
        quantization=quant,
        upcast_to_fp32_patterns=[],
        lora=lora,
        trainable_full_patterns=[".*"]
        if t.strategy is Strategy.FULL
        else list(t.lora.modules_to_save),
        optimizer=optimizer,
        microbatch=microbatch,
        accumulation=accumulation,
        pad_to_multiple_of=t.pad_to_multiple_of,
        gradient_checkpointing=t.gradient_checkpointing,
        checkpointing_granularity="per_decoder_layer" if t.gradient_checkpointing else "none",
        attention_path_by_layer_type=paths,
        loss_path=loss_path,
        use_cache_during_training=False,
        dpo=dpo,
        grpo=grpo,
        workspace=_workspace(profile),
        template_kwargs=template_kwargs,
        empty_system_policy=mapping.empty_system_policy if mapping else EmptySystemPolicy.OMIT,
        resolutions=res.records,
    )


def support_for(facts: ArchitectureFacts) -> tuple[str | None, list[SupportEntry]]:
    """Adapter id (or None) and the objective × strategy support grades for these facts."""
    adapter_id = match_adapter(facts)
    if adapter_id is None:
        return None, []
    profile = load_registry().profile_for_adapter(adapter_id)
    return adapter_id, _support(profile) if profile else []


def backend_profiles() -> BackendProfilesResponse:
    registry = load_registry()
    return BackendProfilesResponse(
        estimator_version=__version__,
        profiles=[
            BackendProfileInfo(
                profile_id=p.id,
                profile_version=p.version,
                architecture_adapter=p.architecture_adapter,
                description=p.description,
                model_types=list(p.model_types),
                environment_id=p.environment,
                support=_support(p),
            )
            for p in registry.analytic.values()
        ],
        environments=[
            EnvironmentInfo(
                environment_id=e.id,
                description=e.description,
                packages=dict(e.packages),
                dependency_lock_digest=e.dependency_lock_digest,
            )
            for e in registry.environments.values()
        ],
        hardware_presets=registry.gpu_presets(),
        gpu_worker_connected=False,
    )


__all__ = ["backend_profiles", "resolve", "support_for", "validate_request"]
