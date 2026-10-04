"""Shape ledgers: one training step, a no-grad forward and a generation (plan.md §9.5, §9.7).

Composition rules: docs/research/architecture-memory.md implications C (GC vs no-GC timepoints,
transient factors), E (generation cache) and §3 (SDPA mask rules); dtype rules of
docs/research/loading-quantization-peft.md §3.6-3.7 and docs/research/trl-grpo.md R3.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from vramforge_estimator.schemas import (
    AllocationCategory,
    AllocationSpec,
    Evidence,
    Objective,
    ResolvedConfig,
    SequenceShape,
    Strategy,
)
from vramforge_estimator.units import canonical_dtype, dtype_bytes

from . import activations as act
from .base import FINAL_HIDDEN_SUFFIX, GenerationTimepoints, StepTimepoints, final_hidden_alias
from .structure import (
    EXECUTED_COMPONENTS,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    SLIDING_ATTENTION,
    Layer,
    ModelStructure,
)
from .trainable import Trainability
from .weights import BNB_BLOCKSIZE, quantized_modules

DOC = "methodology-architectures.md"
SUPPORTED_ACT_DTYPES = frozenset({"bfloat16", "float16"})
# accelerate native AMP: ResolvedConfig.mixed_precision -> autocast dtype ("none": no autocast).
AMP_DTYPES = {"bf16": "bfloat16", "fp16": "float16"}

# Transient factors (peak - retained) / S_max, docs/research/architecture-memory.md §4.4:
# low = smallest CPU/real-dims measurement, high = the research default (or a larger measurement).
K_GC_FORWARD = {"full": (0.26, 0.45), "lora_shared": (0.32, 0.45), "lora_fp32": (0.37, 0.45)}
K_GC_BACKWARD = {"full": (0.60, 1.35), "lora_shared": (1.05, 1.35), "lora_fp32": (1.14, 1.36)}
K_NOGC_FORWARD = {"full": (0.04, 0.06), "lora_shared": (0.04, 0.09), "lora_fp32": (0.19, 0.20)}
K_NOGC_BACKWARD = {"full": (0.0, 0.15), "lora_shared": (0.12, 0.16), "lora_fp32": (0.12, 0.16)}

_SDPA_AUTO = frozenset({"sdpa", "sdpa_flash", "sdpa_cudnn", "flash", "cudnn"})
_SDPA_MEM = frozenset({"sdpa_mem_efficient", "sdpa_efficient", "mem_efficient", "efficient"})
_LINEAR_PATHS: dict[str, act.LinearKernel] = {
    "torch_fallback": "torch",
    "torch": "torch",
    "fla": "fla",
}
# Defaults when the resolved config leaves a layer type out: transformers 5.18 picks sdpa
# (§1.1) and the pinned environment has no fla/causal-conv1d (§2.1, trl-grpo §4.5).
DEFAULT_ATTENTION = "sdpa"
DEFAULT_LINEAR = "torch_fallback"
_GROUP_REFS = {
    "norms": "act-norm",
    "mlp": "act-norm",
    "attention": "act-attention",
    "linear_attention": "act-linear",
    "lora": "act-lora",
    "mask": "act-mask",
}


# ---------------------------------------------------------------- small helpers


def _ceil(x: float) -> int:
    return math.ceil(x)


def spec(
    name: str,
    category: AllocationCategory,
    low: int | None,
    high: int | None,
    live_at: Sequence[str],
    *,
    ref: str,
    evidence: Evidence | None = None,
    shape: str | None = None,
    dims: dict[str, int] | None = None,
    dtype: str | None = None,
    count: int = 1,
    saved: bool = False,
    recompute: str | None = None,
    alias: str | None = None,
    note: str | None = None,
) -> AllocationSpec:
    if low is None or high is None:
        low = high = None
        evidence = Evidence.UNKNOWN
    elif evidence is None:
        evidence = Evidence.ANALYTIC if low == high else Evidence.ASSUMPTION
    return AllocationSpec(
        name=name,
        category=category,
        shape_expression=shape,
        dims=dims or {},
        dtype=dtype,
        count=count,
        bytes_low=low,
        bytes_high=high,
        live_at=list(dict.fromkeys(live_at)),
        saved_for_backward=saved,
        recompute_group=recompute,
        storage_alias_group=alias,
        evidence=evidence,
        formula_ref=f"{DOC}#{ref}",
        note=note,
    )


def amp_dtype(cfg: ResolvedConfig) -> str | None:
    """Autocast dtype of the run: TRL bf16/fp16 mixed precision = accelerate native AMP, which wraps
    the model forward in `torch.autocast` (research §2.4). Read from
    `ResolvedConfig.mixed_precision`; None = no autocast ("none", e.g. fp32 training)."""
    return AMP_DTYPES.get(cfg.mixed_precision)


def autocast_on(cfg: ResolvedConfig) -> bool:
    return amp_dtype(cfg) is not None


def dtype_issue(cfg: ResolvedConfig, autocast: bool | None = None) -> str | None:
    """Why the verified saved-set formulas do not apply to the precision setup (None = they do).

    They are measured for 16-bit loads, without autocast or with autocast to the load dtype. A
    different autocast dtype casts every Linear/conv input and weight (copies saved per module,
    as measured for fp32 weights, AM §4.3), which is not modeled."""
    load = canonical_dtype(cfg.load_dtype)
    if load not in SUPPORTED_ACT_DTYPES:
        return (
            f"load dtype {load}: 검증된 saved-tensor 식은 16-bit(bf16/fp16) 로드 전용입니다 "
            "(architecture-memory.md §5)."
        )
    amp = amp_dtype(cfg)
    on = amp is not None if autocast is None else autocast
    if on and amp is not None and amp != load:
        return (
            f"mixed precision {cfg.mixed_precision}: autocast dtype {amp}가 load dtype {load}와 "
            "달라 Linear마다 입력·weight cast 사본이 저장되는 경로는 검증되지 않았습니다 "
            "(architecture-memory.md §4.3)."
        )
    return None


def mode_class(trainability: Trainability, mode: act.ActMode) -> str:
    if trainability.full:
        return "full"
    return "lora_shared" if mode.adapter_bytes == mode.b else "lora_fp32"


def padding_possible(cfg: ResolvedConfig, shape: SequenceShape) -> bool:
    """Rows of different lengths or pad_to_multiple_of may pad the batch; then SDPA gets a
    materialized mask (§3.1). B = 1 without padding uses `enable_gqa` and no mask. GRPO micro-
    batches carry left-padded prompts and right-padded completions of the generation batch."""
    pad_multiple = cfg.pad_to_multiple_of is not None and cfg.pad_to_multiple_of > 1
    return shape.batch > 1 or pad_multiple or cfg.objective is Objective.GRPO


def padding_variants(cfg: ResolvedConfig, shape: SequenceShape) -> list[bool]:
    """Padding cases to evaluate: exactly the trainer-reported `SequenceShape.has_padding` (masked
    SDPA terms only when True); when it is None, the unmasked and the masked path whenever padding
    is possible (each group then spans low = min .. high = max)."""
    if shape.has_padding is not None:
        return [shape.has_padding]
    return [False, True] if padding_possible(cfg, shape) else [False]


# ---------------------------------------------------------------- kernel paths


@dataclass(frozen=True)
class Paths:
    attention: dict[str, str | None]  # layer type -> "sdpa" | "sdpa_mem_efficient" | "eager"
    linear: act.LinearKernel | None
    notes: dict[str, str] = field(default_factory=dict)  # layer type -> defaulted path


def attention_dropout(structure: ModelStructure) -> float:
    """Config `attention_dropout` (applied functionally in training; not an nn.Dropout, so TRL's
    disable_dropout does not reach it)."""
    value = structure.facts.extra.get("attention_dropout")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def resolve_paths(structure: ModelStructure, cfg: ResolvedConfig) -> Paths:
    configured = {k: v.strip().lower() for k, v in cfg.attention_path_by_layer_type.items()}
    notes: dict[str, str] = {}
    attention: dict[str, str | None] = {}
    for lt in sorted(set(structure.layer_types) - {LINEAR_ATTENTION}):
        raw = configured.get(lt) or configured.get(FULL_ATTENTION)
        if raw in (None, "auto"):
            raw = DEFAULT_ATTENTION
            notes[lt] = "attention 경로 미지정 → transformers 기본 sdpa로 계산"
        if attention_dropout(structure) > 0:
            # dropout changes the saved set (eager mask) and the CUDA SDPA backend choice
            # (flash refuses head_dim > 224 with dropout on sm86/89/120): not verified
            attention[lt] = None
        elif raw in _SDPA_AUTO:
            attention[lt] = "sdpa"
        elif raw in _SDPA_MEM:
            attention[lt] = "sdpa_mem_efficient"
        elif raw == "eager":
            attention[lt] = "eager"
        else:
            attention[lt] = None  # flash_attention_2, math, flex, ...: saved set not verified
    linear: act.LinearKernel | None = None
    if LINEAR_ATTENTION in structure.layer_types:
        kernel = configured.get(LINEAR_ATTENTION)
        if kernel in (None, "auto"):
            kernel = DEFAULT_LINEAR
            notes[LINEAR_ATTENTION] = "kernel 미지정 → 고정 환경의 torch fallback으로 계산"
        linear = _LINEAR_PATHS.get(kernel)
    return Paths(attention, linear, notes)


def attn_path(
    path: str, layer_type: str, structure: ModelStructure, seq: int, padded: bool
) -> act.AttnPath:
    """CUDA SDPA backend per torch 2.14.1 rules on sm80+ (§3.1-3.2): flash/cuDNN when no mask is
    materialized and head_dim <= 256, else mem-efficient; transformers expands K/V with repeat_kv
    unless `attention_mask is None and head_dim <= 256` (enable_gqa)."""
    d = structure.dims
    window = d.sliding_window
    mask = padded or (layer_type == SLIDING_ATTENTION and window is not None and seq >= window)
    if path == "eager":
        return act.AttnPath("eager", True, d.heads != d.kv_heads)
    expand = (mask or d.head_dim > 256) and d.heads != d.kv_heads
    if path == "sdpa_mem_efficient" or mask or d.head_dim > 256:
        return act.AttnPath("mem_efficient", mask, expand)
    return act.AttnPath("flash", False, False)


# ---------------------------------------------------------------- per-layer saved sets


def act_dims(structure: ModelStructure) -> act.Dims:
    d = structure.dims
    return act.Dims(
        hidden=d.hidden,
        intermediate=d.intermediate,
        heads=d.heads,
        kv_heads=d.kv_heads,
        head_dim=d.head_dim,
        lin_key_heads=d.lin_key_heads,
        lin_value_heads=d.lin_value_heads,
        lin_key_dim=d.lin_key_dim,
        lin_value_dim=d.lin_value_dim,
        conv_kernel=d.conv_kernel,
    )


def layer_train(structure: ModelStructure, layer: Layer, tr: Trainability) -> act.LayerTrain:
    linears = {
        kind: act.ModuleTrain(
            kind=kind,
            in_features=m.in_features,
            out_features=m.out_features,
            trainable=tr.module_trainable(m.name),
            rank=tr.lora_rank(m.name),
        )
        for kind, m in layer.linears.items()
    }
    norms = {leaf: tr.module_trainable(t.module) for leaf, t in layer.norms.items()}
    # g = -exp(A_log.float()) * softplus(a.float() + dt_bias): only A_log changes the saved set
    # (a frozen one still leaves -exp(A_log) [Hv] in the mul); dt_bias enters an add, which saves
    # nothing (measured, test_arch_parity).
    a_log = [t for t in layer.tensors if t.name.rpartition(".")[2] == "A_log"]
    params_trainable = any(tr.module_trainable(t.module) for t in a_log)
    widths = None
    if structure.family == "dense" and "q_norm" in layer.norms and "k_norm" in layer.norms:
        widths = (layer.norms["q_norm"].shape[0], layer.norms["k_norm"].shape[0])
    return act.LayerTrain(linears, norms, params_trainable, widths)


def act_mode(
    cfg: ResolvedConfig, *, use_cache: bool = False, autocast: bool | None = None
) -> act.ActMode | None:
    """None when the verified 16-bit formulas do not apply (`dtype_issue`). `autocast` overrides
    `ResolvedConfig.mixed_precision` for forwards that bypass accelerate's wrapper."""
    if dtype_issue(cfg, autocast) is not None:
        return None
    load = canonical_dtype(cfg.load_dtype)
    lora = cfg.lora
    adapter = canonical_dtype(cfg.effective_dtypes.adapter) if lora else load
    return act.ActMode(
        b=int(dtype_bytes(load)),
        autocast=autocast_on(cfg) if autocast is None else autocast,
        adapter_bytes=int(dtype_bytes(adapter)),
        dropout=lora.dropout if lora else 0.0,
        use_cache=use_cache,
    )


@dataclass
class LayerSet:
    """Per-layer saved terms of one padding variant (None = path not computable)."""

    padded: bool
    terms: list[list[act.SavedTerm] | None]

    @property
    def known(self) -> bool:
        return all(t is not None for t in self.terms)

    def layer_bytes(self) -> list[int]:
        return [act.total(t) for t in self.terms if t is not None]


def layer_sets(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    paths: Paths,
    batch: int,
    seq: int,
    mode: act.ActMode | None,
    padded_variants: Iterable[bool],
) -> list[LayerSet]:
    dims = act_dims(structure)
    out: list[LayerSet] = []
    for padded in padded_variants:
        ls = LayerSet(padded, [])
        for layer in structure.layers:
            if mode is None:
                ls.terms.append(None)
                continue
            lt = layer_train(structure, layer, tr)
            if layer.layer_type == LINEAR_ATTENTION:
                if paths.linear is None:
                    ls.terms.append(None)
                    continue
                terms = act.q35_linear_attention_layer(batch, seq, dims, lt, mode, paths.linear)
            else:
                path = paths.attention.get(layer.layer_type)
                if path is None:
                    ls.terms.append(None)
                    continue
                ap = attn_path(path, layer.layer_type, structure, seq, padded)
                if structure.family == "qwen3_5":
                    terms = act.q35_full_attention_layer(batch, seq, dims, lt, mode, ap)
                else:
                    terms = act.dense_layer(batch, seq, dims, lt, mode, ap)
            ls.terms.append(terms)
        out.append(ls)
    return out


def _range(values: Iterable[int | None]) -> tuple[int | None, int | None]:
    vals = list(values)
    if not vals or any(v is None for v in vals):
        return None, None
    known = [v for v in vals if v is not None]
    return min(known), max(known)


def s_max(sets: Sequence[LayerSet]) -> tuple[int | None, int | None]:
    return _range(max(s.layer_bytes()) if s.known and s.layer_bytes() else None for s in sets)


def unknown_reason(
    structure: ModelStructure, cfg: ResolvedConfig, paths: Paths, autocast: bool | None = None
) -> str:
    issue = dtype_issue(cfg, autocast)
    if issue is not None:
        return issue
    bad = [lt for lt, p in paths.attention.items() if p is None]
    if bad and attention_dropout(structure) > 0:
        return (
            f"attention_dropout={attention_dropout(structure):g}: dropout이 있는 attention의 "
            "saved set과 CUDA SDPA backend 선택이 검증되지 않았습니다."
        )
    if bad:
        raw = ", ".join(f"{lt}={cfg.attention_path_by_layer_type.get(lt)}" for lt in bad)
        return f"attention 경로({raw})의 saved set이 검증되지 않았습니다."
    if LINEAR_ATTENTION in structure.layer_types and paths.linear is None:
        kernel = cfg.attention_path_by_layer_type.get(LINEAR_ATTENTION)
        return f"linear-attention kernel({kernel})의 saved set이 검증되지 않았습니다."
    return "계산할 수 없는 layer가 있습니다."


# ---------------------------------------------------------------- model-level kwargs


def rope_bytes(
    structure: ModelStructure, cfg: ResolvedConfig, batch: int, seq: int
) -> tuple[int, int]:
    """cos/sin shared by every attention layer. Qwen3.5 mrope: [B,T,r] (§1.5); dense with
    position_ids=None: [1,T,d] (§8), [B,T,d] when the trainer passes position_ids."""
    b = dtype_bytes(cfg.load_dtype)
    r = structure.dims.rotary_dim
    if structure.family == "qwen3_5":
        v = _ceil(2 * b * batch * seq * r)
        return v, v
    return _ceil(2 * b * seq * r), _ceil(2 * b * batch * seq * r)


def mask_kwargs(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    paths: Paths,
    batch: int,
    seq: int,
    padded_variants: Sequence[bool],
) -> list[tuple[str, int, int, str]]:
    """Shared 4-D masks, one per attention layer type: SDPA bool [B,1,T,T] when materialized,
    eager float [B,1,T,T] in the model dtype always (§3.1)."""
    out: list[tuple[str, int, int, str]] = []
    window = structure.dims.sliding_window
    for lt, path in paths.attention.items():
        if path is None:
            continue
        if path == "eager":
            v = _ceil(dtype_bytes(cfg.load_dtype) * batch * seq * seq)
            out.append((lt, v, v, "eager float mask [B,1,T,T]"))
            continue
        sizes = []
        for padded in padded_variants:
            mask = padded or (lt == SLIDING_ATTENTION and window is not None and seq >= window)
            sizes.append(batch * seq * seq if mask else 0)
        if max(sizes):
            out.append((lt, min(sizes), max(sizes), "SDPA bool mask [B,1,T,T]"))
    return out


# ---------------------------------------------------------------- bitsandbytes transients


def q4_transient(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    tokens: int,
    *,
    forward: bool,
    backward: bool,
) -> tuple[int, int, str] | None:
    """Linear4bit runtime transients of the executed text layers (research §3.6, INFERRED CUDA):
    forward dequant `n·b(compute) + 8·nb` when M = tokens > 1536 (fused kernel M <= 4, heuristic
    in between), PEFT bnb wrapper `result.clone()`, backward dequant `n·b(load) + 4·nb`."""
    qmods = quantized_modules(structure, cfg)
    executed = [
        m
        for m in structure.linear_modules
        if m.name in qmods and m.component in EXECUTED_COMPONENTS and m.kind != "lm_head"
    ]
    if not executed:
        return None
    dq = cfg.quantization.double_quant
    blocksize = cfg.quantization.blocksize or BNB_BLOCKSIZE
    n_max = max(m.in_features * m.out_features for m in executed)
    nb = math.ceil(n_max / blocksize)
    comp_b = dtype_bytes(cfg.quantization.compute_dtype or cfg.effective_dtypes.compute)
    act_b = dtype_bytes(cfg.load_dtype)
    fwd = _ceil(n_max * comp_b) + (8 * nb if dq else 0)
    fwd_low = fwd if tokens > 1536 else 0
    fwd_high = fwd if tokens > 4 else 0
    clone = max(
        (tokens * m.out_features * _ceil(act_b) for m in executed if tr.lora_rank(m.name)),
        default=0,
    )
    low = high = 0
    if forward:
        low, high = max(fwd_low, clone), max(fwd_high, clone)
    if backward:
        bwd = _ceil(n_max * act_b) + (_ceil(n_max * comp_b) if comp_b != act_b else 0)
        bwd += 4 * nb if dq else 0
        low, high = max(low, bwd), max(high, bwd)
    if not high:
        return None
    note = (
        f"Linear4bit 최대 weight {n_max:,}개 원소의 dequant 임시값"
        + (" / LoRA bnb 래퍼의 출력 clone" if clone else "")
        + f" (M={tokens}, CUDA 경로 소스 추론)."
    )
    return low, high, note


# ---------------------------------------------------------------- training step


def train_step(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    shape: SequenceShape,
    tps: StepTimepoints,
    prefix: str,
) -> list[AllocationSpec]:
    batch, seq = shape.batch, shape.seq_len
    n = batch * seq
    d = structure.dims
    load = canonical_dtype(cfg.load_dtype)
    load_b = dtype_bytes(load)
    gc = cfg.gradient_checkpointing and cfg.checkpointing_granularity == "per_decoder_layer"
    use_cache = cfg.use_cache_during_training and not gc
    mode = act_mode(cfg)
    paths = resolve_paths(structure, cfg)
    variants = padding_variants(cfg, shape)
    sets = layer_sets(structure, cfg, tr, paths, batch, seq, mode, variants)
    every = [tps.forward, tps.loss, tps.backward]
    dims = {"B": batch, "T": seq, "H": d.hidden}
    p = f"{prefix}.act"
    out: list[AllocationSpec] = []

    if gc:
        nl = len(structure.layers)
        out.append(
            spec(
                f"{p}.ckpt_boundaries",
                AllocationCategory.SAVED_ACTIVATIONS,
                _ceil(nl * n * d.hidden * load_b),
                _ceil(nl * n * d.hidden * load_b),
                every,
                ref="act-gc",
                shape="layers × [B,T,H]",
                dims={**dims, "layers": nl},
                dtype=load,
                count=nl,
                saved=True,
                note="decoder layer마다 checkpoint 입력 hidden state 1개 (load dtype).",
            )
        )
    lo, hi = rope_bytes(structure, cfg, batch, seq)
    if paths.attention:
        out.append(
            spec(
                f"{p}.rope_cos_sin",
                AllocationCategory.SAVED_ACTIVATIONS,
                lo,
                hi,
                every,
                ref="act-kwargs",
                shape="2×[B,T,r]" if structure.family == "qwen3_5" else "2×[1|B,T,r]",
                dims={**dims, "r": d.rotary_dim},
                dtype=load,
                saved=True,
                note="모든 attention layer가 공유하는 cos/sin (모델당 1번).",
            )
        )
    for lt, mlo, mhi, what in mask_kwargs(structure, cfg, paths, batch, seq, variants):
        live = every if gc else [tps.forward]
        out.append(
            spec(
                f"{p}.attn_mask.{lt}",
                AllocationCategory.SAVED_ACTIVATIONS,
                mlo,
                mhi,
                live,
                ref="act-mask",
                shape="[B,1,T,T]",
                dims=dims,
                note=f"{what}: {lt} layer 공유. GC면 checkpoint kwargs로 backward까지 유지."
                + (" padding 유무에 따라 0일 수 있습니다." if mlo != mhi else ""),
            )
        )
    final_norm = structure.final_norm
    norm_trains = tr.module_trainable(final_norm.module) if final_norm else tr.full
    if mode is not None:
        fn = (
            act.rmsnorm_q35(n, d.hidden, norm_trains)
            if structure.family == "qwen3_5"
            else act.rmsnorm_llama(n, d.hidden, norm_trains, mode.b)
        )
    else:
        fn = None
    out.append(
        spec(
            f"{p}.final_norm",
            AllocationCategory.SAVED_ACTIVATIONS,
            fn,
            fn,
            every,
            ref="act-norm",
            shape="[N,H] fp32 + rstd",
            dims=dims,
            saved=True,
            note=None if fn is not None else unknown_reason(structure, cfg, paths),
        )
    )
    hidden = _ceil(n * d.hidden * load_b)
    out.append(
        spec(
            f"{p}.{FINAL_HIDDEN_SUFFIX}",
            AllocationCategory.SAVED_ACTIVATIONS,
            hidden,
            hidden,
            [tps.forward, tps.loss],
            ref="act-boundary",
            shape="[B,T,H]",
            dims=dims,
            dtype=load,
            alias=final_hidden_alias(prefix),
            note="final norm 출력 = LM head 입력. LM head ledger가 저장분으로 셀 때 같은 alias.",
        )
    )

    known = all(s.known for s in sets)
    reason = None if known else unknown_reason(structure, cfg, paths)
    inferred = inferred_notes(structure, paths, seq, variants, mode)
    if not gc:
        out.extend(
            _saved_layer_allocs(structure, sets, every, p, dims, reason, paths.notes, inferred)
        )
        if use_cache and known:
            # A DynamicCache kept by the training forward pins the last state-update branch of
            # every linear layer until the outputs are dropped (§4.2 cache variant, measured).
            cached = layer_sets(
                structure, cfg, tr, paths, batch, seq, act_mode(cfg, use_cache=True), variants
            )
            extra = [
                sum(c.layer_bytes()) - sum(s.layer_bytes())
                for c, s in zip(cached, sets, strict=True)
            ]
            live = [tps.forward, tps.loss]
            if max(extra):
                out.append(
                    spec(
                        f"{p}.cache_branch",
                        AllocationCategory.SAVED_ACTIVATIONS,
                        min(extra),
                        max(extra),
                        live,
                        ref="act-linear",
                        evidence=Evidence.ANALYTIC,
                        note="use_cache=True: cache가 붙잡은 마지막 state-update branch.",
                    )
                )
            out.extend(_train_cache_states(structure, cfg, batch, live, p))
    smin, smax = s_max(sets)
    cls = mode_class(tr, mode) if mode else "full"
    defaulted = " ".join(
        [f"{lt}: {why}." for lt, why in paths.notes.items()]
        + list(dict.fromkeys(inferred.values()))
    )
    W, R = AllocationCategory.WORKSPACE, AllocationCategory.RECOMPUTE_WORKING_SET
    factors = (
        [
            (
                "layer_transient.forward",
                K_GC_FORWARD,
                tps.forward,
                W,
                "act-transient",
                "checkpoint된 layer 1개의 forward 작업 집합",
                None,
            ),
            (
                "recompute.backward",
                K_GC_BACKWARD,
                tps.backward,
                R,
                "act-gc",
                "가장 큰 layer의 재계산 saved set + backward 임시값",
                "decoder_layer",
            ),
        ]
        if gc
        else [
            (
                "layer_transient.forward",
                K_NOGC_FORWARD,
                tps.forward,
                W,
                "act-transient",
                "마지막 layer forward의 임시값",
                None,
            ),
            (
                "layer_transient.backward",
                K_NOGC_BACKWARD,
                tps.backward,
                W,
                "act-transient",
                "backward 시작 시 layer 1개분 임시값",
                None,
            ),
        ]
    )
    for name, table, tp, category, ref, what, group in factors:
        out.append(
            _factor(
                f"{p}.{name}",
                table[cls],
                smin,
                smax,
                [tp],
                category,
                ref,
                what,
                reason,
                recompute=group,
                extra=defaulted,
            )
        )
    q4f = q4_transient(structure, cfg, tr, n, forward=True, backward=False)
    q4b = q4_transient(structure, cfg, tr, n, forward=gc, backward=True)
    for when, res, tp in (("forward", q4f, tps.forward), ("backward", q4b, tps.backward)):
        if res:
            out.append(
                spec(
                    f"{p}.q4_dequant.{when}",
                    AllocationCategory.WORKSPACE,
                    res[0],
                    res[1],
                    [tp],
                    ref="act-q4",
                    dims={"tokens": n},
                    note=res[2],
                )
            )
    if tr.dora:
        out.append(
            spec(
                f"{p}.dora_extra",
                AllocationCategory.WORKSPACE,
                None,
                None,
                [tps.forward, tps.backward],
                ref="act-lora",
                note="DoRA의 saved tensor와 eye(in)·weight norm 임시값은 검증되지 않았습니다.",
            )
        )
    return out


def _factor(
    name: str,
    k: tuple[float, float],
    smin: int | None,
    smax: int | None,
    live: list[str],
    category: AllocationCategory,
    ref: str,
    what: str,
    reason: str | None,
    *,
    recompute: str | None = None,
    extra: str = "",
) -> AllocationSpec:
    if smin is None or smax is None:
        return spec(name, category, None, None, live, ref=ref, note=reason)
    return spec(
        name,
        category,
        _ceil(k[0] * smin),
        _ceil(k[1] * smax),
        live,
        ref=ref,
        evidence=Evidence.ASSUMPTION,
        shape="k × S_max",
        dims={"S_max": smax},
        recompute=recompute,
        note=(
            f"{what}: k={k[0]:g}–{k[1]:g} × 가장 큰 layer saved set(S_max). k는 CPU 측정 기반 "
            "계수(GPU 보정 전)입니다." + (f" {extra}" if extra else "")
        ),
    )


_FLA_NOTE = (
    "fla·causal-conv1d saved set은 CUDA 전용이라 소스 정독 기반 추정입니다(INFERRED, AM §2.3)."
)
_MEM_EFFICIENT_NOTE = (
    "mask가 있는 CUDA mem-efficient 경로(repeat_kv K/V, 8 정렬 additive mask, 32 정렬 lse, "
    "contiguous 복사 없음)는 torch 소스 기반 추정입니다(INFERRED, AM §3.2-3.4)."
)
_DROPOUT_NOTE = "LoRA dropout mask는 CUDA native_dropout의 bool(1 B/원소)로 가정합니다(INFERRED)."


def inferred_notes(
    structure: ModelStructure,
    paths: Paths,
    seq: int,
    variants: Sequence[bool],
    mode: act.ActMode | None,
) -> dict[tuple[str, str], str]:
    """(layer type, ledger group) -> note for terms read from CUDA sources but not measured."""
    notes: dict[tuple[str, str], str] = {}
    if paths.linear == "fla":
        notes[(LINEAR_ATTENTION, "linear_attention")] = _FLA_NOTE
    for lt, path in paths.attention.items():
        kinds = {attn_path(path, lt, structure, seq, v).kind for v in variants} if path else set()
        if "mem_efficient" in kinds:
            notes[(lt, "attention")] = notes[(lt, "mask")] = _MEM_EFFICIENT_NOTE
    if mode is not None and mode.dropout > 0:
        for lt in set(structure.layer_types):
            notes[(lt, "lora")] = _DROPOUT_NOTE
    return notes


def _saved_layer_allocs(
    structure: ModelStructure,
    sets: Sequence[LayerSet],
    live: list[str],
    p: str,
    dims: dict[str, int],
    reason: str | None,
    path_notes: dict[str, str],
    inferred: dict[tuple[str, str], str],
) -> list[AllocationSpec]:
    out: list[AllocationSpec] = []
    for lt in dict.fromkeys(structure.layer_types):
        idx = [i for i, t in enumerate(structure.layer_types) if t == lt]
        if any(s.terms[i] is None for s in sets for i in idx):
            out.append(
                spec(
                    f"{p}.{lt}",
                    AllocationCategory.SAVED_ACTIVATIONS,
                    None,
                    None,
                    live,
                    ref="act-layers",
                    count=len(idx),
                    note=reason,
                )
            )
            continue
        per_set: list[dict[str, int]] = []
        shapes: dict[str, dict[str, None]] = {}  # group -> ordered "label shape" of its terms
        for s in sets:
            sums: dict[str, int] = {}
            for i in idx:
                for term in s.terms[i] or []:
                    sums[term.group] = sums.get(term.group, 0) + term.nbytes
                    shapes.setdefault(term.group, {})[f"{term.label} {term.shape}"] = None
            per_set.append(sums)
        # A group missing from one padding variant (e.g. "mask") is 0 bytes there.
        for g in dict.fromkeys(name for sums in per_set for name in sums):
            values = [sums.get(g, 0) for sums in per_set]
            lo, hi = min(values), max(values)
            if not hi:
                continue
            note = f"{lt} layer {len(idx)}개의 saved tensor ({g})."
            if lo != hi:
                note += " padding mask 유무에 따른 범위입니다."
            mixer = "linear_attention" if lt == LINEAR_ATTENTION else "attention"
            if g == mixer and lt in path_notes:
                note += f" ({path_notes[lt]})"
            if (lt, g) in inferred:
                note += " " + inferred[(lt, g)]
            out.append(
                spec(
                    f"{p}.{lt}.{g}",
                    AllocationCategory.SAVED_ACTIVATIONS,
                    lo,
                    hi,
                    live,
                    ref=_GROUP_REFS[g],
                    evidence=Evidence.ANALYTIC,
                    shape="layer당 " + " + ".join(shapes[g]),
                    dims={**dims, "layers": len(idx)},
                    count=len(idx),
                    saved=True,
                    note=note,
                )
            )
    return out


def _train_cache_states(
    structure: ModelStructure, cfg: ResolvedConfig, batch: int, live: list[str], p: str
) -> list[AllocationSpec]:
    n_lin = structure.count(LINEAR_ATTENTION)
    if not n_lin:
        return []
    d = structure.dims
    conv_b = dtype_bytes(forward_dtype(cfg, autocast_on(cfg)))  # accelerate-wrapped forward
    v = (
        n_lin
        * batch
        * (
            _ceil(d.conv_dim * d.conv_kernel * conv_b)
            + 4 * d.lin_value_heads * d.lin_key_dim * d.lin_value_dim
        )
    )
    return [
        spec(
            f"{p}.cache_states",
            AllocationCategory.RECURRENT_STATE,
            v,
            v,
            live,
            ref="gen-linear",
            note="use_cache=True 학습 forward의 conv/recurrent state.",
        )
    ]


# ---------------------------------------------------------------- no-grad forward


def no_grad_forward(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    shape: SequenceShape,
    live_at: list[str],
    prefix: str,
    *,
    autocast: bool | None = None,
) -> list[AllocationSpec]:
    """Inference-mode forward: one layer's working set at a time plus the hidden state (§7)."""
    batch, seq = shape.batch, shape.seq_len
    d = structure.dims
    load = canonical_dtype(cfg.load_dtype)
    mode = act_mode(cfg, autocast=autocast)
    paths = resolve_paths(structure, cfg)
    variants = padding_variants(cfg, shape)
    sets = layer_sets(structure, cfg, tr, paths, batch, seq, mode, variants)
    p = f"{prefix}.nograd"
    dims = {"B": batch, "T": seq, "H": d.hidden}
    hidden = _ceil(batch * seq * d.hidden * dtype_bytes(load))
    out = [
        spec(
            f"{p}.hidden",
            AllocationCategory.WORKSPACE,
            hidden,
            hidden,
            live_at,
            ref="act-nograd",
            shape="[B,T,H]",
            dims=dims,
            dtype=load,
            note="layer 사이에 유지되는 hidden state 1개.",
        )
    ]
    if paths.attention:
        lo, hi = rope_bytes(structure, cfg, batch, seq)
        out.append(
            spec(
                f"{p}.rope_cos_sin",
                AllocationCategory.WORKSPACE,
                lo,
                hi,
                live_at,
                ref="act-kwargs",
                dims=dims,
                dtype=load,
            )
        )
    for lt, mlo, mhi, what in mask_kwargs(structure, cfg, paths, batch, seq, variants):
        out.append(
            spec(
                f"{p}.attn_mask.{lt}",
                AllocationCategory.WORKSPACE,
                mlo,
                mhi,
                live_at,
                ref="act-mask",
                dims=dims,
                note=what,
            )
        )
    smin, smax = s_max(sets)
    known = all(s.known for s in sets)
    reason = None if known else unknown_reason(structure, cfg, paths, autocast)
    cls = mode_class(tr, mode) if mode else "full"
    out.append(
        _factor(
            f"{p}.layer_working_set",
            K_GC_FORWARD[cls],
            smin,
            smax,
            live_at,
            AllocationCategory.WORKSPACE,
            "act-nograd",
            "no_grad forward의 layer 1개 작업 집합",
            reason,
            extra=" ".join(
                dict.fromkeys(inferred_notes(structure, paths, seq, variants, mode).values())
            ),
        )
    )
    q4 = q4_transient(structure, cfg, tr, batch * seq, forward=True, backward=False)
    if q4:
        out.append(
            spec(
                f"{p}.q4_dequant",
                AllocationCategory.WORKSPACE,
                q4[0],
                q4[1],
                live_at,
                ref="act-q4",
                note=q4[2],
            )
        )
    if tr.dora:
        out.append(
            spec(
                f"{p}.dora_extra",
                AllocationCategory.WORKSPACE,
                None,
                None,
                live_at,
                ref="act-lora",
                note="DoRA forward 임시값은 검증되지 않았습니다.",
            )
        )
    return out


# ---------------------------------------------------------------- generation


def _dtype_note(resolved: str, used: str, field_name: str) -> str:
    """Flag a resolver dtype that disagrees with the architecture rule (the rule wins)."""
    try:
        same = canonical_dtype(resolved) == used
    except ValueError:
        same = False
    if same:
        return ""
    return f" (해석된 {field_name} dtype {resolved} 대신 구조 규칙의 {used}를 사용)"


def forward_dtype(cfg: ResolvedConfig, autocast: bool) -> str:
    """Output dtype of the Linear modules in a forward: the autocast dtype under mixed precision,
    else the load dtype."""
    amp = amp_dtype(cfg)
    return amp if autocast and amp is not None else canonical_dtype(cfg.load_dtype)


def rollout_autocast(cfg: ResolvedConfig) -> bool:
    """generate() of a non-PEFT policy runs the accelerate-wrapped forward (autocast); PEFT does not
    (PeftModel.generate bypasses the wrapper) — trl-grpo R3."""
    return cfg.strategy is Strategy.FULL and autocast_on(cfg)


def kv_cache_dtype(cfg: ResolvedConfig) -> tuple[str, str]:
    """(dtype, note) of the rollout K/V cache. K is the RoPE product of the k_proj output (autocast
    dtype under autocast) with cos/sin in the load dtype, so type promotion brings it back to the
    load dtype for bf16/bf16 and fp32-load runs, and V follows K through the cache's lazy init
    (trl-grpo R3, verified). A 16-bit autocast dtype other than a 16-bit load dtype promotes both
    to float32 (torch type promotion, not measured)."""
    load = canonical_dtype(cfg.load_dtype)
    amp = amp_dtype(cfg)
    if not rollout_autocast(cfg) or amp in (None, load) or load == "float32":
        return load, ""
    return "float32", (
        f" autocast {amp} K × load dtype {load} cos/sin의 type promotion으로 K/V가 float32가 "
        "됩니다(INFERRED, 미측정)."
    )


def conv_state_dtype(cfg: ResolvedConfig) -> tuple[str, str]:
    """(dtype, note) of the rollout conv state: the resolved `effective_dtypes.conv_state` when set,
    else the forward dtype of generate() — autocast only for a non-PEFT policy (trl-grpo R3)."""
    rule = forward_dtype(cfg, rollout_autocast(cfg))
    resolved = cfg.effective_dtypes.conv_state
    if not resolved:
        return rule, ""
    try:
        dtype = canonical_dtype(resolved)
    except ValueError:
        return rule, f" (해석된 conv_state dtype {resolved}를 알 수 없어 구조 규칙의 {rule}를 사용)"
    if dtype != rule:
        return dtype, f" (해석된 conv_state dtype {dtype}를 사용, 구조 규칙으로는 {rule})"
    return dtype, ""


def generation(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    num_sequences: int,
    prompt_len: int,
    new_tokens: int,
    tps: GenerationTimepoints,
    prefix: str,
) -> list[AllocationSpec]:
    """DynamicCache of `num_sequences` live sequences (research §7, E; trl-grpo R3)."""
    c, plen = num_sequences, prompt_len
    d = structure.dims
    kv_dtype, kv_inferred = kv_cache_dtype(cfg)
    kv_b = dtype_bytes(kv_dtype)  # the load dtype through RoPE type promotion (R3)
    p = f"{prefix}.gen"
    l_dec = plen + max(new_tokens, 1) - 1  # last token is never fed back: L = P + new - 1
    out: list[AllocationSpec] = []
    per_pos = _ceil(2 * d.kv_heads * d.head_dim * kv_b)  # K+V bytes per layer per position
    dec_positions: dict[str, int] = {}  # attention layer type -> cached positions at decode
    kv_note = kv_inferred + _dtype_note(cfg.effective_dtypes.kv_cache, kv_dtype, "kv_cache")
    window = d.sliding_window
    for lt in (FULL_ATTENTION, SLIDING_ATTENTION):
        n_layers = structure.count(lt)
        if not n_layers:
            continue
        if lt == SLIDING_ATTENTION and window is not None:
            # DynamicSlidingWindowLayer keeps a view of the last W-1 tokens of each cat result.
            pre_pos = plen
            dec_pos = min(l_dec, window) if new_tokens >= 2 else plen
        else:
            pre_pos, dec_pos = plen, l_dec
        for when, pos, tp in (("prefill", pre_pos, tps.prefill), ("decode", dec_pos, tps.decode)):
            v = n_layers * c * pos * per_pos
            out.append(
                spec(
                    f"{p}.kv_cache.{lt}.{when}",
                    AllocationCategory.GENERATION_CACHE,
                    v,
                    v,
                    [tp],
                    ref="gen-kv",
                    shape="2 × layers × [C,nkv,L,d]",
                    dims={"layers": n_layers, "C": c, "L": pos, "nkv": d.kv_heads, "d": d.head_dim},
                    dtype=kv_dtype,
                    count=n_layers,
                    note=f"{lt} layer {n_layers}개만 K/V를 가집니다 (L={pos}).{kv_note}",
                )
            )
        dec_positions[lt] = dec_pos
    n_lin = structure.count(LINEAR_ATTENTION)
    if n_lin:
        conv_dtype, conv_note = conv_state_dtype(cfg)
        conv = n_lin * c * _ceil(d.conv_dim * d.conv_kernel * dtype_bytes(conv_dtype))
        rec = n_lin * c * 4 * d.lin_value_heads * d.lin_key_dim * d.lin_value_dim
        both = [tps.prefill, tps.decode]
        out.append(
            spec(
                f"{p}.linear_state.conv",
                AllocationCategory.RECURRENT_STATE,
                conv,
                conv,
                both,
                ref="gen-linear",
                shape="layers × [C,conv_dim,K]",
                dims={"layers": n_lin, "C": c, "conv_dim": d.conv_dim, "K": d.conv_kernel},
                dtype=conv_dtype,
                count=n_lin,
                note="linear_attention layer의 conv state (forward dtype, L과 무관)." + conv_note,
            )
        )
        out.append(
            spec(
                f"{p}.linear_state.recurrent",
                AllocationCategory.RECURRENT_STATE,
                rec,
                rec,
                both,
                ref="gen-linear",
                shape="layers × [C,Hv,dk,dv]",
                dims={"layers": n_lin, "C": c, "Hv": d.lin_value_heads},
                dtype="float32",
                count=n_lin,
                note="recurrent state는 config(mamba_ssm_dtype)와 무관하게 fp32입니다."
                + _dtype_note(cfg.effective_dtypes.recurrent_state, "float32", "recurrent_state"),
            )
        )
    if new_tokens >= 2:  # max_new_tokens = 1 ends after the prefill: no decode forward
        padded = padding_possible(cfg, SequenceShape(batch=c, seq_len=plen))
        out.append(_decode_step(structure, cfg, tr, c, dec_positions, kv_b, padded, tps.decode, p))
    autocast = rollout_autocast(cfg)
    prefill = no_grad_forward(
        structure,
        cfg,
        tr,
        SequenceShape(batch=c, seq_len=plen),
        [tps.prefill],
        f"{p}.prefill",
        autocast=autocast,
    )
    out.extend(prefill)
    return out


# Token-sized fp32 temporaries of one recurrent decode step (q/k/v/beta/g upcasts, l2norm, output,
# delta), bounded per sequence by 8·Hv·(dk+dv) fp32 elements; measured ≤ half of it on CPU.
_RECURRENT_TOKEN_TEMPS = 8


def _decode_step(
    structure: ModelStructure,
    cfg: ResolvedConfig,
    tr: Trainability,
    c: int,
    dec_positions: dict[str, int],
    kv_b: float,
    padded: bool,
    decode_tp: str,
    p: str,
) -> AllocationSpec:
    """Working set of one decode step: the layers run one after another, so the largest of their
    transients counts (not the sum).

    - attention layer: DynamicLayer `torch.cat` growth (old + new K, or K and V), and with a padding
      mask the GQA `repeat_kv` copies of the whole cached K/V (`use_gqa_in_sdpa` needs mask None,
      research §3.1 VERIFIED); eager always repeats K/V;
    - linear layer, torch recurrent path: the cached state is reused but `state * decay`,
      `k ⊗ delta` and their sum are new [C,Hv,dk,dv] fp32 tensors (3 alive, CPU measured 3.06·S);
      fla `fused_recurrent` allocates the fp32 final state (INFERRED, §2.5);
    - bitsandbytes Linear4bit at M = C tokens (`q4_transient`).
    """
    d = structure.dims
    paths = resolve_paths(structure, cfg)
    parts: dict[str, tuple[int, int] | None] = {}
    max_pos = max(dec_positions.values(), default=0)
    if max_pos:
        k_one = _ceil(c * d.kv_heads * max_pos * d.head_dim * kv_b)
        parts["cat"] = (k_one, 2 * k_one)
        if d.heads != d.kv_heads:
            expand = _ceil(2 * c * d.heads * max_pos * d.head_dim * kv_b)
            kinds = {paths.attention.get(lt) for lt in dec_positions}
            if None in kinds:
                parts["kv_repeat"] = None
            elif "eager" in kinds:
                scores = _ceil((act.F + kv_b) * c * d.heads * max_pos)
                parts["kv_repeat"] = (expand, expand + scores)
            elif padded:
                parts["kv_repeat"] = (0, expand)
    if structure.count(LINEAR_ATTENTION):
        state = c * 4 * d.lin_value_heads * d.lin_key_dim * d.lin_value_dim
        tokens = (
            _RECURRENT_TOKEN_TEMPS * c * 4 * d.lin_value_heads * (d.lin_key_dim + d.lin_value_dim)
        )
        if paths.linear == "torch":
            parts["recurrent_step"] = (3 * state, 3 * state + tokens)
        elif paths.linear == "fla":
            parts["recurrent_step"] = (state, 3 * state + tokens)
        else:
            parts["recurrent_step"] = None
    q4 = q4_transient(structure, cfg, tr, c, forward=True, backward=False)
    if q4:
        parts["q4"] = (q4[0], q4[1])
    name = f"{p}.decode.step_transient"
    unknown = [k for k, v in parts.items() if v is None]
    if unknown:
        what = {
            "kv_repeat": "attention 경로",
            "recurrent_step": "linear-attention kernel",
        }
        return spec(
            name,
            AllocationCategory.WORKSPACE,
            None,
            None,
            [decode_tp],
            ref="gen-step",
            note="decode step에서 "
            + ", ".join(what[k] for k in unknown)
            + "의 임시값이 검증되지 않았습니다 (경로: "
            + ", ".join(f"{k}={v}" for k, v in cfg.attention_path_by_layer_type.items())
            + ").",
        )
    known = {k: v for k, v in parts.items() if v is not None}
    low = max((v[0] for v in known.values()), default=0)
    high = max((v[1] for v in known.values()), default=0)
    labels = {
        "cat": "K/V torch.cat 성장",
        "kv_repeat": "GQA repeat_kv 사본(padding mask가 있을 때)",
        "recurrent_step": "recurrent state fp32 임시값",
        "q4": "Linear4bit dequant/clone",
    }
    detail = ", ".join(f"{labels[k]} {v[0]:,}–{v[1]:,} B" for k, v in known.items())
    inferred = " fla 경로는 소스 정독 기반 추정(INFERRED)." if paths.linear == "fla" else ""
    return spec(
        name,
        AllocationCategory.WORKSPACE,
        low,
        high,
        [decode_tp],
        ref="gen-step",
        shape="max(layer별 decode 임시값)",
        dims={"C": c, "L": max_pos, **{k: v[1] for k, v in known.items()}},
        note="decode 한 step에서 layer가 차례로 실행되므로 가장 큰 임시값만 동시에 존재합니다: "
        f"{detail}. SDPA kernel workspace는 제외(CUDA 미측정).{inferred}",
    )
