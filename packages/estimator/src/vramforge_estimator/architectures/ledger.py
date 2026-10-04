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
from .base import GenerationTimepoints, StepTimepoints
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


def autocast_on(cfg: ResolvedConfig) -> bool:
    """TRL bf16/fp16 mixed precision = accelerate native AMP autocast (research §2.4). The resolved
    config has no explicit flag; a 16-bit compute dtype is taken as mixed precision."""
    return canonical_dtype(cfg.effective_dtypes.compute) in SUPPORTED_ACT_DTYPES


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


# ---------------------------------------------------------------- kernel paths


@dataclass(frozen=True)
class Paths:
    attention: dict[str, str | None]  # layer type -> "sdpa" | "sdpa_mem_efficient" | "eager"
    linear: act.LinearKernel | None
    notes: dict[str, str] = field(default_factory=dict)  # layer type -> defaulted path


def resolve_paths(structure: ModelStructure, cfg: ResolvedConfig) -> Paths:
    configured = {k: v.strip().lower() for k, v in cfg.attention_path_by_layer_type.items()}
    notes: dict[str, str] = {}
    attention: dict[str, str | None] = {}
    for lt in sorted(set(structure.layer_types) - {LINEAR_ATTENTION}):
        raw = configured.get(lt) or configured.get(FULL_ATTENTION)
        if raw in (None, "auto"):
            raw = DEFAULT_ATTENTION
            notes[lt] = "attention 경로 미지정 → transformers 기본 sdpa로 계산"
        if raw in _SDPA_AUTO:
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
    params = [t for t in layer.tensors if t.name.endswith(("A_log", "dt_bias"))]
    params_trainable = any(tr.module_trainable(t.module) for t in params)
    widths = None
    if structure.family == "dense" and "q_norm" in layer.norms and "k_norm" in layer.norms:
        widths = (layer.norms["q_norm"].shape[0], layer.norms["k_norm"].shape[0])
    return act.LayerTrain(linears, norms, params_trainable, widths)


def act_mode(
    cfg: ResolvedConfig, *, use_cache: bool = False, autocast: bool | None = None
) -> act.ActMode | None:
    """None when the load dtype is not 16-bit: the verified formulas are bf16-only (§5)."""
    load = canonical_dtype(cfg.load_dtype)
    if load not in SUPPORTED_ACT_DTYPES:
        return None
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


def unknown_reason(structure: ModelStructure, cfg: ResolvedConfig, paths: Paths) -> str:
    load = canonical_dtype(cfg.load_dtype)
    if load not in SUPPORTED_ACT_DTYPES:
        return (
            f"load dtype {load}: 검증된 saved-tensor 식은 16-bit(bf16/fp16) 로드 전용입니다 "
            "(architecture-memory.md §5)."
        )
    bad = [lt for lt, p in paths.attention.items() if p is None]
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
    variants = [False, True] if padding_possible(cfg, shape) else [False]
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
            f"{p}.final_hidden",
            AllocationCategory.SAVED_ACTIVATIONS,
            hidden,
            hidden,
            [tps.forward, tps.loss],
            ref="act-boundary",
            shape="[B,T,H]",
            dims=dims,
            dtype=load,
            alias=f"{prefix}.final_hidden",
            note="final norm 출력 = LM head 입력. LM head ledger가 저장분으로 셀 때 같은 alias.",
        )
    )

    known = all(s.known for s in sets)
    reason = None if known else unknown_reason(structure, cfg, paths)
    if not gc:
        out.extend(_saved_layer_allocs(structure, sets, every, p, dims, reason, paths.notes))
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
    defaulted = "; ".join(f"{lt}: {why}" for lt, why in paths.notes.items())
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


def _saved_layer_allocs(
    structure: ModelStructure,
    sets: Sequence[LayerSet],
    live: list[str],
    p: str,
    dims: dict[str, int],
    reason: str | None,
    path_notes: dict[str, str],
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
        for s in sets:
            sums: dict[str, int] = {}
            for i in idx:
                for term in s.terms[i] or []:
                    sums[term.group] = sums.get(term.group, 0) + term.nbytes
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
            out.append(
                spec(
                    f"{p}.{lt}.{g}",
                    AllocationCategory.SAVED_ACTIVATIONS,
                    lo,
                    hi,
                    live,
                    ref=_GROUP_REFS[g],
                    evidence=Evidence.ANALYTIC,
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
    conv_b = dtype_bytes(conv_state_dtype(cfg))
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
    variants = [False, True] if padding_possible(cfg, shape) else [False]
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
    reason = None if all(s.known for s in sets) else unknown_reason(structure, cfg, paths)
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


def conv_state_dtype(cfg: ResolvedConfig) -> str:
    """Conv state follows the forward dtype: autocast (bf16) only for a non-PEFT policy, the load
    dtype under PEFT (PeftModel.generate bypasses the autocast wrapper) — trl-grpo R3."""
    if cfg.strategy is Strategy.FULL and autocast_on(cfg):
        return canonical_dtype(cfg.effective_dtypes.compute)
    return canonical_dtype(cfg.load_dtype)


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
    load = canonical_dtype(cfg.load_dtype)
    kv_b = dtype_bytes(load)  # K/V follow the load dtype (RoPE promotion, R3)
    p = f"{prefix}.gen"
    l_dec = plen + max(new_tokens, 1) - 1  # last token is never fed back: L = P + new - 1
    out: list[AllocationSpec] = []
    per_pos = _ceil(2 * d.kv_heads * d.head_dim * kv_b)  # K+V bytes per layer per position
    cat_tensor = 0
    kv_note = _dtype_note(cfg.effective_dtypes.kv_cache, load, "kv_cache")
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
                    dtype=load,
                    count=n_layers,
                    note=f"{lt} layer {n_layers}개만 K/V를 가집니다 (L={pos}).{kv_note}",
                )
            )
        cat_tensor = max(cat_tensor, c * dec_pos * per_pos // 2)
    n_lin = structure.count(LINEAR_ATTENTION)
    if n_lin:
        conv_dtype = conv_state_dtype(cfg)
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
                note="linear_attention layer의 conv state (forward dtype, L과 무관).",
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
    if cat_tensor and new_tokens >= 2:
        out.append(
            spec(
                f"{p}.kv_cat_transient",
                AllocationCategory.WORKSPACE,
                cat_tensor,
                2 * cat_tensor,
                [tps.decode],
                ref="gen-kv",
                note="DynamicLayer torch.cat 성장: layer 1개의 새 K(또는 K+V)가 이전 "
                "버전과 잠시 공존합니다.",
            )
        )
    autocast = cfg.strategy is Strategy.FULL and autocast_on(cfg)
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
    q4 = q4_transient(structure, cfg, tr, c, forward=True, backward=False)
    if q4:
        out.append(
            spec(
                f"{p}.decode.q4_dequant",
                AllocationCategory.WORKSPACE,
                q4[0],
                q4[1],
                [tps.decode],
                ref="act-q4",
                note=q4[2],
            )
        )
    return out
