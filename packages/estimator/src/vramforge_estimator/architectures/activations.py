"""Saved-activation formulas of one decoder layer (plan.md §9.5 shape ledger).

Closed forms from docs/research/architecture-memory.md, ported term by term:
- §4.2 Qwen3.5 layers (torch fallback, torch + bf16 autocast, fla INFERRED) with the verification
  corrections (nc = 1 without cache drops key-decay exp and chunk_decay), byte-exact on CPU at tiny
  and real dims;
- §5 RMSNorm / gated MLP saved sets, §8 dense (Llama/Qwen3) layer, §3 SDPA backends and eager;
- §4.3 LoRA rules (bf16 adapters share the input, fp32 adapters save per-module copies).
All formulas assume 16-bit activations (`b` = 2: bf16/fp16 load). The golden convention excludes
the 0-dim wrapped scalars autograd keeps for `tensor * python_float` (8 B each, §F).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

F = 4  # fp32 bytes
CHUNK = 64  # Gated DeltaNet chunk size of torch_chunk_gated_delta_rule / fla

Group = Literal["norms", "mlp", "attention", "linear_attention", "lora", "mask"]
AttnKind = Literal["flash", "mem_efficient", "eager"]
LinearKernel = Literal["torch", "fla"]


@dataclass(frozen=True)
class SavedTerm:
    group: Group
    label: str
    nbytes: int
    shape: str


@dataclass(frozen=True)
class ModuleTrain:
    """How one Linear module trains."""

    kind: str
    in_features: int
    out_features: int
    trainable: bool = False  # its own weight trains (full FT / modules_to_save)
    rank: int | None = None  # LoRA rank when targeted


@dataclass(frozen=True)
class LayerTrain:
    linears: Mapping[str, ModuleTrain]
    norm_trainable: Mapping[str, bool] = field(default_factory=dict)  # norm leaf -> trains
    params_trainable: bool = False  # Gated DeltaNet A_log / dt_bias train
    qk_norm_widths: tuple[int, int] | None = None  # dense q_norm / k_norm weight widths


@dataclass(frozen=True)
class ActMode:
    b: int  # activation (compute) bytes, 2 for bf16/fp16
    autocast: bool  # accelerate native AMP
    adapter_bytes: int = 4  # LoRA parameter bytes
    dropout: float = 0.0  # LoRA dropout
    use_cache: bool = False  # DynamicCache kept during the training forward

    @property
    def lora_shares_input(self) -> bool:
        """bf16 adapter reads the bf16 input itself (no cast, no dropout copy)."""
        return self.adapter_bytes == self.b and self.dropout == 0.0


@dataclass(frozen=True)
class AttnPath:
    kind: AttnKind
    mask: bool  # a 4-D mask is materialized (padding, sliding window beyond the window)
    expand_kv: bool  # repeat_kv copies K/V to all query heads


@dataclass(frozen=True)
class Dims:
    """The subset of `structure.Dims` the formulas read (kept local to stay import-light)."""

    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    lin_key_heads: int = 0
    lin_value_heads: int = 0
    lin_key_dim: int = 0
    lin_value_dim: int = 0
    conv_kernel: int = 0

    @property
    def conv_dim(self) -> int:
        return 2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim

    @property
    def value_width(self) -> int:
        return self.lin_value_heads * self.lin_value_dim


class _Acc:
    def __init__(self) -> None:
        self.terms: list[SavedTerm] = []

    def add(self, group: Group, label: str, nbytes: int, shape: str) -> None:
        if nbytes:
            self.terms.append(SavedTerm(group, label, int(nbytes), shape))


def ceil_to(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def total(terms: Sequence[SavedTerm]) -> int:
    return sum(t.nbytes for t in terms)


# ---------------------------------------------------------------- shared parts (§5)


def rmsnorm_q35(rows: int, width: int, trainable: bool) -> int:
    """Qwen3_5RMSNorm: x.float() copy, rsqrt [rows,1], (1+w).float() [w]; normalized fp32 only
    when the weight trains (§4.2 common parts)."""
    return F * rows * width + F * rows + F * width + (F * rows * width if trainable else 0)


def rmsnorm_llama(rows: int, width: int, trainable: bool, b: int) -> int:
    """Llama/Qwen3 RMSNorm: x.float() copy, rsqrt; the bf16 normalized input of `weight * x`
    only when the weight trains."""
    return F * rows * width + F * rows + (b * rows * width if trainable else 0)


def linear_inputs(
    acc: _Acc,
    group: Group,
    label: str,
    modules: Sequence[ModuleTrain],
    n: int,
    mode: ActMode,
    *,
    saved_elsewhere: bool = False,
) -> None:
    """Saved inputs of Linear modules that read the same tensor (§4.3, implications B.1).

    A frozen Linear saves nothing. A trained weight or a bf16 adapter saves the shared input once
    (storage alias); an fp32 adapter breaks the alias with a per-module copy (`b·N·in` cast under
    autocast plus the cast weights, `f·N·in` without autocast). `lora_B` always saves its
    `[N, r]` input. Dropout > 0 adds a per-module mask (CUDA bool, INFERRED) and output.
    """
    if not modules:
        return
    b = mode.b
    in_f = modules[0].in_features
    shared = any(m.trainable for m in modules) or any(
        m.rank is not None and mode.lora_shares_input for m in modules
    )
    if shared and not saved_elsewhere:
        acc.add(group, f"{label}.input", b * n * in_f, f"[N,{in_f}]")
    for m in modules:
        r = m.rank
        if r is None:
            continue
        name = f"{m.kind}"
        if mode.adapter_bytes == b:
            acc.add("lora", f"{name}.lora_B_input", b * n * r, f"[N,{r}]")
            if mode.dropout > 0:
                acc.add("lora", f"{name}.dropout_output", b * n * m.in_features, "[N,in]")
                acc.add("lora", f"{name}.dropout_mask", n * m.in_features, "[N,in] bool")
        elif mode.autocast:
            acc.add("lora", f"{name}.input_cast", b * n * m.in_features, "[N,in]")
            acc.add("lora", f"{name}.lora_B_input", b * n * r, f"[N,{r}]")
            acc.add(
                "lora",
                f"{name}.weight_cast",
                b * (m.in_features * r + r * m.out_features),
                "[r,in]+[out,r]",
            )
            if mode.dropout > 0:
                acc.add("lora", f"{name}.dropout_mask", n * m.in_features, "[N,in] bool")
        else:
            acc.add("lora", f"{name}.input_fp32", F * n * m.in_features, "[N,in] fp32")
            acc.add("lora", f"{name}.lora_B_input", F * n * r, f"[N,{r}] fp32")
            if mode.dropout > 0:
                acc.add("lora", f"{name}.dropout_mask", n * m.in_features, "[N,in] bool")


def _mlp(acc: _Acc, lt: LayerTrain, n: int, d: Dims, mode: ActMode) -> None:
    """down(silu(gate(x)) * up(x)): SiLU input/output and up output always; the gate/up input
    and the product (down input) only for trained/adapted projections (§1.4, §5)."""
    b, i = mode.b, d.intermediate
    linear_inputs(acc, "mlp", "gate_up", [lt.linears["gate_proj"], lt.linears["up_proj"]], n, mode)
    acc.add("mlp", "silu.input", b * n * i, "[N,I]")
    acc.add("mlp", "silu.output", b * n * i, "[N,I]")
    acc.add("mlp", "up.output", b * n * i, "[N,I]")
    linear_inputs(acc, "mlp", "down", [lt.linears["down_proj"]], n, mode)


def _norms(acc: _Acc, lt: LayerTrain, n: int, h: int, mode: ActMode, *, q35: bool) -> None:
    for leaf in ("input_layernorm", "post_attention_layernorm"):
        trains = lt.norm_trainable.get(leaf, False)
        nbytes = rmsnorm_q35(n, h, trains) if q35 else rmsnorm_llama(n, h, trains, mode.b)
        acc.add("norms", leaf, nbytes, "[N,H] fp32 (+rstd)")


def _sdpa_or_eager(
    acc: _Acc,
    batch: int,
    seq: int,
    d: Dims,
    mode: ActMode,
    path: AttnPath,
) -> None:
    """q, k, v as saved by the attention op plus its own outputs (§3.3)."""
    b, n = mode.b, batch * seq
    nq, hd = d.heads, d.head_dim
    kv = nq if path.expand_kv else d.kv_heads
    acc.add("attention", "query", b * n * nq * hd, "[B,nq,T,d]")
    acc.add("attention", "key_value", 2 * b * n * kv * hd, f"2×[B,{kv},T,d]")
    if path.kind == "eager":
        # softmax output fp32 + bf16 probs: (f+b)·B·nq·T² (fit coefficient 6·nq, §4.2)
        acc.add("attention", "softmax_fp32", F * batch * nq * seq * seq, "[B,nq,T,T] fp32")
        acc.add("attention", "probs", b * batch * nq * seq * seq, "[B,nq,T,T]")
        return
    # flash/cuDNN lse [B,nq,T]; mem-efficient pads it to 32 (§3.3)
    lse_len = ceil_to(seq, 32) if path.kind == "mem_efficient" else seq
    acc.add("attention", "lse", F * batch * nq * lse_len, "[B,nq,T] fp32")
    acc.add("attention", "output", b * n * nq * hd, "[B,T,nq,d]")
    if path.mask:
        # bool mask -> query-dtype additive mask per layer, padded to 8 for mem-efficient
        # (torch v2.14.1 attention.cpp:579-639)
        acc.add("mask", "attn_bias", b * batch * seq * ceil_to(seq, 8), "[B,1,T,ceil8(T)]")


# ---------------------------------------------------------------- Qwen3.5 layers (§1.2, §4.2)


def q35_full_attention_layer(
    batch: int, seq: int, d: Dims, lt: LayerTrain, mode: ActMode, path: AttnPath
) -> list[SavedTerm]:
    """S = 2·RMSNorm + MLP + S_fullmix (cos/sin excluded: once per model)."""
    acc, b, n = _Acc(), mode.b, batch * seq
    nq, hd = d.heads, d.head_dim
    _norms(acc, lt, n, d.hidden, mode, q35=True)
    _mlp(acc, lt, n, d, mode)
    qkv = [lt.linears["q_proj"], lt.linears["k_proj"], lt.linears["v_proj"]]
    linear_inputs(acc, "attention", "qkv", qkv, n, mode)
    acc.add(
        "attention",
        "q_norm",
        rmsnorm_q35(n * nq, hd, lt.norm_trainable.get("q_norm", False)),
        "[N·nq,d]",
    )
    acc.add(
        "attention",
        "k_norm",
        rmsnorm_q35(n * d.kv_heads, hd, lt.norm_trainable.get("k_norm", False)),
        "[N·nkv,d]",
    )
    _sdpa_or_eager(acc, batch, seq, d, mode, path)
    if path.kind == "eager":
        acc.add("attention", "output", b * n * nq * hd, "[B,T,nq,d]")  # saved by the gate mul
    elif path.kind == "flash":
        # partial-rotary cat gives q a [B,nq,T,d] layout; flash/cuDNN output follows it, so
        # transpose(1,2).contiguous() copies (§3.4). mem-efficient allocates [B,T,nq,d]: no copy.
        acc.add("attention", "output_contiguous_copy", b * n * nq * hd, "[B,T,nq·d]")
    acc.add("attention", "gate_sigmoid", b * n * nq * hd, "[B,T,nq·d]")
    linear_inputs(acc, "attention", "o_proj", [lt.linears["o_proj"]], n, mode)
    return acc.terms


def _delta_rule_torch(acc: _Acc, batch: int, seq: int, d: Dims, *, cache: bool) -> None:
    """torch_chunk_gated_delta_rule without autocast: fp32 upcast, 64-token chunk padding and one
    saved [B,Hv,dk,dv] state per chunk (§2.2, §4.2 with verification corrections)."""
    hv, dk, dv = d.lin_value_heads, d.lin_key_dim, d.lin_value_dim
    nc = math.ceil(seq / CHUNK)
    np_ = batch * CHUNK * nc
    c = nc >= 2 or cache
    g: Group = "linear_attention"
    acc.add(g, "delta.beta_v_k_padded", F * np_ * hv * (1 + dv + dk), "[Np,Hv,1+dv+dk] fp32")
    acc.add(g, "delta.chunk_matrices", 5 * F * np_ * hv * CHUNK, "5×[Np,Hv,64] fp32")
    acc.add(g, "delta.k_beta_q_kcumdecay", 3 * F * np_ * hv * dk, "3×[Np,Hv,dk] fp32")
    if c:
        acc.add(g, "delta.key_decayed", F * np_ * hv * dk, "[Np,Hv,dk] fp32")
    if nc >= 2:
        acc.add(g, "delta.query_decayed", F * np_ * hv * dk, "[Np,Hv,dk] fp32")
    acc.add(g, "delta.cum_decay_exp", (2 + (1 if c else 0)) * F * np_ * hv, "[Np,Hv] fp32")
    if c:
        acc.add(g, "delta.chunk_decay", F * batch * hv * nc, "[B,Hv,nc] fp32")
    acc.add(g, "delta.new_values_v_new", 2 * F * np_ * hv * dv, "2×[Np,Hv,dv] fp32")
    acc.add(g, "delta.chunk_states", nc * F * batch * hv * dk * dv, "nc×[B,Hv,dk,dv] fp32")


def _delta_rule_torch_autocast(
    acc: _Acc, batch: int, seq: int, d: Dims, b: int, *, cache: bool
) -> None:
    """Same kernel under bf16 autocast: fp32 matmul operands are saved as per-use bf16 casts
    (§2.4, formulas.py q35_linear_attn_torch_autocast, corrected for nc = 1)."""
    hv, dk, dv = d.lin_value_heads, d.lin_key_dim, d.lin_value_dim
    nc = math.ceil(seq / CHUNK)
    np_ = batch * CHUNK * nc
    c = nc >= 2 or cache
    s = batch * hv * dk * dv
    n_upd = nc if cache else nc - 1
    g: Group = "linear_attention"
    acc.add(g, "delta.beta_v_k_padded", F * np_ * hv * (1 + dv + dk), "[Np,Hv,1+dv+dk] fp32")
    acc.add(g, "delta.decay_exp", F * np_ * hv * CHUNK, "[Np,Hv,64] fp32")
    acc.add(g, "delta.bf16_operand_casts", 4 * b * np_ * hv * dk, "4×[Np,Hv,dk] bf16")
    acc.add(g, "delta.bf16_matmul_results", 2 * b * np_ * hv * CHUNK, "2×[Np,Hv,64] bf16")
    acc.add(g, "delta.cum_decay_exp", (2 + (1 if c else 0)) * F * np_ * hv, "[Np,Hv] fp32")
    if c:
        acc.add(g, "delta.chunk_decay", F * batch * hv * nc, "[B,Hv,nc] fp32")
    acc.add(g, "delta.k_beta", F * np_ * hv * dk, "[Np,Hv,dk] fp32")
    acc.add(
        g,
        "delta.ut_newvalues_kcumdecay",
        F * np_ * hv * (CHUNK + dv + dk),
        "[Np,Hv,64+dv+dk] fp32",
    )
    acc.add(g, "delta.query", F * np_ * hv * dk, "[Np,Hv,dk] fp32")
    acc.add(
        g,
        "delta.per_chunk_casts",
        nc * (2 * b * s + b * batch * hv * CHUNK * dv + b * batch * hv * CHUNK * CHUNK),
        "nc×(2 states + v_new + intra) bf16",
    )
    acc.add(g, "delta.chunk_operand_casts", (nc - 1) * 2 * b * batch * hv * CHUNK * dk, "bf16")
    acc.add(
        g,
        "delta.state_update_branch",
        n_upd * (F * s + b * batch * hv * CHUNK * dv + b * batch * hv * CHUNK * dk),
        "n_upd×(state fp32 + casts)",
    )


def q35_linear_attention_layer(
    batch: int,
    seq: int,
    d: Dims,
    lt: LayerTrain,
    mode: ActMode,
    kernel: LinearKernel,
) -> list[SavedTerm]:
    """S = 2·RMSNorm + MLP + S_linmix (Gated DeltaNet, §1.3, §4.2)."""
    acc, b, n = _Acc(), mode.b, batch * seq
    hv, dk = d.lin_value_heads, d.lin_key_dim
    vd, conv = d.value_width, d.conv_dim
    g: Group = "linear_attention"
    _norms(acc, lt, n, d.hidden, mode, q35=True)
    _mlp(acc, lt, n, d, mode)
    in_proj = [lt.linears[k] for k in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")]
    linear_inputs(acc, g, "in_proj", in_proj, n, mode)
    acc.add(g, "conv.input", b * n * conv, "[B,C,T]")  # saved by conv1d even when frozen
    if kernel == "torch":
        acc.add(g, "conv.silu_input", b * batch * conv * (seq + d.conv_kernel - 1), "[B,C,T+K-1]")
    acc.add(g, "beta", b * n * hv, "[N,Hv]")
    # g = -exp(A_log.float()) * softplus(a.float() + dt_bias) runs in the modeling code for both
    # kernels. The mul keeps -exp(A_log) [Hv] even for a frozen A_log; a trainable A_log adds
    # the softplus output and exp's result (measured: one [Hv] tensor stays when frozen).
    acc.add(g, "softplus.input", F * n * hv, "[N,Hv] fp32")
    if lt.params_trainable:
        acc.add(g, "softplus.output", F * n * hv, "[N,Hv] fp32")
    acc.add(g, "a_path", (2 if lt.params_trainable else 1) * F * hv, "[Hv] fp32")
    if kernel == "torch":
        acc.add(g, "l2norm_qk", 2 * (F * n * hv * dk + F * n * hv), "2×([N,Hv,dk]+[N,Hv]) fp32")
        if mode.autocast:
            _delta_rule_torch_autocast(acc, batch, seq, d, b, cache=mode.use_cache)
        else:
            _delta_rule_torch(acc, batch, seq, d, cache=mode.use_cache)
        acc.add(g, "delta.chunk_mask", CHUNK * CHUNK, "[64,64] bool")
    else:
        # fla-core 0.5.2 + causal-conv1d 1.7.0 (§2.3, INFERRED: CUDA-only, source-read)
        acc.add(g, "fla.qk_normalized", 2 * b * n * hv * dk, "2×[N,Hv,dk]")
        acc.add(g, "fla.rstd", 2 * F * n * hv, "2×[N·Hv] fp32")
        acc.add(g, "fla.v_contiguous", b * n * vd, "[N,Vd]")
        acc.add(g, "fla.g_cumsum", F * n * hv, "[N,Hv] fp32")
        acc.add(g, "fla.A", b * n * hv * CHUNK, "[N,Hv,64]")
    norm_trains = lt.norm_trainable.get("norm", False)
    acc.add(
        g,
        "gated_norm",
        F * n * vd + F * n * hv + (b * n * vd if norm_trains else 0) + 2 * F * n * vd + b * n * vd,
        "fp32 x, rstd, z, silu(z) + bf16 w·x̂",
    )
    linear_inputs(acc, g, "out_proj", [lt.linears["out_proj"]], n, mode)
    return acc.terms


# ---------------------------------------------------------------- dense decoder (§8)


def dense_layer(
    batch: int, seq: int, d: Dims, lt: LayerTrain, mode: ActMode, path: AttnPath
) -> list[SavedTerm]:
    """S_dense = 2·RMSNorm_llama + MLP + attention (cos/sin once per model)."""
    acc, b, n = _Acc(), mode.b, batch * seq
    _norms(acc, lt, n, d.hidden, mode, q35=False)
    _mlp(acc, lt, n, d, mode)
    qkv = [lt.linears["q_proj"], lt.linears["k_proj"], lt.linears["v_proj"]]
    linear_inputs(acc, "attention", "qkv", qkv, n, mode)
    if lt.qk_norm_widths is not None:
        qw, kw = lt.qk_norm_widths
        q_rows = n * d.heads * d.head_dim // qw
        k_rows = n * d.kv_heads * d.head_dim // kw
        q_tr = lt.norm_trainable.get("q_norm", False)
        k_tr = lt.norm_trainable.get("k_norm", False)
        acc.add("attention", "q_norm", rmsnorm_llama(q_rows, qw, q_tr, b), "[N·nq,d]")
        acc.add("attention", "k_norm", rmsnorm_llama(k_rows, kw, k_tr, b), "[N·nkv,d]")
    _sdpa_or_eager(acc, batch, seq, d, mode, path)
    # SDPA output keeps q's [B,T,nq,d] layout, so it IS the o_proj input (no copy, §3.4).
    linear_inputs(
        acc,
        "attention",
        "o_proj",
        [lt.linears["o_proj"]],
        n,
        mode,
        saved_elsewhere=path.kind != "eager",
    )
    return acc.terms
