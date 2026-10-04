"""Per-layer saved-activation formulas (plan §9.5) against docs/research/architecture-memory.md.

Golden values exclude the 0-dim wrapped scalars autograd keeps for `tensor * python_float` (§F).
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from vramforge_estimator.architectures import activations as act
from vramforge_estimator.architectures.activations import (
    ActMode,
    AttnPath,
    Dims,
    LayerTrain,
    ModuleTrain,
    total,
)

MiB = 2**20
NORMS = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "norm")

# tiny config A (§4.1): H=96, I=176, nq=6, nkv=2, d=40, Hk=3, Hv=9, dk=24, dv=28, K=4
A = Dims(96, 176, 6, 2, 40, 3, 9, 24, 28, 4)
A_MLP = {"gate_proj": (96, 176), "up_proj": (96, 176), "down_proj": (176, 96)}
A_LIN = {
    **A_MLP,
    "in_proj_qkv": (96, 396),
    "in_proj_z": (96, 252),
    "in_proj_a": (96, 9),
    "in_proj_b": (96, 9),
    "out_proj": (252, 96),
}
A_FULL = {**A_MLP, "q_proj": (96, 480), "k_proj": (96, 80), "v_proj": (96, 80), "o_proj": (240, 96)}
A_DENSE = {
    **A_MLP,
    "q_proj": (96, 240),
    "k_proj": (96, 80),
    "v_proj": (96, 80),
    "o_proj": (240, 96),
}

# example model (§0): H=4096, I=12288, nq=16, nkv=4, d=256, Hk=16, Hv=32, dk=dv=128, K=4
X = Dims(4096, 12288, 16, 4, 256, 16, 32, 128, 128, 4)
X_MLP = {"gate_proj": (4096, 12288), "up_proj": (4096, 12288), "down_proj": (12288, 4096)}
X_LIN = {
    **X_MLP,
    "in_proj_qkv": (4096, 8192),
    "in_proj_z": (4096, 4096),
    "in_proj_a": (4096, 32),
    "in_proj_b": (4096, 32),
    "out_proj": (4096, 4096),
}
X_FULL = {
    **X_MLP,
    "q_proj": (4096, 8192),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
}

FLASH = AttnPath("flash", mask=False, expand_kv=False)
PLAIN = ActMode(b=2, autocast=False)
AUTOCAST = ActMode(b=2, autocast=True)


def layer(
    kinds: Mapping[str, tuple[int, int]],
    *,
    full: bool = True,
    rank: int | None = None,
    qk: tuple[int, int] | None = None,
) -> LayerTrain:
    return LayerTrain(
        linears={k: ModuleTrain(k, i, o, trainable=full, rank=rank) for k, (i, o) in kinds.items()},
        norm_trainable=dict.fromkeys(NORMS, full),
        params_trainable=full,
        qk_norm_widths=qk,
    )


def rope(batch: int, seq: int, rotary: int = 10) -> int:
    return 2 * 2 * batch * seq * rotary  # cos/sin [B,T,r] bf16, once per model


# §F golden table: tiny config A, bf16, full FT, SDPA, use_cache=False, no GC
GOLDEN = {
    (1, 100): (3_737_584, 3_370_096, 872_288, 620_800, 425_600),
    (2, 200): (14_930_776, 13_739_608, 3_485_888, 2_483_200, 1_702_400),
    (1, 40): (1_645_696, 1_502_848, 349_568, 248_320, 170_240),  # corrected nc = 1 row
}


@pytest.mark.parametrize(("batch", "seq"), list(GOLDEN))
def test_research_golden_layers(batch: int, seq: int) -> None:
    lin, lin_ac, full, qwen3, llama = GOLDEN[(batch, seq)]
    assert total(act.q35_linear_attention_layer(batch, seq, A, layer(A_LIN), PLAIN, "torch")) == lin
    ac = act.q35_linear_attention_layer(batch, seq, A, layer(A_LIN), AUTOCAST, "torch")
    assert total(ac) == lin_ac
    attn = act.q35_full_attention_layer(batch, seq, A, layer(A_FULL), PLAIN, FLASH)
    assert total(attn) + rope(batch, seq) == full  # "first full layer, cos/sin included"
    q3 = act.dense_layer(batch, seq, A, layer(A_DENSE, qk=(40, 40)), PLAIN, FLASH)
    assert total(q3) == qwen3
    assert total(act.dense_layer(batch, seq, A, layer(A_DENSE), PLAIN, FLASH)) == llama


def test_research_lora_fixture_a() -> None:
    # §F (a): 3-layer [lin, lin, full], all-linear r=8, B=1, T=100, layer 1 (no autocast)
    fp32 = act.q35_linear_attention_layer(
        1, 100, A, layer(A_LIN, full=False, rank=8), ActMode(2, False, adapter_bytes=4), "torch"
    )
    bf16 = act.q35_linear_attention_layer(
        1, 100, A, layer(A_LIN, full=False, rank=8), ActMode(2, False, adapter_bytes=2), "torch"
    )
    assert total(fp32) == 3_909_948
    assert total(bf16) == 3_619_548


def test_frozen_a_log_keeps_one_hv_tensor() -> None:
    # measured: a frozen A_log still leaves -exp(A_log) [Hv] fp32 saved by the mul (36 B here)
    frozen = act.q35_linear_attention_layer(1, 100, A, layer(A_LIN, full=False), PLAIN, "torch")
    assert total(frozen) == 3_482_748
    assert [t.nbytes for t in frozen if t.label == "a_path"] == [4 * 9]


@pytest.mark.parametrize(("batch", "seq"), [(1, 100), (2, 200), (1, 64)])
def test_eager_adds_6_nq_bytes_per_t_squared(batch: int, seq: int) -> None:
    eager = AttnPath("eager", mask=True, expand_kv=True)
    e = total(act.q35_full_attention_layer(batch, seq, A, layer(A_FULL), PLAIN, eager))
    f = total(act.q35_full_attention_layer(batch, seq, A, layer(A_FULL), PLAIN, FLASH))
    # fit (§4.2): +136 B/token and (f+b)·nq = 36 B per B·T² for config A
    assert e - f == 136 * batch * seq + 36 * batch * seq * seq


def test_mem_efficient_mask_path_composition() -> None:
    b, batch, seq = 2, 2, 100
    n, nq, nkv, d = batch * seq, 6, 2, 40
    masked = AttnPath("mem_efficient", mask=True, expand_kv=True)
    m = total(act.q35_full_attention_layer(batch, seq, A, layer(A_FULL), PLAIN, masked))
    f = total(act.q35_full_attention_layer(batch, seq, A, layer(A_FULL), PLAIN, FLASH))
    expected = (
        2 * b * n * (nq - nkv) * d  # repeat_kv K/V copies (§3.1)
        + b * batch * seq * 104  # additive mask per layer, last dim padded to 8 (attention.cpp)
        + 4 * batch * nq * (128 - seq)  # mem-efficient lse padded to 32
        - b * n * nq * d  # no contiguous copy: output already [B,T,nq,d] (§3.4)
    )
    assert m - f == expected


def test_dense_mask_path_has_no_copy_to_remove() -> None:
    masked = AttnPath("mem_efficient", mask=True, expand_kv=True)
    m = total(act.dense_layer(2, 128, A, layer(A_DENSE), PLAIN, masked))
    f = total(act.dense_layer(2, 128, A, layer(A_DENSE), PLAIN, FLASH))
    assert m - f == 2 * 2 * 256 * 4 * 40 + 2 * 2 * 128 * 128  # T multiple of 8 and 32


def _mib(terms: list[act.SavedTerm]) -> float:
    return round(total(terms) / MiB, 2)


def test_example_model_real_dims_full_finetune() -> None:
    # §10.2 + verification V3 (FakeTensorMode at real dims, B=1, T=4096)
    lin = act.q35_linear_attention_layer(1, 4096, X, layer(X_LIN), PLAIN, "torch")
    assert total(lin) == 2_219_700_480  # measured 2,219,700,488 incl. one 8 B wrapped scalar
    lin_ac = act.q35_linear_attention_layer(1, 4096, X, layer(X_LIN), AUTOCAST, "torch")
    assert _mib(lin_ac) == 2288.87
    full = act.q35_full_attention_layer(1, 4096, X, layer(X_FULL), PLAIN, FLASH)
    assert _mib(full) == 1040.63  # 1,041.63 MiB measured incl. the 1 MiB cos/sin
    assert (
        _mib(act.q35_linear_attention_layer(1, 4096, X, layer(X_LIN), AUTOCAST, "fla")) == 1171.31
    )
    # Σ saved, no GC: 24·S_lin + 8·S_full + final norm + lm_head input + cos/sin + indices
    n = 4096
    outside = act.rmsnorm_q35(n, 4096, True) + 2 * n * 4096 + rope(1, n, 64) + 8 * n
    assert round((24 * total(lin) + 8 * total(full) + outside) / 2**30, 2) == 57.90
    # GC retained: 32 checkpoint inputs + the same outside items (1,185.1 MiB)
    assert round((32 * 2 * n * 4096 + outside) / MiB, 1) == 1185.1


@pytest.mark.parametrize(
    ("adapter_bytes", "autocast", "kernel", "s_lin", "s_full"),
    [
        (2, False, "torch", 1957.37, 833.50),  # TRL QLoRA bf16 adapters
        (2, True, "torch", 2129.37, 833.50),
        (2, True, "fla", 1011.81, 833.50),
        (4, True, "torch", 2260.00, 931.94),  # non-quantized LoRA: fp32 adapters + autocast
        (4, True, "fla", 1142.44, 931.94),
    ],
)
def test_example_model_real_dims_lora(
    adapter_bytes: int, autocast: bool, kernel: act.LinearKernel, s_lin: float, s_full: float
) -> None:
    mode = ActMode(2, autocast, adapter_bytes=adapter_bytes)
    lin = act.q35_linear_attention_layer(
        1, 4096, X, layer(X_LIN, full=False, rank=16), mode, kernel
    )
    full = act.q35_full_attention_layer(1, 4096, X, layer(X_FULL, full=False, rank=16), mode, FLASH)
    assert (_mib(lin), _mib(full)) == (s_lin, s_full)


def test_lora_terms_are_attributed_to_the_lora_group() -> None:
    mode = ActMode(2, True, adapter_bytes=4)
    terms = act.q35_linear_attention_layer(
        1, 100, A, layer(A_LIN, full=False, rank=8), mode, "torch"
    )
    lora = [t for t in terms if t.group == "lora"]
    # 8 adapted Linear: input cast + lora_B input + cast A/B weights each
    assert len(lora) == 3 * 8
    assert {t.group for t in terms} == {"norms", "mlp", "linear_attention", "lora"}


def test_dropout_adds_masks_and_unshares_bf16_inputs() -> None:
    base = ActMode(2, True, adapter_bytes=2)
    drop = ActMode(2, True, adapter_bytes=2, dropout=0.1)
    lt = layer(A_LIN, full=False, rank=8)
    a = act.q35_linear_attention_layer(1, 100, A, lt, base, "torch")
    b = act.q35_linear_attention_layer(1, 100, A, lt, drop, "torch")
    n = 100
    ins = [96] * 4 + [252] + [96] * 2 + [176]
    shared = 2 * n * (96 + 252 + 96 + 176)  # four shared inputs no longer saved
    assert total(b) - total(a) == sum(3 * n * i for i in ins) - shared
