"""Training-step and no-grad ledgers (plan §9.5, research implications C)."""

from __future__ import annotations

import math
from types import ModuleType

import pytest
from arch_helpers import by_name, make_cfg

from vramforge_estimator.architectures import StepTimepoints, get_adapter
from vramforge_estimator.architectures import activations as act
from vramforge_estimator.architectures.ledger import K_GC_BACKWARD, K_GC_FORWARD
from vramforge_estimator.schemas import (
    AllocationCategory,
    Evidence,
    ModelInventory,
    Objective,
    ResolvedConfig,
    SequenceShape,
    Strategy,
)

TPS = StepTimepoints(
    "POLICY_FORWARD_BACKWARD:forward",
    "POLICY_FORWARD_BACKWARD:loss",
    "POLICY_FORWARD_BACKWARD:backward",
)
ALL = [TPS.forward, TPS.loss, TPS.backward]
MiB = 2**20
H = 4096
HYBRID = get_adapter("qwen3_5_hybrid")
DENSE = get_adapter("dense_decoder")


@pytest.fixture(scope="module")
def auto(mimo: ModelInventory) -> list[str]:
    return [m.name for m in HYBRID.lora_target_modules(mimo, "auto_verified", [])]


def step(inv: ModelInventory, cfg: ResolvedConfig, batch: int = 1, seq: int = 4096) -> dict:
    return by_name(
        HYBRID.train_step_ledger(inv, cfg, SequenceShape(batch=batch, seq_len=seq), TPS, "policy")
    )


def _x(lin: dict[str, tuple[int, int]]) -> act.LayerTrain:
    return act.LayerTrain(
        {k: act.ModuleTrain(k, i, o, rank=16) for k, (i, o) in lin.items()}, {}, False
    )


S_LIN_QLORA = act.total(  # LoRA bf16 adapters, torch fallback under bf16 autocast, B=1 T=4096
    act.q35_linear_attention_layer(
        1,
        4096,
        act.Dims(4096, 12288, 16, 4, 256, 16, 32, 128, 128, 4),
        _x(
            {
                "gate_proj": (H, 12288),
                "up_proj": (H, 12288),
                "down_proj": (12288, H),
                "in_proj_qkv": (H, 8192),
                "in_proj_z": (H, H),
                "in_proj_a": (H, 32),
                "in_proj_b": (H, 32),
                "out_proj": (H, H),
            }
        ),
        act.ActMode(2, True, adapter_bytes=2),
        "torch",
    )
)


def test_gc_keeps_boundaries_and_recomputes_the_largest_layer(
    mimo: ModelInventory, auto: list[str]
) -> None:
    led = step(mimo, make_cfg(targets=auto))
    b = led["policy.act.ckpt_boundaries"]
    assert b.bytes_low == b.bytes_high == 32 * 4096 * H * 2 and b.live_at == ALL
    assert b.count == 32 and b.saved_for_backward
    assert not any(".linear_attention." in n or ".full_attention." in n for n in led)
    rec = led["policy.act.recompute.backward"]
    assert rec.category is AllocationCategory.RECOMPUTE_WORKING_SET
    assert rec.live_at == [TPS.backward] and rec.evidence is Evidence.ASSUMPTION
    lo, hi = K_GC_BACKWARD["lora_shared"]
    assert (rec.bytes_low, rec.bytes_high) == (
        math.ceil(lo * S_LIN_QLORA),
        math.ceil(hi * S_LIN_QLORA),
    )
    fwd = led["policy.act.layer_transient.forward"]
    assert fwd.live_at == [TPS.forward]
    assert fwd.bytes_high == math.ceil(K_GC_FORWARD["lora_shared"][1] * S_LIN_QLORA)
    assert led["policy.act.rope_cos_sin"].bytes_low == 2 * 2 * 4096 * 64  # [B,T,r] bf16, 1 MiB
    assert led["policy.act.final_norm"].bytes_low == act.rmsnorm_q35(4096, H, False)  # 64.03 MiB
    hidden = led["policy.act.final_hidden"]
    assert hidden.storage_alias_group == "policy.final_hidden"
    assert hidden.live_at == [TPS.forward, TPS.loss]


def test_boundaries_follow_the_load_dtype(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(targets=auto, load="float32"))
    assert led["policy.act.ckpt_boundaries"].bytes_low == 32 * 4096 * H * 4
    # verified saved-set formulas are bf16-only: per-layer sizes are unknown, never 0
    rec = led["policy.act.recompute.backward"]
    assert rec.bytes_low is None and rec.bytes_high is None
    assert rec.evidence is Evidence.UNKNOWN and "float32" in (rec.note or "")
    assert led["policy.act.final_norm"].bytes_low is None


def test_hybrid_layers_use_their_own_formulas(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(targets=auto, gc=False))
    lin = [a for n, a in led.items() if n.startswith("policy.act.linear_attention.")]
    full = [a for n, a in led.items() if n.startswith("policy.act.full_attention.")]
    assert {a.count for a in lin} == {24} and {a.count for a in full} == {8}
    assert "policy.act.linear_attention.linear_attention" in led
    assert "policy.act.full_attention.attention" in led
    assert "policy.act.linear_attention.attention" not in led
    total = sum(a.bytes_low for a in lin + full)
    # 24 linear + 8 full layers (LoRA bf16 adapters, autocast): 56.42 GiB
    assert round(total / 2**30, 2) == 56.42
    assert all(a.live_at == ALL and a.evidence is Evidence.ANALYTIC for a in lin + full)
    assert all(
        (a.formula_ref or "").startswith("methodology-architectures.md#") for a in led.values()
    )


def test_no_gc_transients(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(targets=auto, gc=False))
    assert "policy.act.ckpt_boundaries" not in led
    assert led["policy.act.layer_transient.backward"].live_at == [TPS.backward]
    assert led["policy.act.layer_transient.backward"].bytes_high == math.ceil(0.16 * S_LIN_QLORA)


def test_padding_adds_mask_terms_as_a_range(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(objective=Objective.DPO, targets=auto, gc=False), batch=2)
    bool_mask = led["policy.act.attn_mask.full_attention"]
    assert (bool_mask.bytes_low, bool_mask.bytes_high) == (0, 2 * 4096 * 4096)  # 32 MiB shared
    mask = led["policy.act.full_attention.mask"]
    assert (mask.bytes_low, mask.bytes_high) == (0, 8 * 64 * MiB)  # additive mask 64 MiB/layer
    attn = led["policy.act.full_attention.attention"]
    # per full layer at B=2: K/V expansion +96 MiB, flash's contiguous copy b·N·nq·d -64 MiB
    assert attn.bytes_high - attn.bytes_low == 8 * (96 - 64) * MiB
    assert mask.live_at == ALL and bool_mask.live_at == [TPS.forward]  # no GC: not kept
    assert mask.formula_ref == "methodology-architectures.md#act-mask"
    assert (led["policy.act.full_attention.lora"].formula_ref or "").endswith("#act-lora")


def test_single_unpadded_sequence_has_no_mask(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(targets=auto, gc=False))
    assert not any("mask" in n for n in led)
    padded = step(mimo, make_cfg(targets=auto, gc=False, pad_to_multiple_of=64))
    assert "policy.act.attn_mask.full_attention" in padded
    grpo = step(mimo, make_cfg(objective=Objective.GRPO, targets=auto, gc=False))
    assert "policy.act.attn_mask.full_attention" in grpo  # left-padded prompts


def test_gc_keeps_the_bool_mask_through_backward(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(objective=Objective.DPO, targets=auto), batch=2)
    assert led["policy.act.attn_mask.full_attention"].live_at == ALL


def test_full_finetune_trains_norms(mimo: ModelInventory) -> None:
    led = step(mimo, make_cfg(strategy=Strategy.FULL, gc=False))
    assert led["policy.act.final_norm"].bytes_low == act.rmsnorm_q35(4096, H, True)  # 128.03 MiB
    lin = led["policy.act.linear_attention.linear_attention"]
    assert "policy.act.linear_attention.lora" not in led
    assert lin.bytes_low > 0


def test_kernel_paths(mimo: ModelInventory, auto: list[str]) -> None:
    torch = step(mimo, make_cfg(targets=auto, gc=False))
    fla = step(mimo, make_cfg(targets=auto, gc=False, paths={"linear_attention": "fla"}))
    key = "policy.act.linear_attention.linear_attention"
    assert fla[key].bytes_low < torch[key].bytes_low
    assert "기본 sdpa" in (fla["policy.act.full_attention.attention"].note or "")
    eager = step(mimo, make_cfg(targets=auto, gc=False, paths={"full_attention": "eager"}))
    em = eager["policy.act.attn_mask.full_attention"]
    assert em.bytes_low == em.bytes_high == 2 * 4096 * 4096  # eager float mask, always
    gap = eager["policy.act.full_attention.attention"].bytes_low
    assert gap - torch["policy.act.full_attention.attention"].bytes_low > 8 * 6 * 16 * 4096**2 // 2
    leaked = step(mimo, make_cfg(targets=auto, gc=False, paths={"full_attention": "auto"}))
    assert leaked[key].bytes_low == torch[key].bytes_low  # "auto" = the default path, noted
    assert "기본 sdpa" in (leaked["policy.act.full_attention.attention"].note or "")
    fa2 = step(
        mimo, make_cfg(targets=auto, gc=False, paths={"full_attention": "flash_attention_2"})
    )
    assert fa2["policy.act.full_attention"].bytes_low is None
    assert fa2[key].bytes_low == torch[key].bytes_low  # linear layers still computed
    assert fa2["policy.act.layer_transient.backward"].bytes_high is None


def test_qlora_dequant_transients(mimo: ModelInventory, auto: list[str]) -> None:
    gc = step(mimo, make_cfg(targets=auto))
    assert gc["policy.act.q4_dequant.forward"].bytes_low == 106_954_752  # gate_proj, +8·nb
    assert gc["policy.act.q4_dequant.backward"].bytes_low == 106_954_752  # recompute forward
    nogc = step(mimo, make_cfg(targets=auto, gc=False))
    assert nogc["policy.act.q4_dequant.backward"].bytes_low == 103_809_024  # bf16 load
    fp32 = step(mimo, make_cfg(targets=auto, gc=False, load="float32"))
    assert fp32["policy.act.q4_dequant.backward"].bytes_low == 305_135_616
    lora = step(mimo, make_cfg(strategy=Strategy.LORA, targets=auto))
    assert not any("q4_dequant" in n for n in lora)
    short = step(mimo, make_cfg(targets=auto), seq=1000)  # 5 <= M <= 1536: fused or dequant
    q = short["policy.act.q4_dequant.forward"]
    assert q.bytes_low == 1000 * 12288 * 2 and q.bytes_high == 106_954_752  # clone .. dequant


def test_dora_is_unknown_not_zero(mimo: ModelInventory, auto: list[str]) -> None:
    led = step(mimo, make_cfg(targets=auto, dora=True))
    dora = led["policy.act.dora_extra"]
    assert dora.bytes_low is None and dora.evidence is Evidence.UNKNOWN


def test_use_cache_branch_lives_until_the_loss(ib: ModuleType) -> None:
    inv = ib.tiny_q35_inventory()
    cfg = make_cfg(strategy=Strategy.FULL, gc=False, use_cache=True)
    led = by_name(HYBRID.train_step_ledger(inv, cfg, SequenceShape(batch=1, seq_len=40), TPS, "p"))
    branch = led["p.act.cache_branch"]
    # nc = 1 with a cache keeps the final state-update branch (bf16 autocast, measured on CPU
    # with the cache tensors as extra graph roots: 1,589,284 vs 1,502,848 B)
    assert branch.bytes_low == 1_589_284 - 1_502_848
    assert branch.live_at == [TPS.forward, TPS.loss]
    states = led["p.act.cache_states"]
    assert states.bytes_low == 396 * 4 * 2 + 4 * 9 * 24 * 28


def test_dense_ledger_rope_range_and_layers(ib: ModuleType) -> None:
    inv = ib.tiny_dense_inventory("qwen3", qk_norm=True)
    cfg = make_cfg(strategy=Strategy.FULL, gc=False, paths={"full_attention": "sdpa"})
    led = by_name(DENSE.train_step_ledger(inv, cfg, SequenceShape(batch=2, seq_len=100), TPS, "p"))
    rope = led["p.act.rope_cos_sin"]
    assert (rope.bytes_low, rope.bytes_high) == (2 * 2 * 100 * 40, 2 * 2 * 2 * 100 * 40)
    lt = act.LayerTrain(
        {
            k: act.ModuleTrain(k, i, o, trainable=True)
            for k, (i, o) in {
                "q_proj": (96, 240),
                "k_proj": (96, 80),
                "v_proj": (96, 80),
                "o_proj": (240, 96),
                "gate_proj": (96, 176),
                "up_proj": (96, 176),
                "down_proj": (176, 96),
            }.items()
        },
        dict.fromkeys(("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"), True),
        True,
        (40, 40),
    )
    one = act.dense_layer(
        2,
        100,
        act.Dims(96, 176, 6, 2, 40),
        lt,
        act.ActMode(2, True),
        act.AttnPath("flash", False, False),
    )
    layers = [a for n, a in led.items() if n.startswith("p.act.full_attention.")]
    assert sum(a.bytes_low for a in layers) == 2 * act.total(one)  # unpadded variant = low


def test_no_grad_forward_working_set(mimo: ModelInventory, auto: list[str]) -> None:
    cfg = make_cfg(objective=Objective.DPO, targets=auto)
    led = by_name(
        HYBRID.no_grad_forward_ledger(
            mimo, cfg, SequenceShape(batch=2, seq_len=1024), ["R:fwd"], "ref"
        )
    )
    assert led["ref.nograd.hidden"].bytes_low == 2 * 1024 * H * 2
    assert all(a.live_at == ["R:fwd"] for a in led.values())
    ws = led["ref.nograd.layer_working_set"]
    assert ws.evidence is Evidence.ASSUMPTION and ws.bytes_low < ws.bytes_high
    assert led["ref.nograd.attn_mask.full_attention"].bytes_high == 2 * 1024 * 1024
    assert led["ref.nograd.q4_dequant"].bytes_low == 106_954_752
    assert not any(a.category is AllocationCategory.SAVED_ACTIVATIONS for a in led.values())
