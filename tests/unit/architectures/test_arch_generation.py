"""Generation ledger (plan §9.7, §19.1, §19.3; research §7, E; trl-grpo R3)."""

from __future__ import annotations

from types import ModuleType

import pytest
from arch_helpers import by_name, make_cfg

from vramforge_estimator.architectures import GenerationTimepoints, get_adapter
from vramforge_estimator.schemas import (
    AllocationCategory,
    Evidence,
    ModelInventory,
    Objective,
    ResolvedConfig,
    Strategy,
)

TPS = GenerationTimepoints(
    "ROLLOUT_PREFILL_AND_DECODE:prefill", "ROLLOUT_PREFILL_AND_DECODE:decode"
)
HYBRID = get_adapter("qwen3_5_hybrid")
DENSE = get_adapter("dense_decoder")
MiB = 2**20


@pytest.fixture(scope="module")
def auto(mimo: ModelInventory) -> list[str]:
    return [m.name for m in HYBRID.lora_target_modules(mimo, "auto_verified", [])]


def gen(inv: ModelInventory, cfg: ResolvedConfig, c: int, p: int, new: int, adapter=HYBRID) -> dict:
    return by_name(adapter.generation_ledger(inv, cfg, c, p, new, TPS, "policy"))


def cache_bytes(led: dict, when: str = "decode") -> int:
    return sum(
        a.bytes_low
        for a in led.values()
        if a.category in (AllocationCategory.GENERATION_CACHE, AllocationCategory.RECURRENT_STATE)
        and getattr(TPS, when) in a.live_at
    )


def test_plan_kv_payload_row(ib: ModuleType) -> None:
    # 2 full-attention layers, 4 KV heads, head_dim 128, 2-byte dtype, 1024 cached positions
    inv = ib.tiny_dense_inventory(
        "llama", hidden_size=512, num_attention_heads=4, num_key_value_heads=4, head_dim=128
    )
    led = gen(inv, make_cfg(strategy=Strategy.LORA), 1, 1024, 1, adapter=DENSE)
    kv = led["policy.gen.kv_cache.full_attention.decode"]
    assert kv.bytes_low == kv.bytes_high == 4_194_304
    assert kv.evidence is Evidence.ANALYTIC and kv.dims["L"] == 1024


@pytest.mark.parametrize(("prompt", "new"), [(100, 925), (2048, 2049), (1, 32768)])
def test_example_model_cache_per_sequence(
    mimo: ModelInventory, auto: list[str], prompt: int, new: int
) -> None:
    led = gen(mimo, make_cfg(objective=Objective.GRPO, targets=auto), 1, prompt, new)
    cached = prompt + new - 1  # the last generated token is never fed back
    assert cache_bytes(led) == 51_904_512 + 32_768 * cached  # research §7 (bf16 load)


def test_hybrid_kv_only_for_full_attention_layers(mimo: ModelInventory, auto: list[str]) -> None:
    led = gen(mimo, make_cfg(objective=Objective.GRPO, targets=auto), 4, 272, 1024)
    kv = led["policy.gen.kv_cache.full_attention.decode"]
    per_layer = 2 * 4 * 256 * 2 * 4 * (272 + 1023)
    assert kv.count == 8 and kv.dims["layers"] == 8
    assert kv.bytes_low == 8 * per_layer == 169_738_240  # 161.875 MiB (trl-grpo R3 table)
    assert kv.bytes_low != 32 * per_layer  # a generic KV formula over all 32 layers is wrong
    assert not any("kv_cache.linear_attention" in n for n in led)
    conv = led["policy.gen.linear_state.conv"]
    rec = led["policy.gen.linear_state.recurrent"]
    assert conv.count == rec.count == 24
    assert conv.bytes_low + rec.bytes_low == 198 * MiB  # C=4: 49.5 MiB per sequence
    assert conv.live_at == rec.live_at == [TPS.prefill, TPS.decode]
    assert (
        led["policy.gen.kv_cache.full_attention.prefill"].bytes_low == 8 * 2 * 4 * 256 * 2 * 4 * 272
    )


@pytest.mark.parametrize(
    ("strategy", "load", "kv_bytes", "conv_dtype"),
    [
        (Strategy.QLORA, "float32", 4, "float32"),  # case Q: PEFT generate has no autocast
        (Strategy.QLORA, "bfloat16", 2, "bfloat16"),  # case R
        (Strategy.LORA, "float32", 4, "float32"),  # case C
        (Strategy.FULL, "float32", 4, "bfloat16"),  # case A: autocast conv, promoted K/V
        (Strategy.FULL, "bfloat16", 2, "bfloat16"),  # case F
    ],
)
def test_rollout_cache_dtypes(
    mimo: ModelInventory,
    auto: list[str],
    strategy: Strategy,
    load: str,
    kv_bytes: int,
    conv_dtype: str,
) -> None:
    targets = auto if strategy is not Strategy.FULL else None
    cfg = make_cfg(objective=Objective.GRPO, strategy=strategy, load=load, targets=targets)
    led = gen(mimo, cfg, 1, 100, 1)
    assert (
        led["policy.gen.kv_cache.full_attention.decode"].bytes_low
        == 8 * 2 * 4 * 256 * kv_bytes * 100
    )
    conv = led["policy.gen.linear_state.conv"]
    assert conv.dtype == conv_dtype
    assert led["policy.gen.linear_state.recurrent"].dtype == "float32"


def test_recurrent_state_stays_fp32_whatever_the_resolver_says(
    mimo: ModelInventory, auto: list[str]
) -> None:
    cfg = make_cfg(objective=Objective.GRPO, targets=auto)
    cfg = cfg.model_copy(
        update={
            "effective_dtypes": cfg.effective_dtypes.model_copy(
                update={"recurrent_state": "bfloat16"}
            )
        }
    )
    rec = gen(mimo, cfg, 1, 10, 2)["policy.gen.linear_state.recurrent"]
    assert rec.bytes_low == 24 * 4 * 32 * 128 * 128 and "bfloat16" in (rec.note or "")


def test_decode_step_and_prefill_working_set(mimo: ModelInventory, auto: list[str]) -> None:
    led = gen(mimo, make_cfg(objective=Objective.GRPO, targets=auto), 4, 272, 1024)
    step = led["policy.gen.decode.step_transient"]
    length = 272 + 1023
    one = 4 * 4 * 256 * 2 * length  # K (or V) of one layer at the last decode step
    expand = 2 * 4 * 16 * length * 256 * 2  # left-padded GRPO prompts: repeat_kv to nq=16 heads
    state = 4 * 32 * 128 * 128 * 4  # one layer's [C,Hv,dk,dv] fp32 state
    assert step.dims == {
        "C": 4,
        "L": length,
        "cat": 2 * one,
        "kv_repeat": expand,
        "recurrent_step": 3 * state + 8 * 4 * 4 * 32 * 256,
        "q4": 4 * 12288 * 2,  # M = C = 4: fused kernel, only the LoRA wrapper's clone
    }
    # layers run one after another: the step holds the largest transient, not the sum
    assert (step.bytes_low, step.bytes_high) == (3 * state, expand)
    assert step.live_at == [TPS.decode] and step.evidence is Evidence.ASSUMPTION
    assert step.category is AllocationCategory.WORKSPACE
    prefill = [a for n, a in led.items() if n.startswith("policy.gen.prefill.")]
    assert prefill and all(a.live_at == [TPS.prefill] for a in prefill)
    assert led["policy.gen.prefill.nograd.hidden"].bytes_low == 4 * 272 * 4096 * 2
    assert led["policy.gen.prefill.nograd.attn_mask.full_attention"].bytes_high == 4 * 272 * 272


def test_decode_kv_repeat_needs_a_padding_mask_and_gqa(ib: ModuleType) -> None:
    gqa = ib.tiny_dense_inventory("llama")  # nq = 6, nkv = 2, d = 40
    lora = make_cfg(strategy=Strategy.LORA)
    one_seq = gen(gqa, lora, 1, 10, 5, adapter=DENSE)["policy.gen.decode.step_transient"]
    assert "kv_repeat" not in one_seq.dims  # B = 1, no padding: enable_gqa, no copy
    padded = gen(gqa, lora, 3, 10, 5, adapter=DENSE)["policy.gen.decode.step_transient"]
    assert padded.dims["kv_repeat"] == 2 * 3 * 6 * 14 * 40 * 2
    assert padded.bytes_low == 3 * 2 * 14 * 40 * 2  # cat low: no padding means no copy
    eager = make_cfg(strategy=Strategy.LORA, paths={"full_attention": "eager"})
    e = gen(gqa, eager, 1, 10, 5, adapter=DENSE)["policy.gen.decode.step_transient"]
    assert e.bytes_low == 2 * 1 * 6 * 14 * 40 * 2  # eager always repeats K/V
    mha = ib.tiny_dense_inventory("llama", num_key_value_heads=6)
    assert "kv_repeat" not in gen(mha, lora, 3, 10, 5, adapter=DENSE)[
        "policy.gen.decode.step_transient"
    ].dims


def test_unverified_decode_paths_are_unknown(mimo: ModelInventory, auto: list[str]) -> None:
    fa2 = make_cfg(objective=Objective.GRPO, targets=auto, paths={"full_attention": "fa2"})
    step = gen(mimo, fa2, 4, 64, 8)["policy.gen.decode.step_transient"]
    assert step.bytes_low is None and step.evidence is Evidence.UNKNOWN
    assert "attention" in (step.note or "")


def test_single_new_token_has_no_decode_step(mimo: ModelInventory, auto: list[str]) -> None:
    led = gen(mimo, make_cfg(objective=Objective.GRPO, targets=auto), 2, 64, 1)
    assert "policy.gen.decode.step_transient" not in led
    assert led["policy.gen.kv_cache.full_attention.decode"].dims["L"] == 64


def test_sliding_window_caps_the_cache(ib: ModuleType) -> None:
    inv = ib.tiny_dense_inventory("mistral", sliding_window=64)
    led = gen(inv, make_cfg(strategy=Strategy.LORA), 2, 50, 40, adapter=DENSE)
    per_pos = 2 * 2 * 40 * 2  # K+V, nkv=2, d=40, bf16
    assert led["policy.gen.kv_cache.sliding_attention.prefill"].bytes_low == 2 * 2 * 50 * per_pos
    assert led["policy.gen.kv_cache.sliding_attention.decode"].bytes_low == 2 * 2 * 64 * per_pos
    long_prompt = gen(inv, make_cfg(strategy=Strategy.LORA), 1, 200, 10, adapter=DENSE)
    # prefill keeps a view of the whole prompt's cat result until the first decode step
    assert long_prompt["policy.gen.kv_cache.sliding_attention.prefill"].dims["L"] == 200
    assert long_prompt["policy.gen.kv_cache.sliding_attention.decode"].dims["L"] == 64


def test_dense_kv_for_every_attention_layer(ib: ModuleType) -> None:
    inv = ib.tiny_dense_inventory("llama")
    led = gen(inv, make_cfg(strategy=Strategy.LORA), 3, 10, 5, adapter=DENSE)
    kv = led["policy.gen.kv_cache.full_attention.decode"]
    assert kv.count == 2 and kv.bytes_low == 2 * 3 * 14 * (2 * 2 * 40 * 2)
    assert not any("linear_state" in n for n in led)
