"""Resident weights, bitsandbytes 4-bit bytes and load transients (plan §9.3, §19.3).

Expected values: docs/research/loading-quantization-peft.md §3.4-3.5 (verified byte-exact E3/E4,
V2/V3) and §3.5 load-phase bounds.
"""

from __future__ import annotations

from types import ModuleType

import pytest
from arch_helpers import by_name, make_cfg

from vramforge_estimator.architectures.structure import ModelStructure
from vramforge_estimator.architectures.trainable import trainable_group_list
from vramforge_estimator.architectures.weights import (
    bnb_q4_bytes,
    device_map_load_bytes,
    load_transient_allocations,
    quantized_modules,
    resident_weight_allocations,
)
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    AllocationCategory,
    ErrorCode,
    Evidence,
    ModelInventory,
    Strategy,
)

LIVE = ["MODEL_LOAD_AND_QUANTIZE:*", "POLICY_FORWARD_BACKWARD:*"]


@pytest.mark.parametrize(
    ("rows", "cols", "no_dq", "dq"),
    [
        (4096, 4096, 9_437_248, 8_655_940),
        (12288, 4096, 28_311_616, 25_965_636),
        (8192, 4096, 18_874_432, 17_310_788),
        (32, 4096, 73_792, 68_708),
        (4097, 3, 6_146 + 4 * 193 + 64, 6_146 + 193 + 4 + 4 + 64 + 1024),  # partial last block
    ],
)
def test_bnb_4bit_bytes_match_research_table(rows: int, cols: int, no_dq: int, dq: int) -> None:
    n = rows * cols
    assert bnb_q4_bytes(n, double_quant=False).total == no_dq
    assert bnb_q4_bytes(n, double_quant=True).total == dq
    assert bnb_q4_bytes(n, double_quant=True).payload == (n + 1) // 2


def _q4_total(allocs: list) -> int:
    named = by_name(allocs)
    total = named["weights.base.q4_payload"].bytes_low + named["weights.base.q4_metadata"].bytes_low
    vision = named.get("weights.vision_tower")
    if vision is not None:
        total += vision.bytes_low - 9_663_968  # minus the vision tower's non-quantized tensors
    return total


@pytest.mark.parametrize(
    ("scope", "double_quant", "expected"),
    [
        ("full_checkpoint", True, 3_802_183_024),
        ("text_only", True, 3_569_313_760),
        ("full_checkpoint", False, 4_145_469_568),
        ("text_only", False, 3_891_674_624),
    ],
)
def test_mimo_4bit_totals(
    mimo: ModelInventory, scope: str, double_quant: bool, expected: int
) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    cfg = make_cfg(scope=scope, double_quant=double_quant)
    assert _q4_total(resident_weight_allocations(st, cfg, LIVE)) == expected
    assert len(quantized_modules(st, cfg)) == (358 if scope == "full_checkpoint" else 248)


def test_mimo_qlora_allocations_are_separate_and_analytic(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    allocs = by_name(resident_weight_allocations(st, make_cfg(), LIVE))
    assert set(allocs) == {
        "weights.base.q4_payload",
        "weights.base.q4_metadata",
        "weights.base.dense",
        "weights.vision_tower",
    }
    # non-quantized text: embed + lm_head + norms + conv1d + A_log/dt_bias in the load dtype
    assert allocs["weights.base.dense"].bytes_high == 2 * 2_035_298_816
    # vision tower: 4-bit payload 225,589,248 + metadata 7,280,016 + 4,831,984 bf16 elements
    assert allocs["weights.vision_tower"].bytes_low == 225_589_248 + 7_280_016 + 2 * 4_831_984
    for a in allocs.values():
        assert a.category is AllocationCategory.WEIGHTS_BASE
        assert a.evidence is Evidence.ANALYTIC and a.bytes_low == a.bytes_high
        assert a.live_at == LIVE
        assert a.formula_ref and a.formula_ref.startswith("methodology-architectures.md#")


def test_text_only_scope_drops_the_vision_tower(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    names = {a.name for a in resident_weight_allocations(st, make_cfg(scope="text_only"), LIVE)}
    assert "weights.vision_tower" not in names


def test_fp32_load_doubles_non_quantized_modules(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    allocs = by_name(resident_weight_allocations(st, make_cfg(load="float32"), LIVE))
    assert allocs["weights.base.dense"].bytes_low == 4 * 2_035_298_816
    assert allocs["weights.base.q4_payload"].bytes_low == 3_459_252_224  # 4-bit is dtype-free


def test_quantization_skip_patterns_keep_modules_dense(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    cfg = make_cfg(skip=["lm_head", "model.visual"])
    qmods = quantized_modules(st, cfg)
    assert len(qmods) == 248 and "lm_head" not in qmods
    vision = by_name(resident_weight_allocations(st, cfg, LIVE))["weights.vision_tower"]
    assert vision.bytes_low == 2 * 456_010_480  # whole tower stays bf16


def test_default_skip_applies_with_an_empty_pattern_list(mimo: ModelInventory) -> None:
    # no llm_int8_skip_modules: transformers' defaults (output embedding + tied modules)
    st = ModelStructure(mimo, "qwen3_5")
    qmods = quantized_modules(st, make_cfg(skip=[]))
    assert len(qmods) == 358 and "lm_head" not in qmods


def test_a_skip_list_replaces_the_defaults(mimo: ModelInventory) -> None:
    # The resolved list is exported as llm_int8_skip_modules, which REPLACES transformers'
    # defaults (add_default_skips=False, LQ §2.2): without lm_head the head would be Linear4bit,
    # which no LM-head/loss ledger models -> refused instead of silently estimated.
    st = ModelStructure(mimo, "qwen3_5")
    with pytest.raises(EstimatorError) as err:
        quantized_modules(st, make_cfg(skip=["model.visual"]))
    assert err.value.issue.code is ErrorCode.UNSUPPORTED_BACKEND_COMBINATION
    assert err.value.issue.details["output_embedding"] == "lm_head"
    target = ["model.language_model.layers.3.self_attn.o_proj"]
    with pytest.raises(EstimatorError):  # the resolver turns this into a blocker
        trainable_group_list(st, make_cfg(targets=target, skip=["model.visual"]))


def test_lora_without_quantization_keeps_everything_in_load_dtype(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    allocs = by_name(resident_weight_allocations(st, make_cfg(strategy=Strategy.LORA), LIVE))
    assert set(allocs) == {"weights.base.dense", "weights.vision_tower"}
    assert allocs["weights.base.dense"].bytes_low == 2 * 8_953_803_264
    assert allocs["weights.vision_tower"].bytes_low == 2 * 456_010_480
    total = sum(a.bytes_low for a in allocs.values())
    assert total == mimo.bytes_serialized_total == 18_819_627_488


def test_tied_weights_are_resident_once(ib: ModuleType) -> None:
    cfg_dict = dict(ib.TINY_DENSE, model_type="llama", architectures=["LlamaForCausalLM"])
    cfg_dict["tie_word_embeddings"] = True
    rows = [*ib.dense_rows(cfg_dict), ("lm_head.weight", "bfloat16", [1000, 96])]
    inv = ib.inventory_from_tensors(cfg_dict, rows)
    st = ModelStructure(inv, "dense")
    dense = by_name(resident_weight_allocations(st, make_cfg(strategy=Strategy.LORA), LIVE))
    assert dense["weights.base.dense"].bytes_low == 2 * inv.params_total
    assert (
        inv.params_total == sum(r[2][0] * (r[2][1] if len(r[2]) > 1 else 1) for r in rows) - 96_000
    )


def test_upcast_patterns_store_fp32(ib: ModuleType) -> None:
    st = ModelStructure(ib.tiny_q35_inventory(), "qwen3_5")
    base = by_name(resident_weight_allocations(st, make_cfg(strategy=Strategy.LORA), LIVE))
    up = by_name(
        resident_weight_allocations(
            st, make_cfg(strategy=Strategy.LORA, upcast=["input_layernorm"]), LIVE
        )
    )
    assert up["weights.base.dense"].bytes_low - base["weights.base.dense"].bytes_low == 2 * 2 * 96


# ---------------------------------------------------------------- load phase (§3.5)


def test_quantized_load_transient_bounds(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    bf16 = load_transient_allocations(st, make_cfg(), ["L"])
    assert len(bf16) == 1 and bf16[0].category is AllocationCategory.LOAD_TRANSIENT
    # gate_proj 12288x4096 in bf16 + double-quant fp32 temporaries (8·nb)
    assert bf16[0].bytes_low == bf16[0].bytes_high == 106_954_752
    fp32 = load_transient_allocations(st, make_cfg(load="float32"), ["L"])[0]
    assert fp32.bytes_low == 207_618_048
    # device-side bf16 -> fp32 conversion would add the embed_tokens source copy (INFERRED)
    assert fp32.bytes_high == 2_034_237_440
    assert fp32.evidence is Evidence.ASSUMPTION
    no_dq = load_transient_allocations(st, make_cfg(double_quant=False), ["L"])[0]
    assert no_dq.bytes_low == 100_663_296


def test_unquantized_load_transient(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    assert load_transient_allocations(st, make_cfg(strategy=Strategy.LORA), ["L"]) == []
    conv = load_transient_allocations(st, make_cfg(strategy=Strategy.LORA, load="float32"), ["L"])
    assert len(conv) == 1 and conv[0].bytes_low == 0
    assert conv[0].bytes_high == 2 * 2_034_237_440 + 2 * 100_663_296  # 4 async workers


def test_device_map_budget_bytes(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    assert device_map_load_bytes(st, make_cfg()) == 7_765_103_072
    assert device_map_load_bytes(st, make_cfg(load="float32")) == 11_845_364_672
