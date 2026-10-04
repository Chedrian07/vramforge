"""Gradients and optimizer state rules shared by every trainer (plan §9.4)."""

from __future__ import annotations

import math

from vf_fakes import (
    FakeArch,
    all_targets,
    by_name,
    lora_numel,
    make_cfg,
    make_inventory,
    make_plan,
    names,
    sft_shape,
    text_targets,
)

from vramforge_estimator.architectures import TrainableGroup
from vramforge_estimator.memory import evaluate
from vramforge_estimator.schemas import (
    AllocationCategory,
    Evidence,
    Objective,
    ScopeConfig,
    Strategy,
)
from vramforge_estimator.trainers import get_trainer
from vramforge_estimator.trainers.trainable import executed_slices, lora_rank

STEP = "OPTIMIZER_STEP:step"


def build(cfg, inventory, arch=None):
    shape = sft_shape(1, 64)
    return get_trainer(Objective.SFT).build_schedule(
        inventory, cfg, arch or FakeArch(), shape, make_plan(cfg, [shape]), ScopeConfig()
    )


def test_qlora_adamw_states_follow_bf16_adapter_dtype() -> None:
    inv = make_inventory()
    cfg = make_cfg(inventory=inv)  # QLoRA: TRL casts trainable params to bf16
    sched = build(cfg, inv)
    n = lora_numel(inv, text_targets(inv))
    assert by_name(sched, "adapter.lora").bytes_high == 2 * n
    assert by_name(sched, "grad.lora").bytes_high == 2 * n
    assert by_name(sched, "optimizer.lora").bytes_high == 2 * 2 * n  # exp_avg + exp_avg_sq
    foreach = by_name(sched, "optimizer.lora.foreach_sqrt")
    assert foreach.live_at == [STEP] and foreach.bytes_high == 2 * n
    assert foreach.category is AllocationCategory.WORKSPACE


def test_lora_adapter_is_fp32_without_quantization() -> None:
    inv = make_inventory()
    sched = build(make_cfg(Objective.SFT, Strategy.LORA, inventory=inv), inv)
    n = lora_numel(inv, text_targets(inv))
    assert by_name(sched, "grad.lora").bytes_high == 4 * n
    assert by_name(sched, "optimizer.lora").bytes_high == 2 * 4 * n


def test_fused_adamw_keeps_device_step_scalars_and_no_foreach_temporaries() -> None:
    inv = make_inventory()
    sched = build(make_cfg(inventory=inv, optimizer="adamw_torch_fused"), inv)
    n = lora_numel(inv, text_targets(inv))
    tensors = 2 * len(text_targets(inv))
    assert by_name(sched, "optimizer.lora").bytes_high == 4 * n + 512 * tensors
    assert "optimizer.lora.foreach_sqrt" not in names(sched)


def test_eight_bit_states_per_tensor_rule() -> None:
    inv = make_inventory()
    sched = build(make_cfg(inventory=inv, optimizer="adamw_8bit"), inv)
    expected = 2 * 256 * 4  # shared qmaps
    by = {m.name: m for m in inv.linear_modules}
    for name in text_targets(inv):
        for n in (16 * by[name].in_features, by[name].out_features * 16):
            expected += 8 * n if n < 4096 else 2 * n + 8 * math.ceil(n / 256)
    states = by_name(sched, "optimizer.lora")
    assert states.bytes_low == states.bytes_high == expected
    assert "optimizer.lora.foreach_sqrt" not in names(sched)


def test_vision_lora_keeps_weights_but_gets_no_gradient_or_state() -> None:
    inv = make_inventory(vision=True)
    cfg = make_cfg(inventory=inv, targets=all_targets(inv))
    n_all = lora_numel(inv, all_targets(inv))
    n_text = lora_numel(inv, text_targets(inv))
    sched = build(cfg, inv)
    assert by_name(sched, "adapter.lora").bytes_high == 2 * n_all
    assert by_name(sched, "grad.lora").bytes_high == 2 * n_text
    assert by_name(sched, "optimizer.lora").bytes_high == 4 * n_text
    assert any("실행되지 않는" in a.text for a in sched.assumptions)

    split = build(cfg, inv, FakeArch(vision_lora_group=True))
    assert by_name(split, "grad.lora:text").bytes_high == 2 * n_text
    assert "grad.lora:vision" not in names(split)
    assert by_name(split, "adapter.lora:vision").bytes_high == 2 * (n_all - n_text)


def test_full_finetune_vision_tower_gets_no_gradient() -> None:
    inv = make_inventory(vision=True)
    cfg = make_cfg(Objective.SFT, Strategy.FULL, inventory=inv)
    sched = build(cfg, inv)
    text = sum(t.numel for t in inv.tensors if t.component.value == "text")
    assert by_name(sched, "grad.full").bytes_high == 2 * text
    assert by_name(sched, "optimizer.full").bytes_high == 2 * 2 * text
    assert not any(a.category is AllocationCategory.WEIGHTS_ADAPTER for a in sched.allocations)


def test_full_finetune_eight_bit_keeps_embeddings_32bit() -> None:
    inv = make_inventory()
    cfg = make_cfg(Objective.SFT, Strategy.FULL, inventory=inv, optimizer="paged_adamw_8bit")
    sched = build(cfg, inv)
    expected = 2 * 256 * 4
    for t in inv.tensors:
        if t.role.value == "embedding" or t.numel < 4096:
            expected += 8 * t.numel
        else:
            expected += 2 * t.numel + 8 * math.ceil(t.numel / 256)
    states = by_name(sched, "optimizer.full")
    assert states.bytes_high == expected
    assert "paged" in (states.note or "")


def test_unmappable_group_is_counted_conservatively() -> None:
    inv = make_inventory()
    cfg = make_cfg(inventory=inv)
    odd = TrainableGroup("lora", "lora", 12345, "bfloat16", 3)
    (s,) = executed_slices([odd], inv, cfg)
    assert s.executed_numel == 12345 and not s.exact and s.tensors is None


def test_rank_pattern_matches_like_peft() -> None:
    inv = make_inventory()
    cfg = make_cfg(inventory=inv)
    cfg = cfg.model_copy(
        update={"lora": cfg.lora.model_copy(update={"rank_pattern": {"q_proj": 4}})}
    )
    assert lora_rank("model.layers.0.self_attn.q_proj", cfg) == 4
    assert lora_rank("model.layers.0.self_attn.k_proj", cfg) == 16


def test_workspace_assumptions_are_assumption_evidence() -> None:
    inv = make_inventory()
    sched = build(make_cfg(inventory=inv), inv)
    ctx = by_name(sched, "cuda_context")
    assert (ctx.bytes_low, ctx.bytes_high) == (300, 1000)
    assert ctx.evidence is Evidence.ASSUMPTION
    assert ctx.category is AllocationCategory.NON_FRAMEWORK
    assert len(ctx.live_at) == len(sched.timepoints)
    lib = by_name(sched, "library_workspace")
    assert all(not t.startswith("MODEL_LOAD") for t in lib.live_at)
    est = evaluate(sched)
    for total in est.timepoints:
        if total.timepoint.endswith("device_map_check"):
            continue
        slack = by_name(sched, f"allocator_slack@{total.timepoint}")
        base_high = total.bytes_high - slack.bytes_high - ctx.bytes_high
        assert slack.bytes_high == math.ceil(base_high * 0.15)
