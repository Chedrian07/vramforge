"""Gradients and optimizer state rules shared by every trainer (plan §9.4)."""

from __future__ import annotations

import math

from vf_fakes import (
    FakeArch,
    H,
    V,
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
from vramforge_estimator.trainers.ledger import alive_by_timepoint, contributions
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

    # Groups are aggregated per (kind, dtype); split or per-shape groups give the same ledger.
    for arch in (FakeArch(vision_lora_group=True), FakeArch(flagged_groups=True)):
        split = build(cfg, inv, arch)
        assert by_name(split, "adapter.lora").bytes_high == 2 * n_all
        assert by_name(split, "grad.lora").bytes_high == 2 * n_text
        assert by_name(split, "optimizer.lora").bytes_high == 4 * n_text
        assert not any(a.name.startswith("adapter.lora:") for a in split.allocations)


def test_flagged_groups_give_exact_eight_bit_states() -> None:
    inv = make_inventory(vision=True)
    cfg = make_cfg(inventory=inv, targets=all_targets(inv), optimizer="adamw_8bit")
    plain = by_name(build(cfg, inv), "optimizer.lora")
    flagged = by_name(build(cfg, inv, FakeArch(flagged_groups=True)), "optimizer.lora")
    assert plain.bytes_low == plain.bytes_high == flagged.bytes_low == flagged.bytes_high
    full = make_cfg(Objective.SFT, Strategy.FULL, inventory=inv, optimizer="adamw_8bit")
    a = by_name(build(full, inv), "optimizer.full")
    b = by_name(build(full, inv, FakeArch(flagged_groups=True)), "optimizer.full")
    assert a.bytes_high == b.bytes_high  # embeddings keep 32-bit state on both paths


def test_modules_to_save_copy_trains_and_saves_lm_head_input() -> None:
    inv = make_inventory()
    cfg = make_cfg(inventory=inv, modules_to_save=["lm_head"])
    sched = build(cfg, inv)
    copy = by_name(sched, "adapter.modules_to_save")
    assert copy.bytes_high == V * H * 2  # TRL casts trainable params of a 4-bit model to bf16
    assert by_name(sched, "grad.modules_to_save").bytes_high == V * H * 2
    # chunked_nll with a trainable lm_head accumulates [V, H] weight gradients
    assert "loss.chunked_nll.lm_head_grad_accumulation" in names(sched)
    assert any(a.id == "chunked_nll_modules_to_save_lm_head" for a in sched.assumptions)
    full = build(make_cfg(Objective.SFT, Strategy.FULL, inventory=inv), inv)
    assert not any(a.id == "chunked_nll_modules_to_save_lm_head" for a in full.assumptions)


def test_trainable_lm_head_input_is_the_final_hidden_storage() -> None:
    # The architecture ledger already reports the final-norm output (= lm_head input) under the
    # "policy.final_hidden" alias; the trainable lm_head saves that same tensor.
    inv = make_inventory()
    cfg = make_cfg(Objective.SFT, Strategy.FULL, inventory=inv, loss_path="hf_ce")
    sched = build(cfg, inv, FakeArch(final_hidden=True))
    head = by_name(sched, "lm_head.input")
    hidden = by_name(sched, "policy.final_hidden")
    assert head.storage_alias_group == hidden.storage_alias_group == "policy.final_hidden"
    loss = "POLICY_FORWARD_BACKWARD:loss"
    alive = alive_by_timepoint(sched.timepoints, sched.allocations)[loss]
    shared = [
        c for c in contributions(alive) if c.spec.storage_alias_group == "policy.final_hidden"
    ]
    assert len(shared) == 1 and shared[0].bytes_high == head.bytes_high == hidden.bytes_high


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
