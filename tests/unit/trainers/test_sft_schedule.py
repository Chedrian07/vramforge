"""SFT schedule (TRL 1.14.1): structure, lifetimes and LM-head/loss buffers."""

from __future__ import annotations

import math

from vf_fakes import (
    WEIGHTS,
    FakeArch,
    H,
    V,
    by_name,
    make_cfg,
    make_inventory,
    make_plan,
    names,
    sft_shape,
    tp_ids,
)

from vramforge_estimator.memory import evaluate
from vramforge_estimator.schemas import (
    AllocationCategory,
    Objective,
    Phase,
    ScopeConfig,
    Strategy,
)
from vramforge_estimator.trainers import get_trainer

INV = make_inventory()
FWD, LOSS, LOSS_BWD, BWD = (
    f"POLICY_FORWARD_BACKWARD:{n}" for n in ("forward", "loss", "loss_backward", "backward")
)
STEP = "OPTIMIZER_STEP:step"
LOAD_CHECK = "MODEL_LOAD_AND_QUANTIZE:policy_device_map_check"
LOAD_PEAK = "MODEL_LOAD_AND_QUANTIZE:policy_load_peak"


def build(cfg=None, shape=None, scope=None, arch=None, inventory=INV):
    cfg = cfg or make_cfg(inventory=inventory)
    shape = shape or sft_shape(rows=1, length=100)
    return get_trainer(Objective.SFT).build_schedule(
        inventory, cfg, arch or FakeArch(), shape, make_plan(cfg, [shape]), scope or ScopeConfig()
    )


def test_timeline_and_excluded_scope() -> None:
    sched = build()
    assert tp_ids(sched) == [LOAD_CHECK, LOAD_PEAK, FWD, LOSS, LOSS_BWD, BWD, STEP]
    assert {e.name for e in sched.excluded} == {"EVALUATION", "CHECKPOINT_SAVE_OR_CONSOLIDATE"}
    assert get_trainer(Objective.SFT).trainer_id == "trl-1.14.1-sft"


def test_chunked_nll_has_no_full_logits_and_is_length_independent() -> None:
    short = build(shape=sft_shape(1, 100))
    long = build(shape=sft_shape(1, 4000))
    cv = 256 * V
    for sched in (short, long):
        fwd = by_name(sched, "loss.chunked_nll.chunk_forward")
        bwd = by_name(sched, "loss.chunked_nll.chunk_backward")
        assert (fwd.bytes_low, fwd.bytes_high) == (15 * cv, math.ceil(16.7 * cv))
        assert (bwd.bytes_low, bwd.bytes_high) == (16 * cv, 18 * cv)
        assert fwd.live_at == [LOSS] and bwd.live_at == [LOSS_BWD]
        assert not any(n.startswith("loss.hf_ce") for n in names(sched))
    gather = by_name(long, "loss.chunked_nll.hidden_gather")
    assert gather.bytes_high == 1 * 3999 * H * 2  # B·(T−1)·H·bf16
    assert gather.live_at == [LOSS, LOSS_BWD]
    assert "loss.chunked_nll.reshape_copy" not in names(long)  # B = 1: reshape is a view
    assert "loss.chunked_nll.reshape_copy" in names(build(shape=sft_shape(2, 100)))


def test_standard_loss_materializes_full_fp32_logits() -> None:
    sched = build(make_cfg(loss_path="hf_ce", strategy=Strategy.LORA))
    nv = 100 * V
    fwd = by_name(sched, "loss.hf_ce.forward")
    bwd = by_name(sched, "loss.hf_ce.backward")
    assert (fwd.bytes_low, fwd.bytes_high) == (10 * nv, 12 * nv)
    assert (bwd.bytes_low, bwd.bytes_high) == (12 * nv, 14 * nv)
    assert "lm_head.input" not in names(sched)  # frozen lm_head does not save its input


def test_trainable_lm_head_accumulates_weight_grads_in_chunked_loss() -> None:
    sched = build(make_cfg(strategy=Strategy.FULL))
    acc = by_name(sched, "loss.chunked_nll.lm_head_grad_accumulation")
    assert acc.bytes_low == 3 * V * H * 2
    assert acc.bytes_high == 3 * V * H * 2 + 18 * 256 * V
    assert "loss.chunked_nll.chunk_backward" not in names(sched)


def test_fp32_load_adds_lm_head_weight_cast() -> None:
    sched = build(make_cfg(load_dtype="float32"))
    cast = by_name(sched, "loss.chunked_nll.lm_head_weight_cast")
    assert cast.bytes_high == V * H * 2
    assert by_name(sched, "loss.chunked_nll.hidden_gather").bytes_high == 99 * H * 4


def test_saved_activations_stay_alive_through_the_loss_backward() -> None:
    sched = build()
    saved = by_name(sched, "policy.saved")
    assert set(saved.live_at) == {FWD, LOSS, BWD, LOSS_BWD}
    assert by_name(sched, "policy.recompute").live_at == [BWD]


def test_weights_resident_everywhere_and_load_transient_only_at_peak() -> None:
    sched = build()
    assert by_name(sched, "weights.base").live_at == [LOAD_PEAK, FWD, LOSS, LOSS_BWD, BWD, STEP]
    assert by_name(sched, "weights.quantize_transient").live_at == [LOAD_PEAK]
    budget = by_name(sched, "policy.device_map_budget")
    assert budget.live_at == [LOAD_CHECK]
    assert budget.bytes_high == math.ceil(WEIGHTS / 0.81)  # bnb 4-bit: 0.9 x 0.90
    dense = by_name(build(make_cfg(strategy=Strategy.LORA)), "policy.device_map_budget")
    assert dense.bytes_high == math.ceil(WEIGHTS / 0.9)


def test_device_map_budget_uses_the_adapters_s_load() -> None:
    # transformers sizes 4-bit weights at 0.5 B/param for device_map="auto" (S_load), which is
    # smaller than the resident bytes with quantization metadata (research §3.5, V9).
    s_load = 900_001
    sched = build(arch=FakeArch(load_budget=s_load))
    budget = by_name(sched, "policy.device_map_budget")
    assert budget.bytes_low == budget.bytes_high == math.ceil(s_load * 100 / 81)
    assert budget.note is not None and "S_load" in budget.note
    fallback = by_name(build(), "policy.device_map_budget")
    assert fallback.note is not None and "보수적" in fallback.note


def test_accumulation_never_multiplies_activations() -> None:
    one = build(make_cfg(accumulation=1))
    many = build(make_cfg(accumulation=64))
    acts = {
        AllocationCategory.SAVED_ACTIVATIONS,
        AllocationCategory.RECOMPUTE_WORKING_SET,
        AllocationCategory.LOGITS_AND_LOSS,
    }

    def activation_view(sched):
        return sorted(
            (a.name, a.bytes_low, a.bytes_high, tuple(a.live_at))
            for a in sched.allocations
            if a.category in acts
        )

    assert activation_view(one) == activation_view(many)
    assert by_name(one, "grad.lora").live_at == [BWD, STEP]
    assert by_name(many, "grad.lora").live_at == [FWD, LOSS, LOSS_BWD, BWD, STEP]


def test_optimizer_states_alive_in_every_steady_state_phase() -> None:
    sched = build()
    states = by_name(sched, "optimizer.lora")
    assert states.live_at == [FWD, LOSS, LOSS_BWD, BWD, STEP]
    assert by_name(sched, "adapter.lora").live_at == [FWD, LOSS, LOSS_BWD, BWD, STEP]


def test_scope_inclusion_makes_unknown_phases_explicit() -> None:
    sched = build(scope=ScopeConfig(include_evaluation=True, include_checkpoint_save=True))
    assert "EVALUATION:forward" in tp_ids(sched)
    assert sched.excluded == []
    est = evaluate(sched)
    assert est.scenario_high_bytes is None
    assert {u.name for u in est.unknown_components} == {
        "evaluation.working_set",
        "checkpoint.save_transient",
    }
    # optimizer state persists into evaluation; gradients were released by zero_grad
    assert "EVALUATION:forward" in by_name(sched, "optimizer.lora").live_at
    assert "EVALUATION:forward" not in by_name(sched, "grad.lora").live_at


def test_evaluated_peak_is_the_backward_with_breakdown_equal_to_peak() -> None:
    sched = build(shape=sft_shape(1, 2000))
    est = evaluate(sched)
    assert est.scenario_high_bytes is not None
    assert est.peak_breakdown is not None
    assert sum(i.bytes_high or 0 for i in est.peak_breakdown.items) == est.scenario_high_bytes
    assert est.peak_phase is Phase.POLICY_FORWARD_BACKWARD
    phases = {p.phase: p for p in est.phases}
    assert phases[Phase.MODEL_LOAD_AND_QUANTIZE].included
    assert not phases[Phase.ROLLOUT_PREFILL_AND_DECODE].included


def test_unknown_architecture_activations_propagate() -> None:
    sched = build(arch=FakeArch(unknown_activations=True))
    est = evaluate(sched)
    assert est.scenario_high_bytes is None
    assert "policy.saved" in {u.name for u in est.unknown_components}
    # slack is only sized where the timepoint total is known
    slack = {a.live_at[0] for a in sched.allocations if a.name.startswith("allocator_slack@")}
    assert FWD not in slack and STEP in slack and LOAD_CHECK not in slack
