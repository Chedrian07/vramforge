"""DPO schedule (TRL 1.14.1): reference strategies and fp32 logits coexistence (plan §9.6)."""

from __future__ import annotations

from vf_fakes import (
    WEIGHTS,
    FakeArch,
    H,
    V,
    by_name,
    dpo_shape,
    make_cfg,
    make_inventory,
    make_plan,
    names,
    tp_ids,
)

from vramforge_estimator.memory import evaluate
from vramforge_estimator.schemas import (
    AllocationCategory,
    ConfigResolution,
    Objective,
    ReferenceStrategy,
    ScopeConfig,
    Strategy,
)
from vramforge_estimator.trainers import get_trainer

INV = make_inventory()
P = "POLICY_FORWARD_BACKWARD"
FWD, REF, LOSS, LOSS_BWD, BWD = (
    f"{P}:{n}" for n in ("forward", "reference_forward", "loss", "loss_backward", "backward")
)
STEP = "OPTIMIZER_STEP:step"
PRE = "REFERENCE_PRECOMPUTE:forward"
POLICY_LOAD = [
    "MODEL_LOAD_AND_QUANTIZE:policy_device_map_check",
    "MODEL_LOAD_AND_QUANTIZE:policy_load_peak",
]
REF_LOAD = [
    "MODEL_LOAD_AND_QUANTIZE:reference_device_map_check",
    "MODEL_LOAD_AND_QUANTIZE:reference_load_peak",
]


def build(cfg, shape=None):
    shape = shape or dpo_shape(pairs=1, length=100)
    return get_trainer(Objective.DPO).build_schedule(
        INV, cfg, FakeArch(), shape, make_plan(cfg, [shape]), ScopeConfig()
    )


def cfg_for(reference: ReferenceStrategy, strategy: Strategy = Strategy.QLORA, **kw):
    return make_cfg(Objective.DPO, strategy, inventory=INV, reference=reference, **kw)


def test_frozen_base_switch_runs_reference_forward_without_extra_weights() -> None:
    sched = build(cfg_for(ReferenceStrategy.FROZEN_BASE_SWITCH))
    assert tp_ids(sched) == [*POLICY_LOAD, FWD, REF, LOSS, LOSS_BWD, BWD, STEP]
    assert not any(a.category is AllocationCategory.WEIGHTS_OTHER_MODELS for a in sched.allocations)
    assert by_name(sched, "reference_forward.no_grad").live_at == [REF]
    # policy graph (saved activations) stays alive through the reference forward
    assert REF in by_name(sched, "policy.saved").live_at


def test_policy_and_reference_fp32_logits_coexist_at_the_loss() -> None:
    sched = build(cfg_for(ReferenceStrategy.FROZEN_BASE_SWITCH), dpo_shape(pairs=2, length=100))
    nv = 4 * 100 * V  # 2B rows x T x V
    policy = by_name(sched, "logits.policy.fp32")
    reference = by_name(sched, "logits.reference.fp32")
    assert policy.bytes_high == reference.bytes_high == 4 * nv
    assert policy.live_at == [FWD, REF, LOSS, LOSS_BWD]
    assert reference.live_at == [REF, LOSS]
    assert by_name(sched, "logits.policy.lm_head_output").bytes_high == 2 * nv
    assert by_name(sched, "loss.dpo.grad_logits").bytes_high == 4 * 4 * 99 * V
    assert by_name(sched, "loss.dpo.grad_logits").live_at == [LOSS_BWD]
    est = evaluate(sched)
    # The reference forward momentarily holds policy fp32 + reference bf16 output + its fp32 copy.
    assert est.peak_timepoint == REF
    items = {i.name for i in est.peak_breakdown.items}
    assert {
        "logits.policy.fp32",
        "logits.reference.fp32",
        "logits.reference.lm_head_output",
        "policy.saved",
    } <= items
    # reference logits are gone before the backward; the kernel's grad buffer takes their place
    at_loss_bwd = {a.name for a in sched.allocations if LOSS_BWD in a.live_at}
    assert "logits.reference.fp32" not in at_loss_bwd and "loss.dpo.grad_logits" in at_loss_bwd


def test_standalone_reference_is_a_second_resident_model() -> None:
    sched = build(cfg_for(ReferenceStrategy.STANDALONE_MODEL, Strategy.FULL))
    assert tp_ids(sched)[:4] == [*POLICY_LOAD, *REF_LOAD]
    ref_weights = by_name(sched, "reference.weights.base")
    assert ref_weights.category is AllocationCategory.WEIGHTS_OTHER_MODELS
    assert ref_weights.live_at == [REF_LOAD[1], FWD, REF, LOSS, LOSS_BWD, BWD, STEP]
    assert by_name(sched, "reference.weights.quantize_transient").live_at == [REF_LOAD[1]]
    assert by_name(sched, "reference.device_map_budget").live_at == [REF_LOAD[0]]
    # the policy is already resident while the reference loads
    assert REF_LOAD[0] in by_name(sched, "weights.base").live_at
    est = evaluate(sched)
    ref_check = next(t for t in est.timepoints if t.timepoint == REF_LOAD[0])
    assert ref_check.known_floor_bytes >= WEIGHTS
    assert "lm_head.input" in names(sched)  # full fine-tuning trains lm_head
    assert set(by_name(sched, "lm_head.input").live_at) == {FWD, REF, LOSS, LOSS_BWD}


def test_lora_adapter_exists_while_the_standalone_reference_loads() -> None:
    sched = build(cfg_for(ReferenceStrategy.STANDALONE_MODEL, Strategy.LORA))
    assert REF_LOAD[0] in by_name(sched, "adapter.lora").live_at


def test_precompute_runs_before_training_without_optimizer_state() -> None:
    cfg = cfg_for(ReferenceStrategy.PRECOMPUTED_LOG_PROBS, precompute_batch_size=4)
    sched = build(cfg)
    assert tp_ids(sched) == [*POLICY_LOAD, PRE, FWD, LOSS, LOSS_BWD, BWD, STEP]
    assert "logits.reference.fp32" not in names(sched)
    pre_logits = by_name(sched, "logits.precompute")
    assert pre_logits.bytes_high == 2 * 4 * 100 * V * 2  # 2 x B_pre rows, model dtype (bf16)
    assert by_name(sched, "precompute.no_grad").live_at == [PRE]
    assert PRE not in by_name(sched, "optimizer.lora").live_at
    assert PRE not in by_name(sched, "grad.lora").live_at
    assert PRE in by_name(sched, "adapter.lora").live_at


def test_precompute_batch_defaults_to_the_microbatch() -> None:
    sched = build(cfg_for(ReferenceStrategy.PRECOMPUTED_LOG_PROBS, microbatch=3))
    assert by_name(sched, "logits.precompute").bytes_high == 2 * 3 * 100 * V * 2


def test_separate_reference_checkpoint_is_unknown() -> None:
    cfg = cfg_for(ReferenceStrategy.STANDALONE_MODEL, Strategy.FULL, reference_model="org/ref")
    sched = build(cfg)
    est = evaluate(sched)
    assert est.scenario_high_bytes is None
    assert {"reference.weights", "reference.device_map_budget"} <= {
        u.name for u in est.unknown_components
    }


def test_reference_identity_comes_from_the_typed_field_not_the_audit_trail() -> None:
    # A stale ConfigResolution entry alone must not turn the reference into another checkpoint.
    audit_only = cfg_for(
        ReferenceStrategy.STANDALONE_MODEL,
        Strategy.FULL,
        resolutions=[
            ConfigResolution(
                field="dpo.reference_model", requested="org/ref", resolved="org/ref", reason="t"
            )
        ],
    )
    sched = build(audit_only)
    assert by_name(sched, "reference.weights.base").bytes_high == WEIGHTS
    assert evaluate(sched).scenario_high_bytes is not None


def test_frozen_lm_head_saves_no_input() -> None:
    sched = build(cfg_for(ReferenceStrategy.FROZEN_BASE_SWITCH))
    assert "lm_head.input" not in names(sched)
    est = evaluate(sched)
    assert est.scenario_high_bytes is not None and H > 0
