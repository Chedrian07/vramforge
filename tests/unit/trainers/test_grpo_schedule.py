"""GRPO schedule (TRL 1.14.1): rollout, conditional log-prob passes, reward scope (plan §9.7)."""

from __future__ import annotations

import pytest
from vf_fakes import (
    FakeArch,
    V,
    by_name,
    grpo_shape,
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
    ErrorCode,
    Objective,
    RewardKind,
    ScopeConfig,
    Strategy,
)
from vramforge_estimator.trainers import get_trainer
from vramforge_estimator.trainers.grpo import grpo_flags, rollout_buffer_bytes

INV = make_inventory()
PREFILL, DECODE = "ROLLOUT_PREFILL_AND_DECODE:prefill", "ROLLOUT_PREFILL_AND_DECODE:decode"
P = "POLICY_FORWARD_BACKWARD"
OLD, REFLP = f"{P}:old_logprob", f"{P}:reference_logprob"
FWD, LOSS, LOSS_BWD, BWD = (f"{P}:{n}" for n in ("forward", "loss", "loss_backward", "backward"))
STEP = "OPTIMIZER_STEP:step"
REWARD = "REWARD:score"
LOAD = [
    "MODEL_LOAD_AND_QUANTIZE:policy_device_map_check",
    "MODEL_LOAD_AND_QUANTIZE:policy_load_peak",
]


def cfg_for(strategy: Strategy = Strategy.QLORA, **kw):
    kw.setdefault("accumulation", 4)
    return make_cfg(Objective.GRPO, strategy, inventory=INV, **kw)


def build(cfg, shape=None, arch=None):
    shape = shape or grpo_shape(prompt=50, budget=100, rows=cfg.microbatch)
    return get_trainer(Objective.GRPO).build_schedule(
        INV, cfg, arch or FakeArch(), shape, make_plan(cfg, [shape]), ScopeConfig()
    )


def test_plan_example_has_no_logprob_passes_and_conditional_reward() -> None:
    cfg = cfg_for()  # G=4, gbs=4, B_update=1, K=4 -> spg=4, C=4
    assert cfg.grpo.live_sequences == 4 and cfg.grpo.steps_per_generation == 4
    assert grpo_flags(cfg.grpo) == (False, False, False)
    sched = build(cfg)
    assert tp_ids(sched) == [*LOAD, PREFILL, DECODE, FWD, LOSS, LOSS_BWD, BWD, STEP]
    reward = next(e for e in sched.excluded if e.name == "REWARD")
    assert reward.code is ErrorCode.GRPO_REWARD_UNSPECIFIED
    kv = by_name(sched, "rollout.kv_final")
    assert kv.live_at == [DECODE] and kv.bytes_high == 2 * 4 * (50 + 100 - 1)
    # aligned settings: no gradients during the rollout, optimizer state (steady state) yes
    assert PREFILL not in by_name(sched, "grad.lora").live_at
    assert PREFILL in by_name(sched, "optimizer.lora").live_at


def test_policy_logits_follow_the_fused_kernel_factors() -> None:
    sched = build(cfg_for())
    e = 1 * (100 + 1) * V  # B_update x (L+1) x V
    assert by_name(sched, "logits.policy.lm_head_output").bytes_high == 2 * e
    assert by_name(sched, "logits.policy.fp32").bytes_high == 4 * e
    assert by_name(sched, "logits.policy.fp32").live_at == [FWD, LOSS, LOSS_BWD]
    assert by_name(sched, "loss.grpo.grad_logits").bytes_high == 4 * 1 * 100 * V
    saved = by_name(sched, "policy.saved")
    assert saved.bytes_high == 10 * 1 * 150  # B_update rows of P + L tokens
    assert LOSS_BWD in saved.live_at


def test_beta_adds_a_reference_pass_peft_without_weights() -> None:
    sched = build(cfg_for(beta=0.04))
    assert REFLP in tp_ids(sched)
    assert not any(a.category is AllocationCategory.WEIGHTS_OTHER_MODELS for a in sched.allocations)
    chunks = 4  # C / B_update
    assert chunks >= 2
    assert by_name(sched, "logits.reference_logprob").bytes_high == 10 * 101 * V
    assert by_name(sched, "reference_logprob.no_grad").live_at == [REFLP]


def test_beta_with_full_finetuning_loads_a_reference_model() -> None:
    sched = build(cfg_for(Strategy.FULL, beta=0.04))
    ids = tp_ids(sched)
    assert "MODEL_LOAD_AND_QUANTIZE:reference_load_peak" in ids
    ref_weights = by_name(sched, "reference.weights.base")
    assert ref_weights.category is AllocationCategory.WEIGHTS_OTHER_MODELS
    assert PREFILL in ref_weights.live_at and STEP in ref_weights.live_at


@pytest.mark.parametrize(
    ("kw", "expect_old", "expect_grads_at_rollout"),
    [
        ({"num_iterations": 2}, True, False),  # K=4 % (4x2) != 0
        ({"generation_batch_size": 8}, True, False),  # spg=8: K=4 % 8 != 0
        ({"generation_batch_size": 2, "num_generations": 2}, False, True),  # spg=2: 2 % 4 != 0
    ],
)
def test_old_logprob_and_rollout_gradient_conditions(kw, expect_old, expect_grads_at_rollout):
    cfg = cfg_for(**kw)
    old, ref, grads = grpo_flags(cfg.grpo)
    assert (old, ref, grads) == (expect_old, False, expect_grads_at_rollout)
    sched = build(cfg)
    assert (OLD in tp_ids(sched)) is expect_old
    assert (PREFILL in by_name(sched, "grad.lora").live_at) is expect_grads_at_rollout


def test_single_chunk_logprob_pass_uses_six_bytes_per_element() -> None:
    cfg = cfg_for(
        microbatch=2, generation_batch_size=2, num_generations=2, num_iterations=2, accumulation=1
    )
    assert cfg.grpo.steps_per_generation == 1 and cfg.grpo.live_sequences == 2
    sched = build(cfg)
    assert by_name(sched, "logits.old_logprob").bytes_high == 6 * 2 * 101 * V


def test_rollout_buffers_bridge_generations() -> None:
    sched = build(cfg_for())
    expected = rollout_buffer_bytes(4, 50, 100, 0)
    assert expected == 4 * (50 * 16 + 100 * 16) + 16
    assert by_name(sched, "rollout_buffers.previous").live_at == [PREFILL, DECODE]
    assert by_name(sched, "rollout_buffers.current").live_at == [FWD, LOSS, LOSS_BWD, BWD, STEP]
    assert by_name(sched, "rollout_buffers.current").bytes_high == expected


def test_local_reward_model_on_gpu_is_unknown_until_inspected() -> None:
    sched = build(cfg_for(reward=RewardKind.LOCAL_MODEL))
    assert REWARD in tp_ids(sched)
    assert not any(e.name == "REWARD" for e in sched.excluded)
    est = evaluate(sched)
    assert est.scenario_high_bytes is None
    assert {"reward_model.weights", "reward_model.forward"} <= {
        u.name for u in est.unknown_components
    }


@pytest.mark.parametrize("kind", [RewardKind.CPU_RULE, RewardKind.REMOTE])
def test_off_gpu_rewards_are_excluded(kind: RewardKind) -> None:
    sched = build(cfg_for(reward=kind))
    reward = next(e for e in sched.excluded if e.name == "REWARD")
    assert reward.code is None
    assert REWARD not in tp_ids(sched)


def test_local_reward_on_another_device_is_excluded() -> None:
    sched = build(cfg_for(reward=RewardKind.LOCAL_MODEL, reward_on_gpu=False))
    assert any(e.name == "REWARD" for e in sched.excluded)
    assert "reward_model.weights" not in names(sched)


def test_reward_placement_comes_from_the_typed_field_not_the_audit_trail() -> None:
    off_gpu = ConfigResolution(
        field="grpo.reward.on_training_gpu", requested=False, resolved=False, reason="t"
    )
    sched = build(cfg_for(reward=RewardKind.LOCAL_MODEL, resolutions=[off_gpu]))
    assert REWARD in tp_ids(sched)  # the typed field (on the GPU) decides
    assert "reward_model.weights" in names(sched)


def test_accumulation_never_multiplies_grpo_activations() -> None:
    one = build(cfg_for(accumulation=1, generation_batch_size=4))
    many = build(cfg_for(accumulation=16, generation_batch_size=4))

    def view(sched):
        return sorted(
            (a.name, a.bytes_high)
            for a in sched.allocations
            if a.category
            in (AllocationCategory.SAVED_ACTIVATIONS, AllocationCategory.GENERATION_CACHE)
        )

    assert view(one) == view(many)


def test_missing_lengths_are_reported_as_unknown() -> None:
    cfg = cfg_for()
    shape = grpo_shape().model_copy(update={"prompt_length": None, "completion_length": None})
    sched = build(cfg, shape)
    est = evaluate(sched)
    assert est.scenario_high_bytes is None
    assert "grpo.shape" in {u.name for u in est.unknown_components}
