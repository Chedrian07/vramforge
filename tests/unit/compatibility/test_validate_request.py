"""Request validation (plan §5, §11.3) and the exact TRL 1.14.1 GRPO batch rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vramforge_estimator.compatibility import validate_request
from vramforge_estimator.compatibility.grpo_rules import resolve_grpo_batch
from vramforge_estimator.compatibility.validation import MAX_COMPLETION_BUDGET
from vramforge_estimator.schemas import AnalysisRequest, ErrorCode, Severity

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "requests" / "plan_example_grpo.json"


def request(**changes: Any) -> AnalysisRequest:
    """The plan §15.2 example with dotted-path overrides, e.g. {"training.packing": True}."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for path, value in changes.items():
        node = data
        *parents, leaf = path.split("__")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return AnalysisRequest.model_validate(data)


def codes(issues, severity: Severity = Severity.ERROR) -> set[ErrorCode]:
    return {i.code for i in issues if i.severity is severity}


def test_plan_example_is_valid() -> None:
    assert validate_request(request()) == []


@pytest.mark.parametrize(
    ("strategy", "four_bit"),
    [("qlora", False), ("full", True), ("lora", True)],
)
def test_strategy_and_four_bit_must_agree(strategy: str, four_bit: bool) -> None:
    issues = validate_request(
        request(training__strategy=strategy, training__quantization={"enabled": four_bit})
    )
    assert ErrorCode.CONFLICTING_OPTIONS in codes(issues)


def test_consistent_strategies_pass() -> None:
    for strategy, four_bit in (("qlora", True), ("lora", False), ("full", False)):
        issues = validate_request(
            request(training__strategy=strategy, training__quantization={"enabled": four_bit})
        )
        assert ErrorCode.CONFLICTING_OPTIONS not in codes(issues)


@pytest.mark.parametrize("changes", [{"training__num_devices": 2}, {"hardware__num_gpus": 2}])
def test_multi_gpu_is_rejected_without_a_topology_adapter(changes) -> None:
    assert ErrorCode.UNSUPPORTED_DISTRIBUTED_TOPOLOGY in codes(validate_request(request(**changes)))


@pytest.mark.parametrize(
    "changes",
    [
        {"training__packing": True},
        {"training__offload": {"optimizer": True}},
        {"training__offload": {"activations": True}},
        {"training__compile": True},
        {"training__loss_kernel": "liger"},
        {"training__loss_kernel": "chunked"},  # GRPO: chunked log-prob needs liger
        {"training__attention_backend": "flash_attention_2"},
        {"training__precision": "fp16"},
        {"training__load_dtype": "float16"},
        {"training__quantization": {"enabled": True, "compute_dtype": "float32"}},
        {"training__lora": {"use_dora": True}},
        {"grpo__rollout_backend": "vllm_colocate"},
        {"training__backend_profile": "nope"},
    ],
)
def test_unmodeled_or_unavailable_options_are_unsupported(changes) -> None:
    assert ErrorCode.UNSUPPORTED_BACKEND_COMBINATION in codes(validate_request(request(**changes)))


def test_linear_attention_kernel_request_is_not_effective() -> None:
    issues = validate_request(request(training__linear_attention_kernel="fla"))
    assert codes(issues) == set()
    assert ErrorCode.REQUESTED_OPTION_NOT_EFFECTIVE in codes(issues, Severity.WARNING)


def test_sft_chunked_loss_rejects_lora_on_lm_head() -> None:
    issues = validate_request(
        request(
            training__objective="sft",
            training__lora={"target_modules": ["q_proj", "lm_head"]},
        )
    )
    assert ErrorCode.CONFLICTING_OPTIONS in codes(issues)
    # A modules_to_save copy is not a PEFT tuner layer, so TRL runs chunked_nll with it.
    mts = validate_request(
        request(training__objective="sft", training__lora={"modules_to_save": ["lm_head"]})
    )
    assert codes(mts) == set() and codes(mts, Severity.WARNING) == set()


@pytest.mark.parametrize(
    ("dpo", "strategy", "four_bit", "code"),
    [
        (
            {"reference_strategy": "precomputed_log_probs", "sync_ref_model": True},
            "full",
            False,
            ErrorCode.CONFLICTING_OPTIONS,
        ),
        ({"sync_ref_model": True}, "lora", False, ErrorCode.CONFLICTING_OPTIONS),
        (
            {"reference_strategy": "frozen_base_switch"},
            "full",
            False,
            ErrorCode.CONFLICTING_OPTIONS,
        ),
        ({"loss_type": "sigmoid,sft"}, "qlora", True, ErrorCode.UNSUPPORTED_BACKEND_COMBINATION),
    ],
)
def test_dpo_conflicts(dpo, strategy, four_bit, code) -> None:
    issues = validate_request(
        request(
            training__objective="dpo",
            training__strategy=strategy,
            training__quantization={"enabled": four_bit},
            dpo=dpo,
        )
    )
    assert code in codes(issues)


def test_dpo_sync_with_full_finetuning_is_allowed() -> None:
    issues = validate_request(
        request(
            training__objective="dpo",
            training__strategy="full",
            training__quantization={"enabled": False},
            dpo={"sync_ref_model": True},
        )
    )
    assert codes(issues) == set()


@pytest.mark.parametrize(
    ("changes", "warned"),
    [
        ({"training__objective": "dpo", "training__lora": {"bias": "all"}}, True),
        (
            {
                "training__objective": "dpo",
                "training__lora": {"bias": "lora_only"},
                "dpo": {"reference_strategy": "frozen_base_switch"},
            },
            True,
        ),
        ({"training__objective": "dpo", "training__lora": {"bias": "none"}}, False),
        (
            {
                "training__objective": "dpo",
                "training__lora": {"bias": "all"},
                "dpo": {"reference_strategy": "precomputed_log_probs"},
            },
            False,  # precomputed in __init__, before any bias is trained
        ),
        ({"grpo__beta": 0.04, "training__lora": {"bias": "all"}}, True),
        ({"grpo__beta": 0.0, "training__lora": {"bias": "all"}}, False),
    ],
)
def test_adapter_off_reference_with_trained_biases_is_flagged(changes, warned) -> None:
    # plan §5.3: disable_adapter() keeps trained base biases, so the reference is not the base.
    issues = validate_request(request(**changes))
    flagged = [i for i in issues if i.affected_component == "training.lora.bias"]
    assert bool(flagged) is warned
    assert all(i.severity is Severity.WARNING for i in flagged)
    assert codes(issues) == set()


def test_local_reward_model_needs_a_reference() -> None:
    issues = validate_request(request(grpo__reward={"kind": "local_model"}))
    assert ErrorCode.INVALID_REQUEST in codes(issues)
    ok = validate_request(
        request(grpo__reward={"kind": "local_model", "model_reference": "org/rm"})
    )
    assert codes(ok) == set()


def test_grpo_batch_constraints_use_trl_rules() -> None:
    bad = validate_request(
        request(grpo__generation_batch_size=6, training__microbatch_per_device=4)
    )
    assert ErrorCode.TRAINER_BATCH_CONSTRAINT in codes(bad)
    both = validate_request(request(grpo__steps_per_generation=4))  # gbs=4 is set in the example
    assert ErrorCode.TRAINER_BATCH_CONSTRAINT in codes(both)
    one = validate_request(request(grpo__num_generations=1, grpo__generation_batch_size=None))
    assert ErrorCode.TRAINER_BATCH_CONSTRAINT in codes(one)


def test_empty_budget_candidates_are_rejected() -> None:
    # The schema requires at least one candidate (min_length=1): rejected before validation.
    with pytest.raises(ValidationError):
        request(grpo__completion_budget_candidates=[])


@pytest.mark.parametrize("bad", [0, -5, MAX_COMPLETION_BUDGET + 1])
def test_budget_candidates_must_be_positive_and_bounded(bad: int) -> None:
    # Candidate items carry the same bound as completion_budget in the schema, so an out-of-range
    # candidate is a 422 before validate_request runs (never zero/negative logits buffers).
    with pytest.raises(ValidationError):
        request(grpo__completion_budget_candidates=[bad, 1024])


def test_candidate_bound_matches_the_schema_bound() -> None:
    from vramforge_estimator.schemas import GrpoConfig

    le = [m.le for m in GrpoConfig.model_fields["completion_budget"].metadata if hasattr(m, "le")]
    assert le == [MAX_COMPLETION_BUDGET]


@pytest.mark.parametrize(
    "hardware",
    [
        {"mode": "gpu_preset"},
        {"mode": "gpu_preset", "gpu_preset": "unknown-gpu"},
        {"mode": "custom"},
        {"mode": "custom", "device_total_bytes": 10, "usable_bytes": 20},
    ],
)
def test_hardware_inputs(hardware) -> None:
    assert ErrorCode.INVALID_REQUEST in codes(validate_request(request(hardware=hardware)))


def test_known_preset_and_custom_capacity_are_valid() -> None:
    assert (
        validate_request(request(hardware={"mode": "gpu_preset", "gpu_preset": "a100-80gb"})) == []
    )
    assert validate_request(request(hardware={"mode": "custom", "usable_bytes": 10**10})) == []


def test_sample_scan_and_profiling_are_flagged_not_blocked() -> None:
    issues = validate_request(request(dataset__scan_mode="sample", profiling={"enabled": True}))
    assert codes(issues) == set()
    assert {ErrorCode.SCAN_PARTIAL, ErrorCode.GPU_WORKER_UNAVAILABLE} <= codes(
        issues, Severity.WARNING
    )
    assert all(i.user_message for i in issues)


# ---------------------------------------------------------------- TRL GRPOConfig table (W=1)


def batch(g, b, k, gbs=None, spg=None, mu=1):
    return resolve_grpo_batch(
        num_generations=g,
        microbatch=b,
        accumulation=k,
        generation_batch_size=gbs,
        steps_per_generation=spg,
        num_iterations=mu,
    )


def test_plan_example_resolution() -> None:
    # plan §8.3 example: G=4, generation_batch=4, B_update=1, K=4 -> spg=4, C=4, U=1
    res, issues = batch(4, 1, 4, gbs=4)
    assert issues == [] and res is not None
    assert (res.steps_per_generation, res.live_sequences, res.unique_prompts) == (4, 4, 1)
    assert res.generation_batch_size == 4
    assert (res.old_logprob_pass, res.grads_alive_during_rollout) == (False, False)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((4, 1, 4), (4, 4)),  # both unset: spg = K, gbs = B x spg
        ((4, 1, 4, None, 8), (8, 8)),  # spg set: gbs = B x spg (K may differ)
        ((4, 2, 4, 16), (16, 8)),  # gbs set: spg = gbs / B
    ],
)
def test_trl_resolution_rows(args, expected) -> None:
    res, issues = batch(*args)
    assert issues == [] and res is not None
    assert (res.generation_batch_size, res.steps_per_generation) == expected


@pytest.mark.parametrize(
    ("args", "trl_message"),
    [
        ((4, 1, 1), "generation_batch_size (1) must be divisible by num_generations (4)."),
        ((4, 1, 4, 4, 4), "can not be both configured"),
        ((4, 4, 4, 6), "generation_batch_size (6) must be divisible by the global batch size (4)."),
        ((3, 1, 4), "generation_batch_size (4) must be divisible by num_generations (3)."),
        ((1, 1, 4), "GRPO requires at least 2 generations"),
    ],
)
def test_trl_rejections(args, trl_message) -> None:
    res, issues = batch(*args)
    assert res is None and issues
    assert any(trl_message in i.details["trl_message"] for i in issues)
    assert all(i.code is ErrorCode.TRAINER_BATCH_CONSTRAINT for i in issues)


def test_logprob_and_gradient_conditions() -> None:
    old_iter, _ = batch(4, 1, 4, gbs=4, mu=2)  # K=4 % (4x2) != 0
    assert old_iter is not None and old_iter.old_logprob_pass
    mid, _ = batch(2, 1, 4, gbs=2)  # spg=2: rollout mid-window -> gradients alive
    assert mid is not None and mid.grads_alive_during_rollout and not mid.old_logprob_pass
