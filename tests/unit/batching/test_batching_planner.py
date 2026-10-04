"""Batch planner: TRL collator token slots, sampler coverage, GRPO arithmetic (plan §8, §19.1)."""

from __future__ import annotations

import pytest

from vramforge_estimator.batching import plan_batches
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, Objective, Severity

SFT, DPO, GRPO = Objective.SFT, Objective.DPO, Objective.GRPO


# ---------------------------------------------------------------- plan §19.1 fixtures


@pytest.mark.parametrize("pad", [None, 1])
def test_sft_padding_b2_100_300_is_600_slots(make_resolved, make_table, pad) -> None:
    lengths = make_table(sequence_tokens=[100, 300], loss_token_count=[60, 200])
    plan = plan_batches(lengths, make_resolved(SFT, microbatch=2, pad_to_multiple_of=pad), seed=42)
    worst = plan.worst_case
    assert (worst.sequences_per_forward, worst.padded_length, worst.token_slots) == (2, 300, 600)
    assert plan.unit == "samples" and plan.padding_side == "right"
    assert worst.source_row_ids == ["train:1", "train:0"]


def test_sft_pad_to_multiple_of_rounds_the_length(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[100, 300])
    plan = plan_batches(lengths, make_resolved(SFT, microbatch=2, pad_to_multiple_of=64), seed=0)
    assert (plan.worst_case.padded_length, plan.worst_case.token_slots) == (320, 640)


def test_dpo_two_pairs_100_150_700_200_is_2800_slots(make_resolved, make_table) -> None:
    lengths = make_table(chosen_total_tokens=[100, 700], rejected_total_tokens=[150, 200])
    plan = plan_batches(lengths, make_resolved(DPO, microbatch=2), seed=42)
    worst = plan.worst_case
    assert (worst.sequences_per_forward, worst.padded_length, worst.token_slots) == (4, 700, 2800)
    assert worst.logits_positions == 2800  # DPO projects every position
    assert plan.unit == "pairs" and worst.rows_per_microbatch == 2


def test_dpo_rejected_only_outlier_drives_the_shape(make_resolved, make_table) -> None:
    lengths = make_table(chosen_total_tokens=[100, 500], rejected_total_tokens=[900, 100])
    worst = plan_batches(lengths, make_resolved(DPO), seed=1).worst_case
    assert worst.padded_length == 900 and worst.source_row_ids == ["train:0"]
    assert (worst.sequences_per_forward, worst.token_slots) == (2, 1800)


def test_prompt_masked_tokens_still_count_in_the_sequence(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[1000, 500], loss_token_count=[10, 400])
    worst = plan_batches(lengths, make_resolved(SFT), seed=1).worst_case
    assert worst.padded_length == 1000 and worst.token_slots == 1000


def test_sft_logits_positions_follow_the_loss_path(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[100, 300, 50], loss_token_count=[60, 200, 49])
    chunked = plan_batches(lengths, make_resolved(SFT, microbatch=2), seed=1).worst_case
    assert chunked.logits_positions == 512  # 260 label positions -> two 256-position chunks
    nll = plan_batches(lengths, make_resolved(SFT, microbatch=2, loss_path="nll"), seed=1)
    assert nll.worst_case.logits_positions == 600


def test_tail_batch_is_included_for_sft_and_dpo(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[5, 6, 7, 8, 9])
    plan = plan_batches(lengths, make_resolved(SFT, microbatch=2, accumulation=3), seed=7)
    assert plan.sampler.covers_all_rows and plan.sampler.dropped_rows == 0
    assert not plan.sampler.drop_last and plan.sampler.kind == "random"
    assert "마지막 부분 batch(1개 row)" in plan.sampler.note
    assert plan.effective_batch == 6 and plan.accumulation == 3


def test_sampler_max_only_when_order_cannot_matter(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[5, 60, 7, 8, 9])
    single = plan_batches(lengths, make_resolved(SFT, microbatch=1), seed=7)
    assert single.sampler_max is not None and single.sampler_max.name == "sampler_max"
    assert single.sampler_max.token_slots == single.worst_case.token_slots == 60
    assert single.issues == []
    grouped = plan_batches(lengths, make_resolved(SFT, microbatch=2), seed=7)
    assert grouped.sampler_max is None
    info = grouped.issues[0]
    assert info.code is ErrorCode.TRAINER_BATCH_CONSTRAINT and info.severity is Severity.INFO
    whole = plan_batches(lengths, make_resolved(SFT, microbatch=8), seed=7)
    assert whole.sampler_max is not None and whole.worst_case.rows_per_microbatch == 5


def test_failed_rows_are_not_planned(make_resolved, make_table) -> None:
    with pytest.raises(EstimatorError) as err:
        plan_batches(make_table(sequence_tokens=[None, None]), make_resolved(SFT), seed=1)
    assert err.value.issue.code is ErrorCode.SCAN_FAILED_ROWS


def test_batch_key_tracks_the_batch_layer(make_resolved, make_table) -> None:
    lengths = make_table(sequence_tokens=[100, 300])
    base = plan_batches(lengths, make_resolved(SFT, microbatch=2), seed=42).batch_key
    assert base.startswith("bat_")
    assert base == plan_batches(lengths, make_resolved(SFT, microbatch=2), seed=42).batch_key
    others = {
        plan_batches(lengths, make_resolved(SFT, microbatch=1), seed=42).batch_key,
        plan_batches(lengths, make_resolved(SFT, microbatch=2), seed=43).batch_key,
        plan_batches(
            lengths, make_resolved(SFT, microbatch=2, pad_to_multiple_of=8), seed=42
        ).batch_key,
        plan_batches(
            make_table(sequence_tokens=[100, 301]), make_resolved(SFT, microbatch=2), seed=42
        ).batch_key,
    }
    assert base not in others and len(others) == 4


# ---------------------------------------------------------------- GRPO


def prompts(n: int) -> list[int]:
    return [18 + (i * 37) % 251 for i in range(n)]


def test_grpo_example_config_uses_every_prompt(make_resolved, make_grpo, make_table) -> None:
    lengths = make_table(prompt_tokens=prompts(4656))
    plan = plan_batches(lengths, make_resolved(GRPO, grpo=make_grpo()), seed=42)
    g = plan.grpo
    assert (g.unique_prompts_per_generation, g.live_sequences, g.update_microbatch) == (1, 4, 1)
    assert (g.generation_batch_size, g.steps_per_generation, g.accumulation) == (4, 4, 4)
    assert g.max_prompt_length == 268 and g.completion_budgets == [1024, 2048, 4096, 8192]
    assert plan.sampler.kind == "repeat_sampler"
    assert plan.sampler.covers_all_rows and plan.sampler.dropped_rows == 0
    assert [s.name for s in plan.scenarios] == [f"budget_{b}" for b in (1024, 2048, 4096, 8192)]
    budget = plan.scenarios[0]
    assert (budget.prompt_length, budget.completion_length) == (268, 1024)
    assert budget.padded_length == 268 + 1024 and budget.token_slots == 1292
    assert budget.logits_positions == 1025  # L + 1 positions per sequence
    assert plan.worst_case.completion_length == 8192 and plan.worst_case.name == "worst_case"
    assert plan.sampler_max is not None  # U = 1: one prompt per generation batch
    assert plan.unit == "completions" and plan.effective_batch == 4
    assert not [i for i in plan.issues if i.severity is not Severity.INFO]


@pytest.mark.parametrize(
    ("gbs", "spg", "unique", "dropped"),
    [(4, 4, 1, 0), (20, 20, 5, 1), (28, 28, 7, 1), (128, 128, 32, 16), (8, 8, 2, 0)],
)
def test_grpo_repeat_sampler_tail_arithmetic(
    make_resolved, make_grpo, make_table, gbs, spg, unique, dropped
) -> None:
    lengths = make_table(prompt_tokens=prompts(4656))
    config = make_grpo(generation_batch_size=gbs, steps_per_generation=spg)
    plan = plan_batches(lengths, make_resolved(GRPO, grpo=config), seed=42)
    assert plan.grpo.unique_prompts_per_generation == unique
    assert plan.sampler.dropped_rows == dropped
    assert plan.sampler.covers_all_rows is (dropped == 0)
    drops = [i for i in plan.issues if i.code is ErrorCode.SAMPLER_DROPS_ROWS]
    assert len(drops) == (1 if dropped else 0)
    if dropped:
        assert drops[0].details["dropped_rows"] == dropped
        assert plan.sampler_max is None


def test_grpo_pad_to_multiple_of_rounds_prompt_and_completion(
    make_resolved, make_grpo, make_table
) -> None:
    lengths = make_table(prompt_tokens=[98, 42])
    config = make_grpo(budgets=[8])
    plan = plan_batches(lengths, make_resolved(GRPO, grpo=config, pad_to_multiple_of=64), seed=1)
    shape = plan.worst_case
    assert (shape.prompt_length, shape.completion_length) == (128, 64)
    assert shape.logits_positions == 65 and shape.padded_length == 192
    assert plan.grpo.max_prompt_length == 98  # rollout input is not rounded


@pytest.mark.parametrize(
    "config",
    [
        {"num_generations": 3},  # gbs 4 % G 3
        {"num_generations": 1, "generation_batch_size": 4},  # G < 2
        {"generation_batch_size": 8},  # gbs != B x spg
    ],
)
def test_grpo_configs_trl_rejects_raise(make_resolved, make_grpo, make_table, config) -> None:
    with pytest.raises(EstimatorError) as err:
        plan_batches(
            make_table(prompt_tokens=[10]), make_resolved(GRPO, grpo=make_grpo(**config)), seed=1
        )
    assert err.value.issue.code is ErrorCode.TRAINER_BATCH_CONSTRAINT


def test_grpo_without_budgets_raises(make_resolved, make_grpo, make_table) -> None:
    with pytest.raises(EstimatorError) as err:
        plan_batches(
            make_table(prompt_tokens=[10]), make_resolved(GRPO, grpo=make_grpo(budgets=[])), seed=1
        )
    assert err.value.issue.code is ErrorCode.GRPO_BUDGET_UNSPECIFIED


def test_grpo_live_sequence_mismatch_is_flagged(make_resolved, make_grpo, make_table) -> None:
    config = make_grpo(live_sequences=1)
    plan = plan_batches(make_table(prompt_tokens=[10]), make_resolved(GRPO, grpo=config), seed=1)
    assert plan.grpo.live_sequences == 4
    conflicts = [i for i in plan.issues if i.code is ErrorCode.CONFLICTING_OPTIONS]
    assert [i.details["trl_live_sequences"] for i in conflicts] == [4]


def test_grpo_common_batch_settings_mismatch_is_flagged(
    make_resolved, make_grpo, make_table
) -> None:
    config = make_grpo(update_microbatch=1, accumulation=4)
    plan = plan_batches(
        make_table(prompt_tokens=[10]),
        make_resolved(GRPO, grpo=config, microbatch=2, accumulation=1),
        seed=1,
    )
    assert (plan.microbatch, plan.accumulation) == (1, 4)  # GRPO settings win
    conflict = next(i for i in plan.issues if i.code is ErrorCode.CONFLICTING_OPTIONS)
    assert conflict.details["resolved_microbatch"] == 2
