"""Exact length statistics (plan §7.7): counts, nearest-rank quantiles, histogram, top rows."""

from __future__ import annotations

from itertools import pairwise

from vramforge_estimator.preprocessing import TokenizedRecord
from vramforge_estimator.scan.stats import LengthAccumulator, branch_values, primary_branch
from vramforge_estimator.schemas import Branch, Objective


def filled(values: list[int]) -> LengthAccumulator:
    acc = LengthAccumulator()
    for i, v in enumerate(values):
        acc.add(v, i, f"train:{i}")
    return acc


def test_counts_min_max_mean_total_are_exact() -> None:
    stats = filled([5, 3, 9, 9, 1]).stats()
    assert (stats.count, stats.min, stats.max, stats.total_tokens) == (5, 1, 9, 27)
    assert stats.mean == 27 / 5
    assert stats.max_row_id == "train:2"  # first row reaching the maximum
    assert stats.quantiles_exact is True


def test_nearest_rank_quantiles() -> None:
    stats = filled(list(range(1, 101))).stats()
    assert (stats.p50, stats.p90, stats.p95, stats.p99) == (50, 90, 95, 99)
    # ceil(q*n): n=7 -> ranks 4, 7, 7, 7
    small = filled([10, 20, 30, 40, 50, 60, 70]).stats()
    assert (small.p50, small.p90, small.p95, small.p99) == (40, 70, 70, 70)
    single = filled([42]).stats()
    assert (single.p50, single.p99) == (42, 42)


def test_quantiles_are_observed_values() -> None:
    values = [3, 3, 3, 1000]
    acc = filled(values)
    assert {acc.quantile(p) for p in (50, 90, 95, 99)} <= set(values)


def test_empty_accumulator_has_no_numbers() -> None:
    stats = LengthAccumulator().stats()
    assert stats.count == 0 and stats.max is None and stats.mean is None and stats.p99 is None
    assert stats.total_tokens == 0 and stats.histogram == []


def test_top_rows_descending_with_index_tiebreak() -> None:
    values = [7, 9, 9, 1, 8, 9, 2, 3, 4, 5, 6, 10, 0]
    top = filled(values).top_rows()
    assert [r.length for r in top] == [10, 9, 9, 9, 8, 7, 6, 5, 4, 3]
    assert [r.row_id for r in top[:4]] == ["train:11", "train:1", "train:2", "train:5"]


def test_histogram_covers_every_row_with_power_of_two_bins() -> None:
    values = [18, 61, 61, 123, 137, 166, 268]
    bins = filled(values).histogram()
    assert sum(b.count for b in bins) == len(values)
    width = bins[0].hi - bins[0].lo
    assert width & (width - 1) == 0  # power of two
    assert all(b.hi - b.lo == width for b in bins)
    assert all(a.hi == b.lo for a, b in pairwise(bins))
    assert bins[0].lo <= min(values) and bins[-1].hi > max(values)


def test_count_above_and_state_round_trip() -> None:
    acc = filled([5, 3, 9, 9, 1, 12])
    assert acc.count_above(8) == 3 and acc.count_above(12) == 0
    again = LengthAccumulator.from_state(acc.to_state())
    assert again.stats() == acc.stats()
    assert again.top_rows() == acc.top_rows()
    again.add(100, 6, "train:6")
    assert again.stats().max_row_id == "train:6"


def test_branch_values_per_objective() -> None:
    dpo = TokenizedRecord(
        row_id="r",
        objective=Objective.DPO,
        ok=True,
        prompt_tokens=10,
        chosen_total_tokens=30,
        rejected_total_tokens=70,
        chosen_completion_tokens=20,
        rejected_completion_tokens=60,
    )
    values = dict(branch_values(Objective.DPO, dpo))
    assert values[Branch.PAIR_MAX] == 70  # rejected-only outlier drives the pair length
    assert values[Branch.CHOSEN_SEQUENCE] == 30 and values[Branch.REJECTED] == 60
    assert primary_branch(Objective.DPO) is Branch.PAIR_MAX
    assert primary_branch(Objective.SFT) is Branch.SEQUENCE
    assert primary_branch(Objective.GRPO) is Branch.PROMPT
