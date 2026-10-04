"""Timepoint evaluation (plan §9.1-9.2, §12.2, §19.1)."""

from __future__ import annotations

import pytest

from vramforge_estimator.memory import evaluate
from vramforge_estimator.schemas import (
    AllocationCategory,
    AllocationSpec,
    Evidence,
    ExcludedComponent,
    Phase,
    Timepoint,
)
from vramforge_estimator.trainers import TrainingSchedule
from vramforge_estimator.units import GiB

LOAD = "MODEL_LOAD_AND_QUANTIZE:load_peak"
FWD = "POLICY_FORWARD_BACKWARD:forward"
BWD = "POLICY_FORWARD_BACKWARD:backward"
OPT = "OPTIMIZER_STEP:step"


def tp(tp_id: str, order: int) -> Timepoint:
    phase, name = tp_id.split(":")
    return Timepoint(id=tp_id, phase=Phase(phase), name=name, description=name, order=order)


def alloc(
    name: str,
    size: int | None,
    live_at: list[str],
    category: AllocationCategory = AllocationCategory.SAVED_ACTIVATIONS,
    *,
    low: int | None = None,
    alias: str | None = None,
    device: str = "cuda:0",
) -> AllocationSpec:
    return AllocationSpec(
        name=name,
        category=category,
        bytes_low=size if low is None else low,
        bytes_high=size,
        live_at=live_at,
        storage_alias_group=alias,
        device=device,
        evidence=Evidence.ANALYTIC if size is not None else Evidence.UNKNOWN,
        note=None if size is not None else "activation profile 없음",
    )


def schedule(*allocs: AllocationSpec, excluded: list[ExcludedComponent] | None = None):
    tps = [tp(LOAD, 0), tp(FWD, 1), tp(BWD, 2), tp(OPT, 3)]
    return TrainingSchedule(timepoints=tps, allocations=list(allocs), excluded=excluded or [])


def test_non_overlapping_phases_take_the_max_not_the_sum() -> None:
    # plan §19.1: 10 GiB / 12 GiB phases that never coexist -> 12 GiB, not 22 GiB.
    est = evaluate(
        schedule(
            alloc("load.transient", 10 * GiB, [LOAD], AllocationCategory.LOAD_TRANSIENT),
            alloc("policy.activations", 12 * GiB, [BWD]),
        )
    )
    assert est.scenario_high_bytes == 12 * GiB
    assert est.scenario_low_bytes == 12 * GiB
    assert est.peak_timepoint == BWD and est.peak_phase is Phase.POLICY_FORWARD_BACKWARD
    load_phase = next(p for p in est.phases if p.phase is Phase.MODEL_LOAD_AND_QUANTIZE)
    assert load_phase.bytes_high == 10 * GiB


def test_resident_weights_are_counted_once_per_timepoint() -> None:
    weights = alloc(
        "weights",
        4 * GiB,
        ["MODEL_LOAD_AND_QUANTIZE:*", FWD, BWD, OPT],
        AllocationCategory.WEIGHTS_BASE,
    )
    est = evaluate(schedule(weights, alloc("acts", 3 * GiB, [FWD, BWD])))
    assert est.scenario_high_bytes == 7 * GiB
    assert est.known_floor_bytes == 4 * GiB
    assert [t.bytes_high for t in est.timepoints] == [4 * GiB, 7 * GiB, 7 * GiB, 4 * GiB]


def test_unknown_activation_is_null_not_zero() -> None:
    # plan §19.1: no activation profile -> null (never 0); the fit verdict is withheld.
    weights = alloc("weights", 4 * GiB, [LOAD, FWD, BWD, OPT], AllocationCategory.WEIGHTS_BASE)
    est = evaluate(schedule(weights, alloc("policy.activations", None, [FWD, BWD])))
    assert est.scenario_low_bytes is None and est.scenario_high_bytes is None
    assert est.known_floor_bytes == 4 * GiB
    assert [u.name for u in est.unknown_components] == ["policy.activations"]
    assert est.unknown_components[0].reason == "activation profile 없음"
    assert est.peak_timepoint in (FWD, BWD)
    totals = {t.timepoint: t for t in est.timepoints}
    assert totals[FWD].bytes_high is None and totals[FWD].unknown == ["policy.activations"]
    assert totals[LOAD].bytes_high == 4 * GiB
    breakdown = est.peak_breakdown
    assert breakdown is not None and breakdown.total_high is None
    assert any(i.bytes_high is None for i in breakdown.items)


def test_unknown_outside_the_peak_still_blocks_the_range() -> None:
    # An unknown part has no upper bound, so a smaller-looking unknown timepoint can still be the peak.
    est = evaluate(
        schedule(
            alloc("load.transient", 20 * GiB, [LOAD], AllocationCategory.LOAD_TRANSIENT),
            alloc("optimizer.workspace", None, [OPT], AllocationCategory.WORKSPACE),
        )
    )
    assert est.scenario_high_bytes is None
    assert est.peak_timepoint == OPT


def test_alias_group_counts_once_and_breakdown_matches_total() -> None:
    est = evaluate(
        schedule(
            alloc("norm.out", 2 * GiB, [FWD], alias="h0"),
            alloc("q_proj.input", 2 * GiB, [FWD], alias="h0"),
            alloc("big.input", 3 * GiB, [FWD], low=1 * GiB, alias="h0"),
            alloc("other", 1 * GiB, [FWD]),
        )
    )
    fwd = next(t for t in est.timepoints if t.timepoint == FWD)
    assert fwd.bytes_high == 4 * GiB  # max(2, 2, 3) + 1
    assert fwd.bytes_low == 3 * GiB  # max(2, 2, 1) + 1
    breakdown = est.peak_breakdown
    assert breakdown is not None
    assert len(breakdown.items) == 2
    assert sum(i.bytes_high or 0 for i in breakdown.items) == breakdown.total_high
    assert sum(i.bytes_low or 0 for i in breakdown.items) == breakdown.total_low
    rep = next(i for i in breakdown.items if i.name == "big.input")
    assert rep.note is not None and "norm.out" in rep.note


def test_breakdown_is_exactly_the_peak_timepoint() -> None:
    weights = alloc("weights", 5 * GiB, [LOAD, FWD, BWD, OPT], AllocationCategory.WEIGHTS_BASE)
    grads = alloc("grads", 1 * GiB, [BWD, OPT], AllocationCategory.GRADIENTS)
    est = evaluate(
        schedule(
            weights,
            grads,
            alloc("acts", 2 * GiB, [FWD, BWD]),
            alloc("recompute", 1 * GiB, [BWD], AllocationCategory.RECOMPUTE_WORKING_SET),
            alloc("loss", 3 * GiB, [FWD], AllocationCategory.LOGITS_AND_LOSS),
        )
    )
    breakdown = est.peak_breakdown
    assert breakdown is not None and breakdown.timepoint == est.peak_timepoint == FWD
    assert {i.name for i in breakdown.items} == {"weights", "acts", "loss"}
    assert breakdown.total_high == est.scenario_high_bytes == 10 * GiB
    assert sum(breakdown.by_category_high.values()) == breakdown.total_high
    assert est.known_floor_bytes == 6 * GiB  # weights + grads at BWD/OPT


def test_low_and_high_maxima_may_come_from_different_timepoints() -> None:
    est = evaluate(
        schedule(
            alloc("a", 10 * GiB, [FWD], low=2 * GiB),
            alloc("b", 8 * GiB, [BWD], low=7 * GiB),
        )
    )
    assert est.scenario_high_bytes == 10 * GiB
    assert est.scenario_low_bytes == 7 * GiB
    assert est.peak_timepoint == FWD


def test_wildcard_and_device_filter() -> None:
    est = evaluate(
        schedule(
            alloc("acts", 2 * GiB, ["POLICY_FORWARD_BACKWARD:*"]),
            alloc("host.copy", 50 * GiB, [FWD], device="cpu"),
        )
    )
    highs = {t.timepoint: t.bytes_high for t in est.timepoints}
    assert highs[FWD] == highs[BWD] == 2 * GiB
    assert highs[LOAD] == highs[OPT] == 0


def test_dangling_timepoint_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown timepoint"):
        evaluate(schedule(alloc("x", 1, ["REWARD:score"])))


def test_ties_pick_the_earliest_timepoint() -> None:
    est = evaluate(schedule(alloc("a", GiB, [FWD]), alloc("b", GiB, [BWD])))
    assert est.peak_timepoint == FWD


def test_phase_peaks_include_excluded_and_not_applicable_phases() -> None:
    reason = "reward가 지정되지 않아 제외했습니다."
    est = evaluate(
        schedule(
            alloc("acts", GiB, [FWD]),
            excluded=[ExcludedComponent(name="REWARD", reason=reason)],
        )
    )
    phases = {p.phase: p for p in est.phases}
    assert list(phases) == list(Phase)
    assert phases[Phase.REWARD].included is False and phases[Phase.REWARD].excluded_reason == reason
    rollout = phases[Phase.ROLLOUT_PREFILL_AND_DECODE]
    assert rollout.included is False and "해당 없음" in (rollout.excluded_reason or "")
    pfb = phases[Phase.POLICY_FORWARD_BACKWARD]
    assert pfb.included and pfb.peak_timepoint == FWD and pfb.bytes_high == GiB


def test_empty_schedule_has_no_peak() -> None:
    est = evaluate(TrainingSchedule(timepoints=[], allocations=[]))
    assert est.peak_timepoint is None and est.scenario_high_bytes is None
    assert est.known_floor_bytes == 0
