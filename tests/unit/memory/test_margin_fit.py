"""Planning margin (plan §10.2) and the six hardware-fit outcomes (plan §10.3)."""

from __future__ import annotations

import pytest

from vramforge_estimator.compatibility.profiles import clear_registry_cache
from vramforge_estimator.memory import assess_fit, recommend
from vramforge_estimator.memory.engine import capacity_bytes, planning_margin
from vramforge_estimator.schemas import (
    DeviceEstimate,
    HardwareConfig,
    HardwareFit,
    HardwareMode,
    MarginPolicy,
    TrainingReadiness,
)
from vramforge_estimator.units import GiB

READY = TrainingReadiness.READY


def est(high: int | None, floor: int = 0, low: int | None = None) -> DeviceEstimate:
    return DeviceEstimate(
        device="cuda:0",
        phases=[],
        timepoints=[],
        known_floor_bytes=floor,
        scenario_low_bytes=high if low is None else low,
        scenario_high_bytes=high,
    )


def custom(total: int | None = None, *, usable: int | None = None, reserved: int = 0):
    return HardwareConfig(
        mode=HardwareMode.CUSTOM,
        device_total_bytes=total,
        usable_bytes=usable,
        external_reserved_bytes=reserved,
    )


def fit(e: DeviceEstimate, hw: HardwareConfig, readiness: TrainingReadiness = READY):
    return assess_fit(e, recommend(e, MarginPolicy(), hw.external_reserved_bytes), hw, readiness)


# ---------------------------------------------------------------- margin


def test_margin_floor_and_fraction() -> None:
    policy = MarginPolicy()
    assert planning_margin(10 * GiB, policy) == 2 * GiB  # 15 % = 1.5 GiB < 2 GiB minimum
    assert planning_margin(20 * GiB, policy) == 3 * GiB  # exactly 15 %, no float drift
    assert planning_margin(100, MarginPolicy(min_bytes=0, fraction=0.15)) == 15
    assert planning_margin(101, MarginPolicy(min_bytes=0, fraction=0.15)) == 16  # ceil(15.15)


def test_recommendation_adds_margin_and_external_reserve() -> None:
    rec = recommend(est(20 * GiB), MarginPolicy(), external_reserved_bytes=GiB)
    assert rec is not None
    assert rec.planning_margin_bytes == 3 * GiB
    assert rec.recommended_application_capacity_bytes == 23 * GiB
    assert rec.required_total_device_capacity_bytes == 24 * GiB


def test_no_recommendation_without_high() -> None:
    assert recommend(est(None), MarginPolicy(), 0) is None


# ---------------------------------------------------------------- six outcomes


def test_capacity_only_is_not_evaluated() -> None:
    res = fit(est(10 * GiB), HardwareConfig())
    assert res.status is HardwareFit.NOT_EVALUATED and res.reason == "not_evaluated"
    assert res.capacity_bytes is None and res.utilization_ratio is None


def test_floor_exceeding_capacity_is_decided_even_with_unknowns() -> None:
    res = fit(est(None, floor=30 * GiB), custom(24 * GiB))
    assert res.status is HardwareFit.EXCEEDS and res.reason == "floor_exceeds_capacity"


def test_expected_fit_with_margin() -> None:
    res = fit(est(16 * GiB, floor=8 * GiB), custom(24 * GiB))
    assert res.status is HardwareFit.EXPECTED_FIT and res.reason == "fits_with_margin"
    # margin = max(2 GiB, 15 % of 16 GiB = 2.4 GiB)
    assert res.utilization_ratio == pytest.approx(18.4 / 24)


def test_low_margin_when_high_fits_but_margin_does_not() -> None:
    res = fit(est(23 * GiB, floor=8 * GiB), custom(24 * GiB))
    assert res.status is HardwareFit.LOW_MARGIN and res.reason == "margin_insufficient"


def test_high_exceeding_capacity_with_floor_below() -> None:
    res = fit(est(26 * GiB, floor=8 * GiB), custom(24 * GiB))
    assert res.status is HardwareFit.EXCEEDS and res.reason == "high_exceeds_capacity"
    assert res.utilization_ratio is not None and res.utilization_ratio > 1.0


def test_unknown_components_withhold_the_verdict() -> None:
    res = fit(est(None, floor=8 * GiB), custom(24 * GiB))
    assert res.status is HardwareFit.UNKNOWN and res.reason == "unknown_components"
    assert res.utilization_ratio is None


def test_unsupported_combination_withholds_the_verdict() -> None:
    res = fit(est(10 * GiB), custom(80 * GiB), TrainingReadiness.UNSUPPORTED)
    assert res.status is HardwareFit.UNKNOWN and res.reason == "unsupported"


def test_conditional_fit_says_so() -> None:
    res = fit(est(10 * GiB), custom(80 * GiB), TrainingReadiness.CONDITIONAL)
    assert res.status is HardwareFit.EXPECTED_FIT and "조건부" in res.message


# ---------------------------------------------------------------- capacity


def test_capacity_sources() -> None:
    assert capacity_bytes(custom(24 * GiB, reserved=2 * GiB)) == 22 * GiB
    assert capacity_bytes(custom(24 * GiB, usable=20 * GiB, reserved=2 * GiB)) == 20 * GiB
    assert capacity_bytes(custom(None)) is None
    clear_registry_cache()
    preset = HardwareConfig(mode=HardwareMode.GPU_PRESET, gpu_preset="h100-80gb")
    assert capacity_bytes(preset) == 80 * GiB
    unknown_preset = HardwareConfig(mode=HardwareMode.GPU_PRESET, gpu_preset="nope")
    assert capacity_bytes(unknown_preset) is None


def test_missing_capacity_is_not_evaluated() -> None:
    res = fit(est(10 * GiB), custom(None))
    assert res.status is HardwareFit.NOT_EVALUATED
    assert "용량 정보" in res.message
