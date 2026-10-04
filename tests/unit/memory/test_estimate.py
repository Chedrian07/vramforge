"""Scenario estimates (plan §5.4, §10, §12.3) with the fake architecture adapter."""

from __future__ import annotations

import pytest
from vf_fakes import FakeArch, dpo_shape, grpo_shape, make_cfg, make_inventory, make_plan, sft_shape

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.memory import estimate as estimate_mod
from vramforge_estimator.memory import estimate_memory
from vramforge_estimator.memory.estimate import estimate_with
from vramforge_estimator.schemas import (
    EvidenceLevel,
    HardwareConfig,
    HardwareFit,
    HardwareMode,
    MarginPolicy,
    Objective,
    ScopeConfig,
    Strategy,
    TrainingReadiness,
)
from vramforge_estimator.units import GiB

INV = make_inventory()
KW = {
    "scope": ScopeConfig(),
    "hardware": HardwareConfig(),
    "margin_policy": MarginPolicy(),
    "readiness": TrainingReadiness.READY,
}


def test_sft_has_one_primary_scenario_with_consistent_breakdown() -> None:
    cfg = make_cfg(inventory=INV)
    est = estimate_with(FakeArch(), INV, cfg, make_plan(cfg, [sft_shape(1, 500)]), **KW)
    assert est.evidence_level is EvidenceLevel.ANALYTIC
    assert [s.scenario_id for s in est.scenarios] == ["default"]
    assert est.primary_scenario_id == "default"
    (scenario,) = est.scenarios
    device = scenario.devices[0]
    assert device.scenario_high_bytes is not None
    assert sum(i.bytes_high or 0 for i in device.peak_breakdown.items) == device.scenario_high_bytes
    rec = scenario.recommendation
    assert rec is not None
    assert rec.recommended_application_capacity_bytes == device.scenario_high_bytes + 2 * GiB
    assert scenario.hardware_fit.status is HardwareFit.NOT_EVALUATED
    assert scenario.timepoints and scenario.allocations
    assert {e.name for e in scenario.excluded_components} == {
        "EVALUATION",
        "CHECKPOINT_SAVE_OR_CONSOLIDATE",
    }
    assert any(a.id == "evidence_analytic" for a in est.assumptions)


def test_dpo_scenario_uses_the_worst_case_shape() -> None:
    cfg = make_cfg(Objective.DPO, inventory=INV)
    shape = dpo_shape(pairs=1, length=300)
    est = estimate_with(FakeArch(), INV, cfg, make_plan(cfg, [shape]), **KW)
    assert est.scenarios[0].batch_shape == shape
    assert est.scenarios[0].params["sequences_per_forward"] == 2


def test_grpo_without_budget_has_one_scenario_per_budget_and_no_primary() -> None:
    budgets = (1024, 2048, 4096, 8192)
    cfg = make_cfg(Objective.GRPO, inventory=INV, accumulation=4, budgets=budgets)
    shapes = [grpo_shape(prompt=272, budget=b) for b in budgets]
    est = estimate_with(FakeArch(), INV, cfg, make_plan(cfg, shapes), **KW)
    assert [s.scenario_id for s in est.scenarios] == [f"budget_{b}" for b in budgets]
    assert est.primary_scenario_id is None
    highs = [s.devices[0].scenario_high_bytes for s in est.scenarios]
    assert all(h is not None for h in highs) and highs == sorted(highs)
    assert est.scenarios[0].params["live_sequences"] == 4
    reward = next(e for s in est.scenarios[:1] for e in s.excluded_components if e.name == "REWARD")
    assert reward.code is not None


def test_grpo_with_explicit_budget_has_a_single_primary_scenario() -> None:
    cfg = make_cfg(Objective.GRPO, inventory=INV, accumulation=4, budgets=(2048,))
    shapes = [grpo_shape(prompt=272, budget=b) for b in (1024, 2048)]
    est = estimate_with(FakeArch(), INV, cfg, make_plan(cfg, shapes), **KW)
    assert [s.scenario_id for s in est.scenarios] == ["budget_2048"]
    assert est.primary_scenario_id == "budget_2048"


def test_estimate_memory_uses_the_registered_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get_adapter(adapter_id: str):
        seen.append(adapter_id)
        return FakeArch()

    monkeypatch.setattr(estimate_mod, "get_adapter", fake_get_adapter)
    cfg = make_cfg(inventory=INV)
    est = estimate_memory(INV, cfg, make_plan(cfg, [sft_shape()]), **KW)
    assert seen == ["fake"] and est.scenarios


def test_missing_adapter_is_an_error() -> None:
    cfg = make_cfg(inventory=INV).model_copy(update={"architecture_adapter": None})
    with pytest.raises(EstimatorError) as info:
        estimate_memory(INV, cfg, make_plan(cfg, [sft_shape()]), **KW)
    assert info.value.issue.code.value == "UNSUPPORTED_ARCHITECTURE"


def test_fit_is_evaluated_per_scenario() -> None:
    cfg = make_cfg(Objective.SFT, Strategy.LORA, inventory=INV)
    hw = HardwareConfig(mode=HardwareMode.CUSTOM, device_total_bytes=80 * GiB)
    est = estimate_with(
        FakeArch(), INV, cfg, make_plan(cfg, [sft_shape()]), **{**KW, "hardware": hw}
    )
    fit = est.scenarios[0].hardware_fit
    assert fit.status is HardwareFit.EXPECTED_FIT and fit.capacity_bytes == 80 * GiB


def test_unknown_activations_withhold_range_and_fit() -> None:
    cfg = make_cfg(inventory=INV)
    hw = HardwareConfig(mode=HardwareMode.CUSTOM, device_total_bytes=80 * GiB)
    est = estimate_with(
        FakeArch(unknown_activations=True),
        INV,
        cfg,
        make_plan(cfg, [sft_shape()]),
        **{**KW, "hardware": hw},
    )
    scenario = est.scenarios[0]
    assert scenario.devices[0].scenario_high_bytes is None
    assert scenario.recommendation is None
    assert scenario.hardware_fit.status is HardwareFit.UNKNOWN
