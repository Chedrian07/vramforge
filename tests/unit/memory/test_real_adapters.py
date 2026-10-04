"""End-to-end through the real architecture adapters on the pinned MiMo header fixture.

The plan example request is resolved and estimated offline; the checks are values that the
research documents state independently of this code (docs/research/loading-quantization-peft.md
§3.4, trl-sft-dpo.md implications D, trl-grpo.md R4), so they cross-check trainer, engine and
architecture adapter together.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from vramforge_estimator.compatibility import resolve
from vramforge_estimator.memory import estimate_memory
from vramforge_estimator.schemas import (
    AnalysisRequest,
    BatchPlan,
    BatchShape,
    HardwareConfig,
    MarginPolicy,
    Objective,
    SamplerPlan,
    ScopeConfig,
)
from vramforge_estimator.units import GiB

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "tests" / "fixtures" / "inventories" / "inventory_builder.py"
REQUEST = ROOT / "tests" / "fixtures" / "requests" / "plan_example_grpo.json"
V = 248_320  # MiMo vocab rows (lm_head)


@pytest.fixture(scope="module")
def mimo():
    if not BUILDER.exists():
        pytest.skip("MiMo inventory fixture not available")
    spec = importlib.util.spec_from_file_location("vf_inventory_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vf_inventory_builder"] = module
    spec.loader.exec_module(module)
    return module.load_mimo_inventory()


def request(**training) -> AnalysisRequest:
    data = json.loads(REQUEST.read_text(encoding="utf-8"))
    data["training"].update(training)
    return AnalysisRequest.model_validate(data)


def shape(objective: Objective, rows: int, length: int, **extra) -> BatchShape:
    return BatchShape(
        name=extra.pop("name", "worst_case"),
        objective=objective,
        rows_per_microbatch=1,
        sequences_per_forward=rows,
        padded_length=length,
        token_slots=rows * length,
        logits_positions=rows * length,
        description="real-adapter test",
        **extra,
    )


def estimate(inv, req, shapes):
    cfg, report = resolve(req, inv, None)
    assert cfg is not None, report.blockers
    plan = BatchPlan(
        objective=cfg.objective,
        unit={"sft": "samples", "dpo": "pairs", "grpo": "completions"}[cfg.objective.value],
        microbatch=cfg.microbatch,
        accumulation=cfg.accumulation,
        effective_batch=cfg.microbatch * cfg.accumulation,
        sampler=SamplerPlan(kind="random", covers_all_rows=True),
        worst_case=shapes[-1],
        scenarios=shapes if cfg.objective is Objective.GRPO else [],
        batch_key="bat_real",
    )
    est = estimate_memory(
        inv,
        cfg,
        plan,
        scope=ScopeConfig(),
        hardware=HardwareConfig(),
        margin_policy=MarginPolicy(),
        readiness=report.readiness,
    )
    return cfg, est


def alloc(scenario, name):
    return next(a for a in scenario.allocations if a.name == name)


def test_plan_example_grpo(mimo) -> None:
    budgets = (1024, 8192)
    shapes = [
        shape(
            Objective.GRPO, 1, 272 + b, name=f"budget_{b}", prompt_length=272, completion_length=b
        )
        for b in budgets
    ]
    cfg, est = estimate(mimo, request(), shapes)
    assert cfg.architecture_adapter == "qwen3_5_hybrid"
    params = next(r.resolved for r in cfg.resolutions if r.field == "lora.trainable_params")
    assert params == 43_278_336  # text-decoder targets, r=16
    assert est.primary_scenario_id is None
    for scenario in est.scenarios:
        device = scenario.devices[0]
        # QLoRA, bf16 load, CondGen, text-only targets: 7.664 GiB resident (weights + LoRA state)
        assert device.known_floor_bytes / GiB == pytest.approx(7.664, abs=0.005)
        assert device.scenario_high_bytes is not None
        assert sum(i.bytes_high or 0 for i in device.peak_breakdown.items) == (
            device.scenario_high_bytes
        )
    long = est.scenarios[-1]
    # device_map="auto" + bnb 4-bit: free x 0.81 must hold S_load = 7,765,103,072 B (research
    # §3.5: "free >= 8.93 GiB" for the bf16 CondGen load)
    budget = alloc(long, "policy.device_map_budget")
    assert budget.bytes_high == math.ceil(7_765_103_072 * 100 / 81)
    assert budget.bytes_high / GiB == pytest.approx(8.928, abs=0.001)
    logits = (
        alloc(long, "logits.policy.fp32").bytes_high
        + alloc(long, "loss.grpo.grad_logits").bytes_high
    )
    assert logits / GiB == pytest.approx(15.158, abs=0.001)  # 8E at L = 8,192, B_update = 1


def test_dpo_fp32_logits_of_the_longest_pair(mimo) -> None:
    _, est = estimate(mimo, request(objective="dpo"), [shape(Objective.DPO, 2, 2272)])
    (scenario,) = est.scenarios
    assert alloc(scenario, "logits.policy.fp32").bytes_high == 4_513_464_320
    assert alloc(scenario, "logits.reference.fp32").bytes_high == 4_513_464_320


def test_sft_chunked_loss_peak(mimo) -> None:
    _, est = estimate(mimo, request(objective="sft"), [shape(Objective.SFT, 1, 2272)])
    (scenario,) = est.scenarios
    chunk = alloc(scenario, "loss.chunked_nll.chunk_backward")
    assert chunk.bytes_high == 18 * 256 * V  # conservative 18 B per element (≈ 1.07 GiB)
    assert scenario.devices[0].scenario_high_bytes is not None


def test_trainable_lm_head_input_shares_the_final_hidden_storage(mimo) -> None:
    full = {"strategy": "full", "quantization": {"enabled": False}}
    _, est = estimate(mimo, request(objective="dpo", **full), [shape(Objective.DPO, 2, 2272)])
    (scenario,) = est.scenarios
    head = alloc(scenario, "lm_head.input")
    hidden = alloc(scenario, "policy.act.final_hidden")
    assert head.storage_alias_group == hidden.storage_alias_group is not None
    assert head.bytes_high == hidden.bytes_high == 2 * 2272 * 4096 * 2  # 2B x T x H, bf16


def test_dpo_lora_dropout_is_disabled_like_trl(mimo) -> None:
    # TRL DPOConfig.disable_dropout=True zeroes LoRA dropout, so p=0.05 must not add activations.
    lora = {"r": 16, "alpha": 32, "target_modules": "auto_verified"}
    shapes = [shape(Objective.DPO, 2, 2272)]
    _, plain = estimate(mimo, request(objective="dpo", lora={**lora, "dropout": 0.0}), shapes)
    _, dropped = estimate(mimo, request(objective="dpo", lora={**lora, "dropout": 0.05}), shapes)
    highs = [e.scenarios[0].devices[0].scenario_high_bytes for e in (plain, dropped)]
    assert highs[0] is not None and highs[0] == highs[1]
