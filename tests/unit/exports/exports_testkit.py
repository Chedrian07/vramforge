"""Builds realistic analysis results for export tests (via the pipeline with fake modules)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vramforge_estimator.pipeline import analyze
from vramforge_estimator.schemas import (
    AnalysisResult,
    DpoResolved,
    GrpoResolved,
    Issue,
    Objective,
    ReferenceStrategy,
    RewardKind,
    RolloutBackend,
    TrainingReadiness,
)


def load_pipeline_fakes() -> ModuleType:
    name = "pipeline_fakes"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "pipeline" / "pipeline_fakes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fakes = load_pipeline_fakes()

GRPO_BUDGET = 1024


def grpo_resolved(*, budget_explicit: bool = True, reward: RewardKind = RewardKind.CPU_RULE):
    return GrpoResolved(
        num_generations=4,
        generation_batch_size=4,
        steps_per_generation=4,
        num_iterations=1,
        completion_budgets=[GRPO_BUDGET] if budget_explicit else [1024, 2048, 4096, 8192],
        budget_explicit=budget_explicit,
        beta=0.0,
        reference_needed=False,
        reward_kind=reward,
        rollout_backend=RolloutBackend.TRANSFORMERS_SHARED_POLICY,
        live_sequences=4,
        update_microbatch=1,
        accumulation=4,
    )


def build_result(
    objective: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    readiness: TrainingReadiness = TrainingReadiness.READY,
    request_overrides: dict[str, Any] | None = None,
    resolved_overrides: dict[str, Any] | None = None,
    report_kwargs: dict[str, Any] | None = None,
    estimate_error: Issue | None = None,
) -> AnalysisResult:
    obj = Objective(objective)
    extra: dict[str, Any] = {}
    if obj is Objective.GRPO:
        extra["grpo"] = grpo_resolved()
    elif obj is Objective.DPO:
        extra["dpo"] = DpoResolved(
            reference_strategy=ReferenceStrategy.FROZEN_BASE_SWITCH,
            beta=0.1,
            loss_type="sigmoid",
        )
    extra.update(resolved_overrides or {})
    resolved = fakes.resolved_config(obj, **extra)
    report = fakes.compat_report(readiness=readiness, **(report_kwargs or {}))
    fakes.FakeModules(
        resolve=lambda req, inv, tok: (resolved, report),
        estimate_memory=fakes.raising(estimate_error) if estimate_error else None,
    ).install(monkeypatch)
    overrides = {"training.objective": objective}
    if obj is Objective.GRPO:
        overrides.update(
            {"grpo.completion_budget": GRPO_BUDGET, "grpo.reward": {"kind": "cpu_rule"}}
        )
    overrides.update(request_overrides or {})
    artifact_dir = tmp_path / f"artifacts-{objective}-{readiness.value}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return analyze(fakes.example_request(**overrides), fakes.FakeContext(artifact_dir=artifact_dir))
