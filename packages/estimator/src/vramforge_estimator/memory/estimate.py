"""Scenario estimates (plan.md §9, §10, §12.3).

One `ScenarioEstimate` per batch-plan scenario: SFT/DPO use the structural worst-case shape; GRPO
without an explicit completion budget gets one scenario per candidate budget and no primary
scenario (plan §5.4), with an explicit budget exactly one.
"""

from __future__ import annotations

from typing import Any

from vramforge_estimator.architectures import ArchitectureAdapter, get_adapter
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import (
    Assumption,
    BatchPlan,
    BatchShape,
    ErrorCode,
    Evidence,
    EvidenceLevel,
    HardwareConfig,
    MarginPolicy,
    MemoryEstimate,
    ModelInventory,
    Objective,
    ResolvedConfig,
    ScenarioEstimate,
    ScopeConfig,
    Stage,
    TrainingReadiness,
)
from vramforge_estimator.trainers import TrainingSchedule, get_trainer

from .engine import assess_fit, evaluate, recommend

DEVICE = "cuda:0"


def scenario_shapes(
    cfg: ResolvedConfig, plan: BatchPlan
) -> list[tuple[str, str, dict[str, Any], BatchShape]]:
    """(scenario_id, Korean label, params, shape) for every scenario to estimate."""
    if cfg.objective is not Objective.GRPO or cfg.grpo is None:
        shape = plan.worst_case
        params = {
            "rows_per_microbatch": shape.rows_per_microbatch,
            "sequences_per_forward": shape.sequences_per_forward,
            "padded_length": shape.padded_length,
            "accumulation": cfg.accumulation,
        }
        return [("default", "최악 batch (데이터 최대 길이 기준)", params, shape)]
    shapes = list(plan.scenarios) or [plan.worst_case]
    if cfg.grpo.budget_explicit:
        wanted = cfg.grpo.completion_budgets[0]
        shapes = [s for s in shapes if s.completion_length == wanted] or shapes[:1]
    out = []
    for shape in shapes:
        budget = shape.completion_length
        sid = shape.name if shape.name.startswith("budget_") else f"budget_{budget}"
        params = {
            "completion_budget": budget,
            "prompt_length": shape.prompt_length,
            "live_sequences": cfg.grpo.live_sequences,
            "update_microbatch": cfg.grpo.update_microbatch,
            "accumulation": cfg.grpo.accumulation,
        }
        label = f"생성 예산 {budget:,} tokens" if budget is not None else "생성 예산 미상"
        out.append((sid, label, params, shape))
    return out


def _assumptions(schedules: list[TrainingSchedule]) -> list[Assumption]:
    seen: dict[str, Assumption] = {}
    for sched in schedules:
        for a in sched.assumptions:
            seen.setdefault(a.id, a)
        for issue in sched.issues:
            key = f"issue:{issue.code.value}:{issue.affected_component or ''}"
            seen.setdefault(
                key, Assumption(id=key, text=issue.user_message, evidence=Evidence.UNKNOWN)
            )
    seen.setdefault(
        "evidence_analytic",
        Assumption(
            id="evidence_analytic",
            text="보정(calibration)·실측 profile이 없어 모든 수치는 analytic 추정입니다.",
            source="profiles/calibrated/README.md",
        ),
    )
    return list(seen.values())


def estimate_with(
    arch: ArchitectureAdapter,
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    plan: BatchPlan,
    *,
    scope: ScopeConfig,
    hardware: HardwareConfig,
    margin_policy: MarginPolicy,
    readiness: TrainingReadiness,
) -> MemoryEstimate:
    """`estimate_memory` with an explicit architecture adapter."""
    trainer = get_trainer(cfg.objective)
    scenarios: list[ScenarioEstimate] = []
    schedules: list[TrainingSchedule] = []
    for sid, label, params, shape in scenario_shapes(cfg, plan):
        sched = trainer.build_schedule(inventory, cfg, arch, shape, plan, scope)
        schedules.append(sched)
        device = evaluate(sched, DEVICE)
        rec = recommend(device, margin_policy, hardware.external_reserved_bytes)
        scenarios.append(
            ScenarioEstimate(
                scenario_id=sid,
                label=label,
                params=params,
                batch_shape=shape,
                devices=[device],
                recommendation=rec,
                hardware_fit=assess_fit(device, rec, hardware, readiness),
                excluded_components=list(sched.excluded),
                timepoints=list(sched.timepoints),
                allocations=list(sched.allocations),
            )
        )
    multi_budget = (
        cfg.objective is Objective.GRPO
        and cfg.grpo is not None
        and not cfg.grpo.budget_explicit
        and len(scenarios) > 1
    )
    return MemoryEstimate(
        evidence_level=EvidenceLevel.ANALYTIC,
        scenarios=scenarios,
        primary_scenario_id=None if multi_budget or not scenarios else scenarios[0].scenario_id,
        assumptions=_assumptions(schedules),
    )


def estimate_memory(
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    plan: BatchPlan,
    *,
    scope: ScopeConfig,
    hardware: HardwareConfig,
    margin_policy: MarginPolicy,
    readiness: TrainingReadiness,
) -> MemoryEstimate:
    """One `ScenarioEstimate` per batch-plan scenario (GRPO: per completion budget)."""
    if cfg.architecture_adapter is None:
        raise EstimatorError(
            make_issue(
                ErrorCode.UNSUPPORTED_ARCHITECTURE,
                "지원되는 architecture adapter가 없어 메모리를 계산할 수 없습니다.",
                stage=Stage.ESTIMATING,
                component="memory",
            )
        )
    return estimate_with(
        get_adapter(cfg.architecture_adapter),
        inventory,
        cfg,
        plan,
        scope=scope,
        hardware=hardware,
        margin_policy=margin_policy,
        readiness=readiness,
    )


__all__ = ["estimate_memory", "estimate_with", "scenario_shapes"]
