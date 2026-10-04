"""Scenario estimates (plan.md §9, §10, §11.4, §12.3).

One `ScenarioEstimate` per batch-plan scenario: SFT/DPO use the structural worst-case shape; GRPO
without an explicit completion budget gets one scenario per candidate budget and no primary
scenario (plan §5.4), with an explicit budget exactly one. The evidence level is the profile's
support grade for the objective x strategy, schedule issues and the device-map load budget become
estimate issues, and stated assumptions stay assumptions.
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
    EvidenceLevel,
    HardwareConfig,
    Issue,
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

from .engine import LoadBudget, assess_fit, evaluate, load_budget_issue, recommend

DEVICE = "cuda:0"


def scenario_shapes(
    cfg: ResolvedConfig, plan: BatchPlan
) -> list[tuple[str, str, dict[str, Any], BatchShape]]:
    """(scenario_id, Korean label, params, shape) for every scenario to estimate."""
    if cfg.objective is not Objective.GRPO or cfg.grpo is None:
        shape = plan.worst_case
        params: dict[str, Any] = {
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


EVIDENCE_NOTES = {
    EvidenceLevel.METADATA_ONLY: (
        "이 조합은 구조·가중치·학습 파라미터 수준까지만 지원되어 전체 VRAM 적합 판정을 하지 "
        "않습니다."
    ),
    EvidenceLevel.ANALYTIC: (
        "보정(calibration)·실측 값이 아니라 명시적 allocation과 workspace 가정을 가진 정적 "
        "추정입니다."
    ),
    EvidenceLevel.CALIBRATED: "등록된 GPU·소프트웨어 영역에서 보정된 정적 추정입니다.",
    EvidenceLevel.MEASURED: "이 job에서 관측한 단계별 peak입니다.",
}


def evidence_level(cfg: ResolvedConfig) -> tuple[EvidenceLevel, Assumption]:
    """The profile's support grade for the objective x strategy (plan §11.4) and the assumption
    that states it. A profile that is no longer registered, or a combination it does not support,
    vouches for nothing beyond the inventory: metadata_only."""
    from vramforge_estimator.compatibility.profiles import load_registry

    profile = load_registry().analytic.get(cfg.profile_id) if cfg.profile_id else None
    rule = profile.support_rule(cfg.objective, cfg.strategy) if profile else None
    if profile is None or rule is None or rule.grade is None:
        level = EvidenceLevel.METADATA_ONLY
        combo = f"{cfg.objective.value}/{cfg.strategy.value}"
        why = (
            f"등록된 profile({cfg.profile_id})을 찾을 수 없어"
            if profile is None
            else f"profile {profile.id}이 {combo}을 지원하지 않아"
        )
        text = f"근거 등급 metadata_only: {why} {EVIDENCE_NOTES[level]}"
    else:
        level = rule.grade
        text = (
            f"근거 등급 {level.value}: profile {profile.id}@{profile.version}의 "
            f"{cfg.objective.value}/{cfg.strategy.value} 지원 등급입니다. {EVIDENCE_NOTES[level]}"
        )
    return level, Assumption(id="evidence_level", text=text, source="docs/support-matrix.md")


def _assumptions(schedules: list[TrainingSchedule], evidence: Assumption) -> list[Assumption]:
    seen: dict[str, Assumption] = {}
    for sched in schedules:
        for a in sched.assumptions:
            seen.setdefault(a.id, a)
    seen.setdefault(evidence.id, evidence)
    return list(seen.values())


def _issues(schedules: list[TrainingSchedule], extra: list[Issue | None]) -> list[Issue]:
    """Schedule issues (the same issue from several scenarios once) plus estimate-level ones."""
    seen: dict[tuple[str, str | None, str], Issue] = {}
    for issue in [*(i for sched in schedules for i in sched.issues), *extra]:
        if issue is not None:
            key = (issue.code.value, issue.affected_component, issue.user_message)
            seen.setdefault(key, issue)
    return list(seen.values())


def load_budget(
    arch: ArchitectureAdapter, inventory: ModelInventory, cfg: ResolvedConfig
) -> LoadBudget | None:
    """`S_load` of the policy load from the architecture adapter; None if it cannot size it (the
    fit verdict then never claims a load failure)."""
    s_load = arch.loading_budget_bytes(inventory, cfg)
    if not isinstance(s_load, int) or s_load <= 0:
        return None
    return LoadBudget(s_load=s_load, quantized=cfg.quantization.enabled)


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
    level, evidence_note = evidence_level(cfg)
    budget = load_budget(arch, inventory, cfg)
    scenarios: list[ScenarioEstimate] = []
    schedules: list[TrainingSchedule] = []
    for sid, label, params, shape in scenario_shapes(cfg, plan):
        sched = trainer.build_schedule(inventory, cfg, arch, shape, plan, scope)
        schedules.append(sched)
        device = evaluate(sched, DEVICE)
        rec = recommend(device, margin_policy, hardware.external_reserved_bytes)
        fit = assess_fit(device, rec, hardware, readiness, evidence=level, load_budget=budget)
        scenarios.append(
            ScenarioEstimate(
                scenario_id=sid,
                label=label,
                params=params,
                batch_shape=shape,
                devices=[device],
                recommendation=rec,
                hardware_fit=fit,
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
        evidence_level=level,
        scenarios=scenarios,
        primary_scenario_id=None if multi_budget or not scenarios else scenarios[0].scenario_id,
        assumptions=_assumptions(schedules, evidence_note),
        issues=_issues(schedules, [load_budget_issue(budget, hardware)]),
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


__all__ = [
    "estimate_memory",
    "estimate_with",
    "evidence_level",
    "load_budget",
    "scenario_shapes",
]
