"""Timepoint evaluation, planning margin and hardware fit (plan.md §9.1-9.2, §10.2-10.3).

`evaluate` sums the allocations alive at each timepoint (an alias group counts once) and takes the
maximum timepoint as the peak; per-phase maxima are never added. An allocation of unknown size
makes its timepoint unknown, and because an unknown part has no upper bound, any unknown timepoint
in an included phase makes the scenario low/high `None` (docs/methodology.md#peak-evaluation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    PHASE_ORDER,
    RESIDENT_CATEGORIES,
    BreakdownItem,
    CapacityRecommendation,
    DeviceEstimate,
    ErrorCode,
    EvidenceLevel,
    HardwareConfig,
    HardwareFit,
    HardwareFitResult,
    HardwareMode,
    Issue,
    MarginPolicy,
    PeakBreakdown,
    PhasePeak,
    Severity,
    Stage,
    Timepoint,
    TimepointTotal,
    TrainingReadiness,
    UnknownComponent,
)
from vramforge_estimator.trainers import TrainingSchedule
from vramforge_estimator.trainers.common import DEVICE_MAP_BUDGET
from vramforge_estimator.trainers.ledger import Contribution, alive_by_timepoint, contributions
from vramforge_estimator.units import to_gib

NOT_APPLICABLE_REASON = "이 학습 방식과 설정에서는 실행되지 않는 단계입니다 (해당 없음)."
UNKNOWN_REASON = "크기를 산정할 근거가 없습니다."


@dataclass(frozen=True)
class _Point:
    tp: Timepoint
    total: TimepointTotal
    known_high: int  # sum of the known parts (a lower bound when the timepoint is unknown)
    contribs: tuple[Contribution, ...]

    @property
    def unknown(self) -> bool:
        return self.total.bytes_high is None


def _points(schedule: TrainingSchedule, device: str) -> list[_Point]:
    timepoints = sorted(schedule.timepoints, key=lambda t: t.order)
    allocations = [a for a in schedule.allocations if a.device == device]
    alive = alive_by_timepoint(timepoints, allocations)
    points: list[_Point] = []
    for tp in timepoints:
        contribs = tuple(contributions(alive[tp.id]))
        unknown = sorted({c.spec.name for c in contribs if c.unknown})
        known = [c for c in contribs if not c.unknown]
        floor = sum(c.bytes_low or 0 for c in known if c.spec.category in RESIDENT_CATEGORIES)
        low = None if unknown else sum(c.bytes_low or 0 for c in known)
        high = None if unknown else sum(c.bytes_high or 0 for c in known)
        total = TimepointTotal(
            timepoint=tp.id,
            phase=tp.phase,
            bytes_low=low,
            bytes_high=high,
            known_floor_bytes=floor,
            unknown=unknown,
        )
        points.append(_Point(tp, total, sum(c.bytes_high or 0 for c in known), contribs))
    return points


def _peak(points: list[_Point]) -> _Point | None:
    """Known schedule: the timepoint with the largest high (earliest on ties). Otherwise the
    unknown timepoint with the largest known part, since only unknown timepoints can hide the
    true peak."""
    if not points:
        return None
    unknown = [p for p in points if p.unknown]
    if unknown:
        return max(unknown, key=lambda p: (p.known_high, -p.tp.order))
    return max(points, key=lambda p: (p.total.bytes_high or 0, -p.tp.order))


def _range(points: list[_Point]) -> tuple[int | None, int | None]:
    if not points or any(p.unknown for p in points):
        return None, None
    return (
        max(p.total.bytes_low or 0 for p in points),
        max(p.total.bytes_high or 0 for p in points),
    )


def _breakdown(point: _Point) -> PeakBreakdown:
    items: list[BreakdownItem] = []
    by_category: dict[str, int] = {}
    for c in point.contribs:
        note = c.spec.note
        if c.aliases:
            shared = "storage 공유: " + ", ".join(c.aliases)
            note = f"{note} ({shared})" if note else shared
        items.append(
            BreakdownItem(
                name=c.spec.name,
                category=c.spec.category,
                bytes_low=c.bytes_low,
                bytes_high=c.bytes_high,
                evidence=c.spec.evidence,
                note=note,
            )
        )
        if c.bytes_high is not None and not c.unknown:
            key = c.spec.category.value
            by_category[key] = by_category.get(key, 0) + c.bytes_high
    return PeakBreakdown(
        timepoint=point.tp.id,
        phase=point.tp.phase,
        items=items,
        by_category_high=by_category,
        total_low=point.total.bytes_low,
        total_high=point.total.bytes_high,
    )


def _phase_peaks(points: list[_Point], schedule: TrainingSchedule) -> list[PhasePeak]:
    excluded = {e.name: e.reason for e in schedule.excluded}
    phases: list[PhasePeak] = []
    for phase in PHASE_ORDER:
        members = [p for p in points if p.tp.phase is phase]
        if not members:
            phases.append(
                PhasePeak(
                    phase=phase,
                    included=False,
                    excluded_reason=excluded.get(phase.value, NOT_APPLICABLE_REASON),
                )
            )
            continue
        peak = _peak(members)
        low, high = _range(members)
        phases.append(
            PhasePeak(
                phase=phase,
                included=True,
                peak_timepoint=peak.tp.id if peak else None,
                bytes_low=low,
                bytes_high=high,
                known_floor_bytes=max(p.total.known_floor_bytes for p in members),
                unknown_components=sorted({n for p in members for n in p.total.unknown}),
            )
        )
    return phases


def _unknown_components(points: list[_Point]) -> list[UnknownComponent]:
    seen: dict[str, UnknownComponent] = {}
    for p in points:
        for c in p.contribs:
            if c.unknown and c.spec.name not in seen:
                seen[c.spec.name] = UnknownComponent(
                    name=c.spec.name, reason=c.spec.note or UNKNOWN_REASON, phase=p.tp.phase.value
                )
    return list(seen.values())


def evaluate(schedule: TrainingSchedule, device: str = "cuda:0") -> DeviceEstimate:
    """Sum allocations alive at each timepoint (alias groups counted once) and pick the maximum
    timepoint. floor = known resident categories; low/high = None if an unknown is alive at the
    peak-candidate timepoints (plan §9.1, §10.2)."""
    points = _points(schedule, device)
    peak = _peak(points)
    low, high = _range(points)
    return DeviceEstimate(
        device=device,
        phases=_phase_peaks(points, schedule),
        timepoints=[p.total for p in points],
        peak_phase=peak.tp.phase if peak else None,
        peak_timepoint=peak.tp.id if peak else None,
        known_floor_bytes=max((p.total.known_floor_bytes for p in points), default=0),
        scenario_low_bytes=low,
        scenario_high_bytes=high,
        peak_breakdown=_breakdown(peak) if peak else None,
        unknown_components=_unknown_components(points),
    )


# ---------------------------------------------------------------- margin (plan §10.2)


def planning_margin(high: int, policy: MarginPolicy) -> int:
    """max(min_bytes, ceil(high x fraction)) with exact decimal arithmetic."""
    return max(policy.min_bytes, math.ceil(high * Fraction(str(policy.fraction))))


def recommend(
    estimate: DeviceEstimate, policy: MarginPolicy, external_reserved_bytes: int
) -> CapacityRecommendation | None:
    """planning_margin = max(min_bytes, high × fraction); None when high is unknown."""
    high = estimate.scenario_high_bytes
    if high is None:
        return None
    margin = planning_margin(high, policy)
    recommended = high + margin
    return CapacityRecommendation(
        planning_margin_bytes=margin,
        recommended_application_capacity_bytes=recommended,
        external_reserved_bytes=external_reserved_bytes,
        required_total_device_capacity_bytes=recommended + external_reserved_bytes,
        policy=policy,
    )


# ---------------------------------------------------------------- hardware fit (plan §10.3)


def capacity_bytes(hardware: HardwareConfig) -> int | None:
    """Application capacity: usable_bytes, else device total (or preset) - external_reserved."""
    if hardware.mode is HardwareMode.CAPACITY_ONLY:
        return None
    if hardware.usable_bytes is not None:
        return hardware.usable_bytes
    total = hardware.device_total_bytes
    if total is None and hardware.gpu_preset:
        from vramforge_estimator.compatibility.profiles import load_registry

        gpu = load_registry().gpu(hardware.gpu_preset)
        total = gpu.total_bytes if gpu else None
    if total is None:
        return None
    return max(total - hardware.external_reserved_bytes, 0)


def _gib(value: int) -> str:
    return f"{to_gib(value):.2f} GiB"


@dataclass(frozen=True)
class LoadBudget:
    """`S_load` of the policy load (architecture adapter `loading_budget_bytes`) and whether
    bitsandbytes 4-bit loading applies (docs/methodology.md#load-phase).

    TRL loads with `device_map="auto"`; on one GPU without `max_memory` transformers budgets
    0.9 x free memory and the bnb 4-bit quantizer scales that by 0.90 again, so 4-bit loading
    needs S_load <= 0.81 x free and fails with ValueError otherwise; a dense model offloads the
    excess to CPU (docs/research/loading-quantization-peft.md §3.5, §4.5)."""

    s_load: int
    quantized: bool

    @property
    def factor(self) -> Fraction:
        return DEVICE_MAP_BUDGET["quantized" if self.quantized else "dense"]

    def exceeded(self, capacity: int) -> bool:
        """S_load > factor x capacity (exact). The CUDA context makes free memory smaller than
        the capacity, so a load that fails here fails for every context size."""
        return self.s_load > self.factor * capacity

    def message(self, capacity: int) -> str:
        budget = math.floor(self.factor * capacity)
        if self.quantized:
            return (
                '4-bit(bitsandbytes) 모델을 단일 GPU에 device_map="auto"(TRL 기본)로 로드하면 '
                "transformers가 가용 메모리의 0.9배, bitsandbytes가 다시 0.9배만 배치 예산으로 "
                f"씁니다(실효 {float(self.factor):g}배). 로딩 크기 S_load {_gib(self.s_load)}가 "
                f"가용 용량 {_gib(capacity)}의 {float(self.factor):g}배({_gib(budget)})를 넘어 "
                "학습 전에 로딩이 ValueError로 실패합니다. model_init_kwargs에 "
                'device_map={"": 0}을 지정하면 이 예산 검사 없이 GPU 0에 바로 올리고, '
                "max_memory를 지정하면 transformers의 0.9배가 빠져 bitsandbytes의 0.9배만 "
                "남습니다."
            )
        return (
            '단일 GPU에 device_map="auto"(TRL 기본)로 로드하면 transformers가 가용 메모리의 '
            f"{float(self.factor):g}배만 배치 예산으로 씁니다. 로딩 크기 S_load "
            f"{_gib(self.s_load)}가 가용 용량 {_gib(capacity)}의 {float(self.factor):g}배"
            f"({_gib(budget)})를 넘어 일부 모듈이 CPU로 offload되며, 모든 가중치가 GPU에 "
            '상주한다는 이 계산의 가정과 달라집니다. model_init_kwargs에 device_map={"": 0}을 '
            "지정하면 예산 검사 없이 GPU 0에 올리고, max_memory를 지정하면 0.9배 축소가 빠집니다."
        )


def load_budget_issue(budget: LoadBudget | None, hardware: HardwareConfig) -> Issue | None:
    """LOAD_BUDGET_EXCEEDED warning when the policy load cannot pass the device-map budget of the
    selected capacity (None when no capacity is known or the load fits)."""
    capacity = capacity_bytes(hardware)
    if budget is None or capacity is None or not budget.exceeded(capacity):
        return None
    return make_issue(
        ErrorCode.LOAD_BUDGET_EXCEEDED,
        budget.message(capacity),
        severity=Severity.WARNING,
        stage=Stage.ESTIMATING,
        component="model.loading",
        s_load_bytes=budget.s_load,
        capacity_bytes=capacity,
        budget_factor=f"{float(budget.factor):g}",
        required_free_bytes=math.ceil(budget.s_load / budget.factor),
    )


def _peak_note(estimate: DeviceEstimate) -> str:
    """The device-map check timepoint holds a budget requirement, not an allocation."""
    if estimate.peak_timepoint and estimate.peak_timepoint.endswith("_device_map_check"):
        return (
            " 피크는 로딩 직전 device_map 예산 검사 시점입니다 (실제 할당이 아니라 로딩에 필요한 "
            "가용 메모리)."
        )
    return ""


def assess_fit(
    estimate: DeviceEstimate,
    recommendation: CapacityRecommendation | None,
    hardware: HardwareConfig,
    readiness: TrainingReadiness,
    *,
    evidence: EvidenceLevel = EvidenceLevel.ANALYTIC,
    load_budget: LoadBudget | None = None,
) -> HardwareFitResult:
    """The outcomes of plan §10.3 (incl. not_evaluated when no hardware is selected), the
    device-map load budget and the metadata-only grade, which allows no full VRAM verdict
    (plan §11.4; docs/methodology.md#hardware-fit)."""
    if hardware.mode is HardwareMode.CAPACITY_ONLY:
        return HardwareFitResult(
            status=HardwareFit.NOT_EVALUATED,
            reason="not_evaluated",
            message="하드웨어를 선택하지 않아 필요한 용량만 표시합니다 (적합 판정 없음).",
        )
    capacity = capacity_bytes(hardware)
    if capacity is None:
        return HardwareFitResult(
            status=HardwareFit.NOT_EVALUATED,
            reason="not_evaluated",
            message="GPU 용량 정보가 없어 적합 판정을 하지 않습니다.",
        )
    high = estimate.scenario_high_bytes
    margin = recommendation.planning_margin_bytes if recommendation else None
    ratio = (
        (high + margin) / capacity if high is not None and margin is not None and capacity else None
    )
    base = {"capacity_bytes": capacity, "utilization_ratio": ratio}
    if readiness is TrainingReadiness.UNSUPPORTED:
        return HardwareFitResult(
            status=HardwareFit.UNKNOWN,
            reason="unsupported",
            message="지원되지 않는 조합이라 적합 판정을 보류합니다.",
            **base,
        )
    if estimate.known_floor_bytes > capacity:
        return HardwareFitResult(
            status=HardwareFit.EXCEEDS,
            reason="floor_exceeds_capacity",
            message=(
                f"확정된 상주 메모리({_gib(estimate.known_floor_bytes)})만으로 "
                f"가용 용량({_gib(capacity)})을 초과합니다."
            ),
            **base,
        )
    if load_budget is not None and load_budget.exceeded(capacity):
        return HardwareFitResult(
            status=HardwareFit.EXCEEDS,
            reason="load_budget_insufficient",
            message=load_budget.message(capacity),
            **base,
        )
    if evidence is EvidenceLevel.METADATA_ONLY:
        return HardwareFitResult(
            status=HardwareFit.UNKNOWN,
            reason="unsupported",
            message=(
                "이 조합의 근거 등급이 metadata_only라 전체 VRAM 적합 판정을 하지 않습니다 "
                "(구조·가중치·학습 파라미터까지만 지원)."
            ),
            **base,
        )
    if high is None or margin is None:
        return HardwareFitResult(
            status=HardwareFit.UNKNOWN,
            reason="unknown_components",
            message="크기를 산정할 수 없는 구성 요소가 있어 적합 판정을 보류합니다.",
            **base,
        )
    if high > capacity:
        return HardwareFitResult(
            status=HardwareFit.EXCEEDS,
            reason="high_exceeds_capacity",
            message=(
                f"예상 피크({_gib(high)})가 가용 용량({_gib(capacity)})을 초과합니다. "
                "실측 또는 설정 검토가 필요합니다." + _peak_note(estimate)
            ),
            **base,
        )
    if high + margin > capacity:
        return HardwareFitResult(
            status=HardwareFit.LOW_MARGIN,
            reason="margin_insufficient",
            message="예상 피크는 들어가지만 계획용 여유를 확보할 수 없습니다 (여유 부족)."
            + _peak_note(estimate),
            **base,
        )
    suffix = (
        " 범위에서 제외한 구성 요소가 있어 조건부 판정입니다."
        if readiness is TrainingReadiness.CONDITIONAL
        else ""
    )
    return HardwareFitResult(
        status=HardwareFit.EXPECTED_FIT,
        reason="fits_with_margin",
        message="선택한 가정에서 예상 적합합니다." + suffix,
        **base,
    )


__all__ = [
    "LoadBudget",
    "assess_fit",
    "capacity_bytes",
    "evaluate",
    "load_budget_issue",
    "planning_margin",
    "recommend",
]
