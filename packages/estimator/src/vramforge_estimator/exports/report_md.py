"""`report.md`: the human-readable Korean report (plan.md §12.4).

Every value that may come from user input or a source (references, column names, messages) is
escaped: Markdown control characters are backslash-escaped and `<`, `>`, `&` become entities, so
the report never carries raw HTML and no Markdown link/image syntax can form (plan §18 XSS; a bare
public http(s) URL that survived redaction may still be auto-linked by GFM renderers). Numbers are
integer bytes with a one-decimal GiB rendering for display.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from vramforge_estimator.schemas import (
    AnalysisResult,
    DataPreservation,
    EvidenceLevel,
    HardwareFit,
    Issue,
    Objective,
    ScanCoverage,
    ScenarioEstimate,
    Severity,
    TrainingReadiness,
)
from vramforge_estimator.units import GiB

from .sanitize import redact_text

# Characters that can start inline Markdown constructs (code, emphasis, links/images, tables,
# strikethrough). With "[" and "]" escaped no link or image can form, so "(", "!" stay readable.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]|~])")

COVERAGE_KO = {
    ScanCoverage.NOT_STARTED: "시작 전",
    ScanCoverage.PARTIAL: "일부만 확인",
    ScanCoverage.COMPLETE: "전체 완료",
    ScanCoverage.FAILED: "실패",
}
PRESERVATION_KO = {
    DataPreservation.PENDING: "검사 대기",
    DataPreservation.VERIFIED: "검증됨",
    DataPreservation.VIOLATED: "위반",
    DataPreservation.UNKNOWN: "확인 불가",
}
READINESS_KO = {
    TrainingReadiness.READY: "실행 준비됨",
    TrainingReadiness.CONDITIONAL: "조건부",
    TrainingReadiness.UNSUPPORTED: "미지원",
}
EVIDENCE_KO = {
    EvidenceLevel.METADATA_ONLY: "메타데이터만 (메모리 산정 없음)",
    EvidenceLevel.ANALYTIC: "정적 추정 (analytic)",
    EvidenceLevel.CALIBRATED: "보정된 추정 (calibrated)",
    EvidenceLevel.MEASURED: "실측 (measured)",
}
FIT_KO = {
    HardwareFit.NOT_EVALUATED: "판정 안 함 (용량만 계산)",
    HardwareFit.EXPECTED_FIT: "선택한 가정에서 예상 적합",
    HardwareFit.LOW_MARGIN: "여유 부족",
    HardwareFit.EXCEEDS: "예상 용량 초과",
    HardwareFit.UNKNOWN: "판정 보류",
}
SEVERITY_KO = {Severity.ERROR: "오류", Severity.WARNING: "경고", Severity.INFO: "안내"}
CONDITIONAL_NOTE = (
    "조건부 결과입니다: 학습 준비 상태가 ready가 아니므로(제외한 구성 요소는 아래 미상·제외 항목 "
    "참고) 권장 용량은 전체 학습 시스템의 확정값이 아닙니다."
)


def esc(value: object) -> str:
    """Escape untrusted text for inline Markdown (also safe inside table cells)."""
    text = redact_text(str(value))
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", text)


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "미상"
    return f"{value / GiB:,.1f} GiB ({value:,} bytes)"


def fmt_range(low: int | None, high: int | None) -> str:
    if low is None or high is None:
        return "산정 불가 (미상 항목 포함)"
    if low == high:
        return fmt_bytes(high)
    return f"{low / GiB:,.1f}–{high / GiB:,.1f} GiB ({low:,}–{high:,} bytes)"


def _table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _primary(result: AnalysisResult) -> ScenarioEstimate | None:
    memory = result.memory
    if memory is None or not memory.scenarios:
        return None
    for scenario in memory.scenarios:
        if scenario.scenario_id == memory.primary_scenario_id:
            return scenario
    return None


def _issue_line(issue: Issue) -> str:
    label = SEVERITY_KO.get(issue.severity, issue.severity.value)
    return f"- {label} \\[{issue.code.value}\\] {esc(issue.user_message)}"


def _conditional(result: AnalysisResult) -> bool:
    """A result that is not ready (e.g. reward unspecified) is not a confirmed recommendation for
    the whole training system (plan §8.4). Scope exclusions such as evaluation alone are not."""
    return result.status.training_readiness is not TrainingReadiness.READY


def render_report(result: AnalysisResult) -> str:
    req = result.requested_config
    lines: list[str] = ["# VRAMForge 분석 보고서", ""]
    measured = result.memory is not None and result.memory.evidence_level is EvidenceLevel.MEASURED
    if not measured:
        lines += [
            "> 정적 추정 결과입니다. GPU에서 측정한 값이 아니며 학습 전체의 OOM 부재를 "
            "보증하지 않습니다.",
            "",
        ]

    # -- summary --------------------------------------------------------------------
    model = result.source_manifests.model
    dataset = result.source_manifests.dataset
    scan = result.dataset_scan
    model_text = (
        f"{esc(model.reference)} @ {esc(model.resolved_revision)}"
        if model
        else esc(req.model.reference) + " (확인 전)"
    )
    dataset_text = (
        f"{esc(dataset.reference)} @ {esc(dataset.resolved_revision)}"
        if dataset
        else esc(req.dataset.reference) + " (확인 전)"
    )
    if scan is not None:
        dataset_text += f" (config: {esc(scan.config or '기본')}, split: {esc(scan.split or '-')})"
    primary = _primary(result)
    summary_rows = [
        ["분석 ID", esc(result.analysis_id)],
        ["생성 시각 (UTC)", esc(result.created_at.isoformat())],
        ["요청 fingerprint", esc(result.analysis_fingerprint)],
        ["모델", model_text],
        ["데이터셋", dataset_text],
        [
            "학습 방식 / 전략",
            f"{req.training.objective.value.upper()} / {req.training.strategy.value.upper()}",
        ],
    ]
    if primary is not None and primary.devices:
        device = primary.devices[0]
        summary_rows.append(
            [
                "예상 피크 범위",
                f"{fmt_range(device.scenario_low_bytes, device.scenario_high_bytes)} "
                f"— 시나리오 {esc(primary.label)}",
            ]
        )
        summary_rows.append(["확정 상주량 (floor)", fmt_bytes(device.known_floor_bytes)])
        rec = primary.recommendation
        summary_rows.append(
            [
                "계획용 권장 용량" + (" (조건부)" if _conditional(result) else ""),
                fmt_bytes(rec.recommended_application_capacity_bytes)
                + f" (여유 {fmt_bytes(rec.planning_margin_bytes)})"
                if rec
                else "산정 불가 (피크 상한 미상)",
            ]
        )
    elif result.memory is not None:
        summary_rows.append(["예상 피크 범위", "시나리오별로 다릅니다 (아래 시나리오 비교 참고)"])
    else:
        summary_rows.append(["예상 피크 범위", "메모리 산정 결과 없음 (아래 오류·경고 참고)"])
    if result.hardware_fit is not None:
        summary_rows.append(["GPU 적합", esc(result.hardware_fit.message)])
    lines += ["## 요약", "", *_table(["항목", "값"], summary_rows), ""]

    # -- status axes ------------------------------------------------------------------
    axes = result.status
    status_rows = [
        ["데이터 스캔 범위", COVERAGE_KO[axes.scan_coverage]],
        ["데이터 보존", PRESERVATION_KO[axes.data_preservation]],
        [
            "학습 준비",
            READINESS_KO[axes.training_readiness] if axes.training_readiness else "판단 전",
        ],
        ["산정 근거", EVIDENCE_KO[axes.estimate_evidence] if axes.estimate_evidence else "없음"],
        ["하드웨어 적합", FIT_KO[axes.hardware_fit]],
    ]
    lines += ["## 상태", "", *_table(["축", "값"], status_rows), ""]

    # -- data -----------------------------------------------------------------------
    if scan is not None:
        lines += ["## 데이터", ""]
        lines.append(f"- 변환: {esc(scan.transformation_note)}")
        lines.append(
            f"- row: 읽음 {scan.rows_seen:,} / 성공 {scan.rows_ok:,} / 실패 {scan.rows_failed:,}"
            + (
                f" / 미처리 {scan.rows_unprocessed:,}"
                if scan.rows_unprocessed is not None
                else " / 미처리 수 미상"
            )
        )
        if scan.mapping_applied is not None:
            mapping = scan.mapping_applied
            roles = [
                f"{role}={esc(getattr(mapping, role))}"
                for role in ("system", "prompt", "chosen", "rejected", "completion", "messages")
                if getattr(mapping, role)
            ]
            lines.append(f"- 컬럼 매핑: {', '.join(roles) or '없음'}")
        lines.append("")
        stats_rows = []
        approx = False
        for branch in scan.branches:
            s = branch.stats
            approx = approx or not s.quantiles_exact

            def n(v: int | None) -> str:
                return "-" if v is None else f"{v:,}"

            stats_rows.append(
                [
                    esc(branch.branch.value),
                    n(s.count),
                    n(s.min),
                    n(s.p50),
                    n(s.p90),
                    n(s.p95),
                    n(s.p99),
                    n(s.max),
                    esc(s.max_row_id or "-"),
                ]
            )
        if stats_rows:
            lines += _table(
                ["branch", "count", "min", "p50", "p90", "p95", "p99", "max", "최대 row"],
                stats_rows,
            )
            lines.append("")
            if approx:
                lines += ["분위수는 근사값입니다 (근사 분위수). count와 max는 정확한 값입니다.", ""]
    context = result.context_validation
    if context is not None:
        observed = context.max_observed_length
        limit = context.effective_limit
        # GRPO checks the longest prompt plus a completion budget (the chosen one, else the
        # smallest candidate; larger candidates are reported per scenario), not an observed length.
        if req.training.objective is not Objective.GRPO:
            checked = "관측 최대 길이"
        elif req.grpo.completion_budget is not None:
            checked = "검사한 최대 길이(가장 긴 prompt + completion budget)"
        else:
            checked = "검사한 최대 길이(가장 긴 prompt + 가장 작은 completion budget 후보)"
        lines.append(
            f"- context: {checked} {f'{observed:,}' if observed is not None else '-'} 토큰, "
            f"상한 {f'{limit:,}' if limit is not None else '미상'} "
            f"({esc(context.limit_source or '근거 없음')}), "
            f"상태 {esc(context.status)}"
        )
        lines.append("")
    audit = result.preservation_audit
    if audit is not None and audit.checks:
        rows = [
            [
                esc(c.name.value),
                "통과" if c.passed else ("해당 없음/미확인" if c.passed is None else "실패"),
                esc(c.detail),
            ]
            for c in audit.checks
        ]
        lines += ["### 무절단 검사", "", *_table(["검사", "결과", "설명"], rows), ""]

    # -- batch plan -------------------------------------------------------------------
    plan = result.batch_plan
    if plan is not None:
        worst = plan.worst_case
        lines += [
            "## Batch 계획",
            "",
            f"- 단위: {esc(plan.unit)}, microbatch {plan.microbatch}, accumulation "
            f"{plan.accumulation}, effective batch {plan.effective_batch}",
            f"- 최악 shape: 길이 {worst.padded_length:,} × 시퀀스 {worst.sequences_per_forward} "
            f"= {worst.token_slots:,} token slots ({esc(worst.description)})",
            f"- sampler: {esc(plan.sampler.kind)}, 모든 row 사용 "
            f"{'예' if plan.sampler.covers_all_rows else '아니오'}",
            "",
        ]

    # -- memory ---------------------------------------------------------------------
    memory = result.memory
    if memory is not None and memory.scenarios:
        rows = []
        for scenario in memory.scenarios:
            first = scenario.devices[0] if scenario.devices else None
            rec = scenario.recommendation
            rows.append(
                [
                    esc(scenario.label),
                    fmt_range(first.scenario_low_bytes, first.scenario_high_bytes)
                    if first
                    else "-",
                    fmt_bytes(rec.recommended_application_capacity_bytes) if rec else "미상",
                    esc(scenario.hardware_fit.message),
                ]
            )
        lines += [
            "## 시나리오 비교",
            "",
            *_table(["시나리오", "예상 피크", "권장 용량", "적합"], rows),
            "",
        ]
        if _conditional(result):
            lines += [CONDITIONAL_NOTE, ""]
        shown = primary or memory.scenarios[0]
        if shown.devices:
            device = shown.devices[0]
            # Excluded phases have no numbers: "제외", never "미상" (unknown ≠ not applicable).
            phase_rows = [
                [
                    esc(p.phase.value),
                    "포함" if p.included else f"제외 ({esc(p.excluded_reason or '-')})",
                    fmt_bytes(p.bytes_low) if p.included else "-",
                    fmt_bytes(p.bytes_high) if p.included else "-",
                    esc(p.peak_timepoint or "-"),
                ]
                for p in device.phases
            ]
            lines += [
                f"## 단계별 피크 — {esc(shown.label)}",
                "",
                *_table(["단계", "범위", "low", "high", "피크 시점"], phase_rows),
                "",
            ]
            breakdown = device.peak_breakdown
            if breakdown is not None:
                item_rows = [
                    [
                        esc(i.name),
                        esc(i.category.value),
                        fmt_bytes(i.bytes_low),
                        fmt_bytes(i.bytes_high),
                        esc(i.evidence.value),
                    ]
                    for i in breakdown.items
                ]
                lines += [
                    f"### 피크 시점 구성 ({esc(breakdown.timepoint)})",
                    "",
                    "같은 시점에 살아 있는 항목의 합입니다.",
                    "",
                    *_table(["항목", "분류", "low", "high", "근거"], item_rows),
                    "",
                ]

    # -- assumptions, unknown, excluded ---------------------------------------------
    if result.assumptions:
        lines += ["## 가정", ""]
        lines += [
            f"- {esc(a.text)} ({esc(a.evidence.value)}"
            + (f", {esc(a.source)}" if a.source else "")
            + ")"
            for a in result.assumptions
        ]
        lines.append("")
    if result.unknown_components or result.excluded_components:
        lines += ["## 미상·제외 항목", ""]
        lines += [f"- 미상: {esc(u.name)} — {esc(u.reason)}" for u in result.unknown_components]
        lines += [f"- 제외: {esc(x.name)} — {esc(x.reason)}" for x in result.excluded_components]
        lines.append("")
    if result.errors or result.warnings:
        lines += ["## 오류와 경고", ""]
        lines += [_issue_line(i) for i in result.errors]
        lines += [_issue_line(i) for i in result.warnings]
        lines.append("")

    # -- requested vs resolved -----------------------------------------------------
    resolved = result.resolved_config
    lora = req.training.lora
    rows = [
        [
            "전략",
            esc(req.training.strategy.value),
            esc(resolved.strategy.value) if resolved else "-",
        ],
        [
            "4-bit 양자화",
            "예" if req.training.quantization.enabled else "아니오",
            ("예" if resolved.quantization.enabled else "아니오") if resolved else "-",
        ],
        [
            "LoRA r / alpha",
            f"{lora.r} / {lora.alpha:g}",
            f"{resolved.lora.r} / {resolved.lora.alpha:g}" if resolved and resolved.lora else "-",
        ],
        [
            "microbatch / accumulation",
            f"{req.training.microbatch_per_device or '프리셋'} / "
            f"{req.training.gradient_accumulation_steps or '프리셋'}",
            f"{resolved.microbatch} / {resolved.accumulation}" if resolved else "-",
        ],
        [
            "optimizer",
            esc(req.training.optimizer.value),
            esc(resolved.optimizer.name) if resolved else "-",
        ],
        [
            "로딩 dtype",
            esc(req.training.load_dtype.value),
            esc(resolved.load_dtype) if resolved else "-",
        ],
        [
            "gradient checkpointing",
            "예" if req.training.gradient_checkpointing else "아니오",
            ("예" if resolved.gradient_checkpointing else "아니오") if resolved else "-",
        ],
    ]
    lines += ["## 적용 설정", "", *_table(["항목", "요청", "적용"], rows), ""]
    lines += [
        f"- profile: {esc(result.profile_id or '없음')}",
        f"- dependency lock digest: {esc(result.dependency_lock_digest or '없음')}",
        f"- estimator: {esc(result.estimator_version)}, schema {esc(result.schema_version)}",
        "",
    ]
    return "\n".join(lines)


__all__ = ["esc", "fmt_bytes", "fmt_range", "render_report"]
