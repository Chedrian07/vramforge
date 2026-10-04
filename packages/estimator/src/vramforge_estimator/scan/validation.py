"""Context-limit validation (plan.md §7.5) and the no-truncation preservation audit (§7.4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from vramforge_estimator.schemas import (
    BatchPlan,
    BranchStats,
    ContextValidation,
    DataPreservation,
    DatasetScanResult,
    ErrorCode,
    Issue,
    Objective,
    PreservationAudit,
    PreservationCheck,
    PreservationCheckName,
    ScanCoverage,
    ScopeConfig,
    Severity,
    Stage,
    TokenizerManifest,
)

from .base import LengthTable
from .stats import primary_branch

# transformers stores "no limit" as LARGE_INTEGER/VERY_LARGE_INTEGER (1e20/1e30) in
# model_max_length; such values are sentinels, not supported context
# (docs/research/example-model-dataset.md R9).
SENTINEL_MIN = 10**20


def _model_input_stats(scan: DatasetScanResult) -> BranchStats | None:
    target = primary_branch(scan.objective)
    return next((b for b in scan.branches if b.branch is target), None)


def _rows_above(branch: BranchStats, threshold: int) -> tuple[int, bool]:
    """Rows (with a known length) longer than `threshold`, and whether the count is exact.

    Exact when the threshold is outside [min, max), when the longest rows kept in `top_rows`
    already reach below it, or when ``threshold + 1`` starts a histogram bin (the power-of-two
    bins of stats.LengthAccumulator make typical context thresholds land there). Otherwise the
    threshold falls inside a bin and the largest provable lower bound is returned."""
    stats = branch.stats
    if stats.max is None or threshold >= stats.max:
        return 0, True
    if stats.min is not None and threshold < stats.min:
        return stats.count, True
    from_top = sum(1 for r in branch.top_rows if r.length > threshold)
    if from_top < len(branch.top_rows) or len(branch.top_rows) >= stats.count:
        return from_top, True
    first_exceeding = threshold + 1
    full_bins = sum(b.count for b in stats.histogram if b.lo >= first_exceeding)
    aligned = any(b.lo == first_exceeding for b in stats.histogram)
    return max(from_top, full_bins), aligned


def _model_input_lengths(objective: Objective, table: LengthTable) -> Iterator[int | None]:
    """Per-row length of the primary branch (what the model sees), from the length table."""
    if objective is Objective.SFT:
        yield from table.sequence_tokens
    elif objective is Objective.DPO:
        for chosen, rejected in zip(
            table.chosen_total_tokens, table.rejected_total_tokens, strict=True
        ):
            present = [v for v in (chosen, rejected) if v is not None]
            yield max(present) if present else None
    else:
        yield from table.prompt_tokens


def _counted_rows_above(scan: DatasetScanResult, table: LengthTable, threshold: int) -> int | None:
    """Exact count from the loaded length table, or None when the table is not this scan's."""
    if len(table) != scan.rows_ok:
        return None
    if table.preprocess_key is not None and table.preprocess_key != scan.preprocess_key:
        return None
    lengths = _model_input_lengths(scan.objective, table)
    return sum(1 for v in lengths if v is not None and v > threshold)


def validate_context(
    scan: DatasetScanResult,
    *,
    model_declared_max: int | None,
    tokenizer: TokenizerManifest | None,
    backend_verified_max: int | None,
    extra_tokens: int = 0,
    lengths: LengthTable | None = None,
) -> ContextValidation:
    """Compare the longest sequence (+ `extra_tokens`, e.g. a GRPO completion budget) with the
    limits, keeping model/tokenizer/backend limits separate (plan §7.5).

    The effective limit is the smallest of the model's declared maximum and the backend-verified
    maximum; the tokenizer's ``model_max_length`` is only a fallback when neither is known and is
    never used when it is a sentinel. A partial scan or failed rows can prove "exceeded" but never
    "ok" (status "unknown").

    ``exceeded_rows`` is None when no limit (or no row length) is known, never 0. It is counted
    from the scan statistics; when those only bound it (threshold inside a histogram bin) and the
    scan's own `lengths` table (``scan.load_lengths``) is given, it is counted exactly from the
    table. ``exceeded_rows_exact`` is False whenever the number is a lower bound: also for a
    partial scan or failed rows, whose unknown lengths may exceed the limit too."""
    tok_max = tokenizer.model_max_length if tokenizer else None
    sentinel = bool(
        tokenizer
        and (
            tokenizer.model_max_length_is_sentinel
            or (tok_max is not None and tok_max >= SENTINEL_MIN)
        )
    )
    limit: int | None = None
    source: str | None = None
    for value, name in (
        (model_declared_max, "model.max_position_embeddings"),
        (backend_verified_max, "backend.verified_max"),
    ):
        if value is not None and (limit is None or value < limit):
            limit, source = value, name
    if limit is None and tok_max is not None and not sentinel:
        limit, source = tok_max, "tokenizer.model_max_length"

    branch = _model_input_stats(scan)
    longest = branch.stats.max if branch else None
    observed = longest + extra_tokens if longest is not None else None
    status: Literal["ok", "exceeded", "unknown"] = "unknown"
    exceeded_rows: int | None = None
    exact = False
    if limit is not None and observed is not None and branch is not None:
        threshold = limit - extra_tokens
        exceeded_rows, exact = _rows_above(branch, threshold)
        if not exact and lengths is not None:
            counted = _counted_rows_above(scan, lengths, threshold)
            if counted is not None:
                exceeded_rows, exact = counted, True
        every_row_known = scan.coverage is ScanCoverage.COMPLETE and scan.rows_failed == 0
        exact = exact and every_row_known
        if observed > limit:
            status = "exceeded"
        elif every_row_known:
            status = "ok"
    return ContextValidation(
        model_declared_max=model_declared_max,
        tokenizer_model_max_length=tok_max,
        tokenizer_limit_is_sentinel=sentinel,
        backend_verified_max=backend_verified_max,
        effective_limit=limit,
        limit_source=source,
        max_observed_length=observed,
        exceeded_rows=exceeded_rows,
        exceeded_rows_exact=exact,
        status=status,
    )


Outcome = Literal["pass", "na", "pending", "unknown", "violated"]


@dataclass
class _Check:
    name: PreservationCheckName
    outcome: Outcome
    detail: str
    issue: Issue | None = None

    @property
    def passed(self) -> bool | None:
        if self.outcome == "pass":
            return True
        if self.issue is not None:  # a failed check (violation, or unverifiable data)
            return False
        return None


def _issue(code: ErrorCode, message: str, **details: object) -> Issue:
    return Issue(
        code=code,
        severity=Severity.ERROR,
        stage=Stage.VALIDATING_DATA,
        user_message=message,
        details=dict(details),
    )


def _full_read(scan: DatasetScanResult) -> _Check:
    """Every row read and tokenized. Failed rows keep the scan PARTIAL even when every row was
    read (`rows_unprocessed == 0`); results cached before that rule may still say COMPLETE with
    failed rows, which is read the same way."""
    name = PreservationCheckName.FULL_READ
    if scan.coverage is ScanCoverage.COMPLETE and not scan.rows_failed:
        return _Check(name, "pass", f"전체 {scan.rows_seen:,} row를 끝까지 읽고 모두 처리했습니다.")
    all_read = scan.coverage is ScanCoverage.COMPLETE or scan.rows_unprocessed == 0
    if scan.rows_seen == 0 and all_read:
        detail = "선택한 split에 row가 없어 검사할 데이터가 없습니다."
        return _Check(name, "unknown", detail, _issue(ErrorCode.EMPTY_DATASET, detail))
    if scan.rows_failed and all_read:
        detail = (
            f"전체 {scan.rows_seen:,} row를 읽었지만 {scan.rows_failed:,} row를 읽거나 토큰화하지 "
            "못해 그 길이와 보존 여부를 검증할 수 없습니다."
        )
        return _Check(
            name,
            "unknown",
            detail,
            _issue(
                ErrorCode.SCAN_FAILED_ROWS,
                detail,
                rows_seen=scan.rows_seen,
                rows_failed=scan.rows_failed,
            ),
        )
    failed = f", 실패 {scan.rows_failed:,} row" if scan.rows_failed else ""
    detail = (
        f"스캔이 끝나지 않았습니다(coverage: {scan.coverage.value}, "
        f"처리 {scan.rows_seen:,} row{failed}). 읽지 않은 row의 길이는 확인되지 않았습니다."
    )
    return _Check(name, "unknown", detail, _issue(ErrorCode.SCAN_PARTIAL, detail))


def _packing_checks(packing: bool) -> list[_Check]:
    if packing:
        reason = (
            "엄격 모드에서는 packing을 지원하지 않습니다. TRL packing은 정수 max_length가 필요하며 "
            "초과 토큰을 자르거나(bfd) 다른 행으로 나누고(bfd_split), 서로 다른 샘플을 한 행에 "
            "잇습니다."
        )
        violation = _issue(ErrorCode.DATA_PRESERVATION_VIOLATION, reason, option="packing")
        return [
            _Check(PreservationCheckName.NO_LENGTH_DROP, "violated", reason, violation),
            _Check(PreservationCheckName.NO_SPLIT_OR_CONCAT, "violated", reason, violation),
            _Check(PreservationCheckName.PACKING_SAFE, "violated", reason, violation),
        ]
    return [
        _Check(
            PreservationCheckName.NO_LENGTH_DROP,
            "pass",
            "max_length=None 조건의 TRL 1.14.1 전처리를 재현했고 토큰화에 truncation이 없어 길이 "
            "때문에 잘리거나 삭제되는 row가 없습니다.",
        ),
        _Check(
            PreservationCheckName.NO_SPLIT_OR_CONCAT,
            "pass",
            "샘플을 분할하거나 다른 샘플과 이어 붙이지 않습니다(packing 사용 안 함).",
        ),
        _Check(PreservationCheckName.PACKING_SAFE, "na", "packing을 사용하지 않습니다(해당 없음)."),
    ]


def _template(scan: DatasetScanResult, loss_rows: int) -> _Check:
    name = PreservationCheckName.TEMPLATE_CONTENT_PRESERVED
    if loss_rows:
        detail = (
            "chat template이 매핑된 content를 원문 그대로 렌더링하지 않은 row가 "
            f"{loss_rows:,}개 있습니다."
        )
        return _Check(
            name,
            "violated",
            detail,
            _issue(ErrorCode.TEMPLATE_CONTENT_LOSS, detail, rows=loss_rows),
        )
    if scan.rows_ok == 0:
        return _Check(name, "unknown", "검사할 수 있는 row가 없습니다.")
    return _Check(name, "pass", "모든 매핑된 content가 렌더링 결과에 원문 그대로 들어 있습니다.")


def _last_batch(batch_plan: BatchPlan | None) -> _Check:
    name = PreservationCheckName.LAST_BATCH_INCLUDED
    if batch_plan is None:
        return _Check(name, "pending", "batch 계획 전이라 아직 확인하지 않았습니다.")
    sampler = batch_plan.sampler
    if sampler.covers_all_rows:
        return _Check(
            name, "pass", sampler.note or "sampler가 마지막 batch까지 모든 row를 사용합니다."
        )
    detail = (
        sampler.note or f"sampler가 epoch마다 {sampler.dropped_rows:,} row를 사용하지 않습니다."
    )
    return _Check(
        name,
        "violated",
        detail,
        _issue(ErrorCode.SAMPLER_DROPS_ROWS, detail, dropped_rows=sampler.dropped_rows),
    )


def _context(context: ContextValidation) -> _Check:
    name = PreservationCheckName.CONTEXT_WITHIN_LIMIT
    if context.status == "ok":
        return _Check(
            name,
            "pass",
            f"최대 길이 {context.max_observed_length:,} 토큰이 context 상한 "
            f"{context.effective_limit:,}({context.limit_source}) 이내입니다.",
        )
    if context.status == "exceeded":
        if context.exceeded_rows is None:
            rows = "초과 row 수 미확인"
        elif context.exceeded_rows_exact:
            rows = f"초과 row {context.exceeded_rows:,}개"
        else:
            rows = f"초과 row {context.exceeded_rows:,}개 이상"
        detail = (
            f"최대 길이 {context.max_observed_length:,} 토큰이 context 상한 "
            f"{context.effective_limit:,}({context.limit_source})을 넘습니다({rows}). "
            "자동 절단·분할·row 제외를 하지 않으며, GPU 메모리를 늘려도 모델의 context 제약은 "
            "해결되지 않습니다."
        )
        return _Check(
            name,
            "violated",
            detail,
            _issue(
                ErrorCode.CONTEXT_EXCEEDED,
                detail,
                max_observed_length=context.max_observed_length,
                effective_limit=context.effective_limit,
                limit_source=context.limit_source,
                exceeded_rows=context.exceeded_rows,
                exceeded_rows_exact=context.exceeded_rows_exact,
            ),
        )
    if context.effective_limit is None:
        return _Check(name, "unknown", "검증된 context 상한이 없어 확인하지 못했습니다.")
    return _Check(
        name,
        "unknown",
        "스캔이 완전하지 않아 모든 row가 context 상한 이내인지 확인하지 못했습니다.",
    )


def _eval_scope(scope: ScopeConfig) -> _Check:
    name = PreservationCheckName.EVAL_SCOPE_CONSISTENT
    if not scope.include_evaluation:
        return _Check(name, "na", "평가 데이터를 범위에 포함하지 않았습니다(해당 없음).")
    return _Check(
        name, "pending", "평가 split의 스캔 결과가 없어 같은 검사를 통과했는지 확인하지 못했습니다."
    )


_ORDER = list(PreservationCheckName)


def audit_preservation(
    scan: DatasetScanResult,
    *,
    context: ContextValidation,
    batch_plan: BatchPlan | None,
    packing: bool,
    scope: ScopeConfig,
    template_content_loss_rows: int,
) -> PreservationAudit:
    """Evaluate the eight no-truncation conditions of plan §7.4.

    VIOLATED if any condition is broken; UNKNOWN if the data could not be fully verified (partial
    scan, failed rows, unknown context); PENDING while a later stage (batch plan, eval scan) is
    missing; VERIFIED only when every applicable condition passed."""
    checks = [
        _full_read(scan),
        *_packing_checks(packing),
        _template(scan, template_content_loss_rows),
        _last_batch(batch_plan),
        _context(context),
        _eval_scope(scope),
    ]
    checks.sort(key=lambda c: _ORDER.index(c.name))
    outcomes = {c.outcome for c in checks}
    if "violated" in outcomes:
        status = DataPreservation.VIOLATED
    elif "unknown" in outcomes:
        status = DataPreservation.UNKNOWN
    elif "pending" in outcomes:
        status = DataPreservation.PENDING
    else:
        status = DataPreservation.VERIFIED
    violations: list[Issue] = []
    for check in checks:
        if check.issue is not None and check.issue not in violations:
            violations.append(check.issue)
    return PreservationAudit(
        status=status,
        checks=[PreservationCheck(name=c.name, passed=c.passed, detail=c.detail) for c in checks],
        violations=violations,
    )
