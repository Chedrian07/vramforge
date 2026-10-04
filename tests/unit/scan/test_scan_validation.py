"""Context validation (plan §7.5) and the eight preservation checks (plan §7.4)."""

from __future__ import annotations

from vramforge_estimator.scan import audit_preservation, validate_context
from vramforge_estimator.scan.stats import LengthAccumulator, primary_branch
from vramforge_estimator.schemas import (
    BatchPlan,
    BatchShape,
    BranchStats,
    DataPreservation,
    DatasetScanResult,
    ErrorCode,
    Objective,
    PreservationCheckName,
    SamplerPlan,
    ScanCoverage,
    ScopeConfig,
    TokenizerManifest,
)

C = PreservationCheckName


def scan_result(
    lengths: list[int],
    *,
    objective: Objective = Objective.SFT,
    coverage: ScanCoverage = ScanCoverage.COMPLETE,
    rows_failed: int = 0,
) -> DatasetScanResult:
    acc = LengthAccumulator()
    for i, n in enumerate(lengths):
        acc.add(n, i, f"train:{i}")
    return DatasetScanResult(
        coverage=coverage,
        objective=objective,
        transformation_note="t",
        rows_seen=len(lengths) + rows_failed,
        rows_ok=len(lengths),
        rows_failed=rows_failed,
        branches=[
            BranchStats(
                branch=primary_branch(objective), stats=acc.stats(), top_rows=acc.top_rows()
            )
        ],
        preprocess_key="pre_x",
        tokenizer_fingerprint="tok",
        preprocessing_adapter="trl-1.14.1-sft",
        preprocessing_adapter_version="1",
    )


def tokenizer(max_len: int | None, *, sentinel: bool = False) -> TokenizerManifest:
    return TokenizerManifest(
        tokenizer_class="T",
        vocab_size=10,
        chat_template_present=True,
        model_max_length=max_len,
        model_max_length_is_sentinel=sentinel,
        fingerprint="f",
    )


def plan(*, covers: bool = True, dropped: int = 0) -> BatchPlan:
    shape = BatchShape(
        name="worst_case",
        objective=Objective.SFT,
        rows_per_microbatch=1,
        sequences_per_forward=1,
        padded_length=10,
        token_slots=10,
        logits_positions=10,
        description="d",
    )
    sampler = SamplerPlan(kind="random", covers_all_rows=covers, dropped_rows=dropped, note="")
    return BatchPlan(
        objective=Objective.SFT,
        unit="samples",
        microbatch=1,
        accumulation=1,
        effective_batch=1,
        sampler=sampler,
        worst_case=shape,
        batch_key="bat_x",
    )


# ---------------------------------------------------------------- validate_context


def test_model_limit_is_effective_and_tokenizer_limit_kept_separately() -> None:
    ctx = validate_context(
        scan_result([51, 2272, 300]),
        model_declared_max=262_144,
        tokenizer=tokenizer(262_144),
        backend_verified_max=None,
    )
    assert ctx.status == "ok" and ctx.exceeded_rows == 0
    assert (ctx.effective_limit, ctx.limit_source) == (262_144, "model.max_position_embeddings")
    assert ctx.tokenizer_model_max_length == 262_144 and not ctx.tokenizer_limit_is_sentinel
    assert ctx.max_observed_length == 2272


def test_sentinel_tokenizer_limit_is_never_a_context() -> None:
    ctx = validate_context(
        scan_result([100]),
        model_declared_max=None,
        tokenizer=tokenizer(int(1e30)),
        backend_verified_max=None,
    )
    assert ctx.tokenizer_limit_is_sentinel and ctx.effective_limit is None
    assert ctx.status == "unknown"


def test_tokenizer_limit_is_only_a_fallback_and_backend_can_be_tighter() -> None:
    fallback = validate_context(
        scan_result([100]),
        model_declared_max=None,
        tokenizer=tokenizer(2048),
        backend_verified_max=None,
    )
    assert (fallback.effective_limit, fallback.limit_source) == (2048, "tokenizer.model_max_length")
    tighter = validate_context(
        scan_result([100]),
        model_declared_max=8192,
        tokenizer=tokenizer(2048),
        backend_verified_max=4096,
    )
    assert (tighter.effective_limit, tighter.limit_source) == (4096, "backend.verified_max")


def test_exceeded_rows_are_counted() -> None:
    ctx = validate_context(
        scan_result([100, 5000, 3000, 4097, 200]),
        model_declared_max=4096,
        tokenizer=None,
        backend_verified_max=None,
    )
    assert ctx.status == "exceeded" and ctx.exceeded_rows == 2 and ctx.max_observed_length == 5000


def test_grpo_budget_is_added_to_the_longest_prompt() -> None:
    prompts = scan_result([18, 61, 268], objective=Objective.GRPO)
    ok = validate_context(
        prompts,
        model_declared_max=262_144,
        tokenizer=None,
        backend_verified_max=None,
        extra_tokens=8192,
    )
    assert ok.status == "ok" and ok.max_observed_length == 268 + 8192
    over = validate_context(
        prompts,
        model_declared_max=4096,
        tokenizer=None,
        backend_verified_max=None,
        extra_tokens=4096,
    )
    assert over.status == "exceeded" and over.exceeded_rows == 3


def test_exceeded_count_is_exact_on_power_of_two_thresholds() -> None:
    lengths = list(range(1, 1001))
    for limit, budget in ((640, 0), (4096, 3456), (1024, 512)):
        ctx = validate_context(
            scan_result(lengths, objective=Objective.GRPO),
            model_declared_max=limit,
            tokenizer=None,
            backend_verified_max=None,
            extra_tokens=budget,
        )
        assert ctx.exceeded_rows == sum(1 for n in lengths if n + budget > limit)


def test_partial_or_failed_scans_cannot_prove_ok() -> None:
    partial = scan_result([100], coverage=ScanCoverage.PARTIAL)
    failed = scan_result([100], rows_failed=1)
    for result in (partial, failed):
        ctx = validate_context(
            result, model_declared_max=4096, tokenizer=None, backend_verified_max=None
        )
        assert ctx.status == "unknown"
    over = validate_context(
        scan_result([9000], coverage=ScanCoverage.PARTIAL),
        model_declared_max=4096,
        tokenizer=None,
        backend_verified_max=None,
    )
    assert over.status == "exceeded"


# ---------------------------------------------------------------- audit_preservation


def audit(result: DatasetScanResult, **overrides):
    context = validate_context(
        result,
        model_declared_max=overrides.pop("limit", 262_144),
        tokenizer=None,
        backend_verified_max=None,
    )
    kwargs = {
        "context": context,
        "batch_plan": plan(),
        "packing": False,
        "scope": ScopeConfig(),
        "template_content_loss_rows": 0,
    }
    kwargs.update(overrides)
    return audit_preservation(result, **kwargs)


def checks(result) -> dict:
    return {c.name: c.passed for c in result.checks}


def test_clean_scan_is_verified_with_all_eight_checks() -> None:
    result = audit(scan_result([10, 20, 30]))
    assert result.status is DataPreservation.VERIFIED and result.violations == []
    assert [c.name for c in result.checks] == list(PreservationCheckName)
    passed = checks(result)
    assert passed[C.PACKING_SAFE] is None and passed[C.EVAL_SCOPE_CONSISTENT] is None  # n/a
    assert all(passed[n] for n in passed if n not in (C.PACKING_SAFE, C.EVAL_SCOPE_CONSISTENT))


def test_missing_batch_plan_is_pending() -> None:
    result = audit(scan_result([10]), batch_plan=None)
    assert result.status is DataPreservation.PENDING
    assert checks(result)[C.LAST_BATCH_INCLUDED] is None


def test_partial_scan_is_unknown_not_verified() -> None:
    result = audit(scan_result([10], coverage=ScanCoverage.PARTIAL))
    assert result.status is DataPreservation.UNKNOWN
    assert checks(result)[C.FULL_READ] is False
    assert [i.code for i in result.violations] == [ErrorCode.SCAN_PARTIAL]


def test_failed_rows_withhold_verification() -> None:
    result = audit(scan_result([10, 20], rows_failed=1))
    assert result.status is DataPreservation.UNKNOWN
    assert checks(result)[C.FULL_READ] is False
    assert ErrorCode.SCAN_FAILED_ROWS in [i.code for i in result.violations]


def test_context_exceeded_is_a_violation_without_truncation_advice() -> None:
    result = audit(scan_result([100, 9000]), limit=4096)
    assert result.status is DataPreservation.VIOLATED
    issue = next(i for i in result.violations if i.code is ErrorCode.CONTEXT_EXCEEDED)
    assert "하지 않으며" in issue.user_message and "GPU 메모리를 늘려도" in issue.user_message
    assert issue.details["effective_limit"] == 4096


def test_sampler_dropping_rows_is_a_violation() -> None:
    result = audit(scan_result([10]), batch_plan=plan(covers=False, dropped=1))
    assert result.status is DataPreservation.VIOLATED
    assert checks(result)[C.LAST_BATCH_INCLUDED] is False
    assert result.violations[0].code is ErrorCode.SAMPLER_DROPS_ROWS


def test_packing_is_unsupported_in_strict_mode() -> None:
    result = audit(scan_result([10]), packing=True)
    assert result.status is DataPreservation.VIOLATED
    passed = checks(result)
    assert passed[C.NO_SPLIT_OR_CONCAT] is False and passed[C.PACKING_SAFE] is False
    assert passed[C.NO_LENGTH_DROP] is False
    assert [i.code for i in result.violations] == [ErrorCode.DATA_PRESERVATION_VIOLATION]


def test_template_content_loss_is_a_violation() -> None:
    result = audit(scan_result([10]), template_content_loss_rows=3)
    assert result.status is DataPreservation.VIOLATED
    assert result.violations[0].code is ErrorCode.TEMPLATE_CONTENT_LOSS


def test_eval_scope_without_eval_scan_is_pending() -> None:
    result = audit(scan_result([10]), scope=ScopeConfig(include_evaluation=True))
    assert result.status is DataPreservation.PENDING
    assert checks(result)[C.EVAL_SCOPE_CONSISTENT] is None


def test_unknown_context_limit_is_unknown() -> None:
    result = audit_preservation(
        scan_result([10]),
        context=validate_context(
            scan_result([10]), model_declared_max=None, tokenizer=None, backend_verified_max=None
        ),
        batch_plan=plan(),
        packing=False,
        scope=ScopeConfig(),
        template_content_loss_rows=0,
    )
    assert result.status is DataPreservation.UNKNOWN
    assert checks(result)[C.CONTEXT_WITHIN_LIMIT] is None
