"""Pipeline orchestration with fake modules (docs/architecture.md §3, plan §14.2, §16.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipeline_fakes import (
    CachingContext,
    FakeContext,
    FakeModules,
    GiB,
    compat_report,
    context_validation,
    dataset_inspection,
    example_request,
    issue,
    mapping_aware,
    memory_estimate,
    raising,
    resolved_config,
    scenario,
    string_columns,
    tokenizer_manifest,
)

from vramforge_estimator import pipeline
from vramforge_estimator.errors import CancelledError
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.pipeline import CachedScan, analyze, terminal_status
from vramforge_estimator.schemas import (
    ColumnMapping,
    DataPreservation,
    DatasetColumn,
    DatasetFormat,
    DatasetSplitInfo,
    EmptySystemPolicy,
    ErrorCode,
    EvidenceLevel,
    HardwareFit,
    HardwareFitResult,
    JobStatus,
    ScanCoverage,
    Stage,
    TrainingReadiness,
)


@pytest.fixture
def ctx(tmp_path: Path) -> CachingContext:
    artifact_dir = tmp_path / "artifacts" / "owner" / ("a" * 32)
    artifact_dir.mkdir(parents=True)
    return CachingContext(artifact_dir=artifact_dir)


def test_happy_path_runs_every_stage_and_fills_the_result(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules().install(monkeypatch)
    result = analyze(example_request(), ctx)

    assert ctx.stages == [
        JobStatus.RESOLVING,
        JobStatus.INSPECTING,
        JobStatus.TOKENIZING,
        JobStatus.VALIDATING_DATA,
        JobStatus.PLANNING_BATCHES,
        JobStatus.ESTIMATING,
    ]
    assert terminal_status(result) is JobStatus.COMPLETED
    assert result.analysis_id == "a" * 32
    assert result.analysis_fingerprint.startswith("req_")
    assert result.source_manifests.model is not None
    assert result.source_manifests.dataset is not None
    assert result.model_inventory_summary is not None
    assert result.tokenizer_manifest is not None
    assert result.dataset_scan is not None and result.dataset_scan.split == "train"
    assert result.dataset_scan.preprocess_key.startswith("pre_")
    assert result.batch_plan is not None and result.memory is not None
    assert result.resolved_config is not None and result.compatibility_report is not None
    assert result.profile_id == "analytic/qwen3_5_hybrid-trl-1.14.1"
    assert result.dependency_lock_digest == "lock_abc"
    assert result.status.scan_coverage is ScanCoverage.COMPLETE
    assert result.status.data_preservation is DataPreservation.VERIFIED
    assert result.status.training_readiness is TrainingReadiness.READY
    assert result.status.estimate_evidence is EvidenceLevel.ANALYTIC
    assert result.status.hardware_fit is HardwareFit.NOT_EVALUATED
    assert result.measurement_scope.measured is False
    assert result.errors == []
    # host RAM / disk / analysis RAM are not implemented by the fakes: honest warnings only
    assert result.host_ram_estimate is None
    assert {w.code for w in result.warnings} == {ErrorCode.NOT_IMPLEMENTED}
    assert {w.affected_component for w in result.warnings} == {
        "memory.estimate_host_ram",
        "memory.estimate_disk",
        "memory.estimate_analysis_ram",
    }
    assert fakes.last_estimate_kwargs["readiness"] is TrainingReadiness.READY

    # artifacts for recompute: inventory, length artifact, manifest
    assert (ctx.artifact_dir / pipeline.INVENTORY_FILE).is_file()
    manifest = json.loads((ctx.artifact_dir / pipeline.ARTIFACTS_FILE).read_text())
    assert manifest["lengths"] == "lengths"
    assert manifest["preprocess_key"] == result.dataset_scan.preprocess_key
    # partial results were published while running; the complete scan was cached
    assert ctx.partials and ctx.partials[0].source_manifests.model is not None
    assert list(ctx.cache) == [result.dataset_scan.preprocess_key]


def test_preprocess_key_changes_with_template_options(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules().install(monkeypatch)
    plain = analyze(example_request(), ctx).dataset_scan
    FakeModules(
        resolve=lambda req, inv, tok: (
            resolved_config(template_kwargs={"enable_thinking": False}),
            compat_report(),
        )
    ).install(monkeypatch)
    thinking = analyze(example_request(), ctx).dataset_scan
    assert plain is not None and thinking is not None
    assert plain.preprocess_key != thinking.preprocess_key


def test_not_implemented_stage_halts_with_a_partial_result(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules(full_scan=raising(NotImplementedError()))
    fakes.install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    stopped = pipeline.halting_issue(result)
    assert stopped is not None
    assert stopped.code is ErrorCode.NOT_IMPLEMENTED
    assert stopped.stage is Stage.TOKENIZING
    assert stopped.affected_component == "scan.full_scan"
    assert "이후 단계를 진행하지 않았습니다" in stopped.user_message
    assert result.dataset_scan is None and result.batch_plan is None and result.memory is None
    assert result.model_inventory_summary is not None  # verified information is kept
    assert result.status.scan_coverage is ScanCoverage.FAILED
    assert result.status.estimate_evidence is EvidenceLevel.METADATA_ONLY
    assert "plan_batches" not in fakes.calls and "estimate_memory" not in fakes.calls


def test_nothing_verified_is_failed(ctx: CachingContext, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModules(
        resolve_model=raising(issue(ErrorCode.SOURCE_NOT_FOUND, "모델을 찾을 수 없습니다.")),
        resolve_dataset=raising(issue(ErrorCode.SOURCE_ACCESS_DENIED, "접근 권한이 없습니다.")),
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.FAILED
    codes = [e.code for e in result.errors]
    assert codes == [ErrorCode.SOURCE_NOT_FOUND, ErrorCode.SOURCE_ACCESS_DENIED]
    assert pipeline.halting_issue(result).code is ErrorCode.SOURCE_ACCESS_DENIED
    assert result.status.scan_coverage is ScanCoverage.NOT_STARTED
    assert result.status.estimate_evidence is None


def test_inspection_failures_are_all_reported(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules(
        load_tokenizer=raising(issue(ErrorCode.TOKENIZER_REQUIRED, "tokenizer가 없습니다.")),
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    assert pipeline.halting_issue(result).code is ErrorCode.TOKENIZER_REQUIRED
    # the compatibility report is still produced from the inventory
    assert result.compatibility_report is not None
    assert "full_scan" not in fakes.calls


def test_ambiguous_mapping_needs_input(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        ColumnMapping(prompt="question", chosen="chosen", rejected="rejected"),
        ColumnMapping(prompt="system", chosen="chosen", rejected="rejected"),
    ]
    fakes = FakeModules(
        inspect_dataset=lambda *a, **k: dataset_inspection(
            mapping_ambiguous=True, mapping_candidates=candidates
        )
    ).install(monkeypatch)
    result = analyze(example_request(**{"dataset.mapping": None}), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    assert result.needs_input is not None
    (choice,) = result.needs_input.choices
    assert choice.field == "dataset.mapping"
    assert choice.options == [
        "prompt=question, chosen=chosen, rejected=rejected",
        "prompt=system, chosen=chosen, rejected=rejected",
    ]
    assert result.needs_input.mapping_candidates == candidates
    assert "question" in result.needs_input.columns
    assert [e.code for e in result.errors] == [ErrorCode.COLUMN_MAPPING_REQUIRED]
    assert "full_scan" not in fakes.calls


# Two prompt candidates ("prompt", "question"): ambiguous for GRPO/DPO until a hint picks one.
AMBIGUOUS_COLUMNS = ("system", "question", "prompt", "chosen", "rejected")


def _scanned_mapping(fakes: FakeModules) -> ColumnMapping:
    (adapter,) = fakes.adapters
    return adapter.mapping


def test_role_hints_resolve_an_ambiguous_mapping(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format=auto with role hints goes through the inspector's suggestion computed with those
    hints: the hints settle the ambiguity and the scan uses the resolved (explicit) mapping."""
    inspect = mapping_aware(string_columns(*AMBIGUOUS_COLUMNS))
    fakes = FakeModules(inspect_dataset=inspect).install(monkeypatch)
    asked = analyze(example_request(**{"dataset.mapping": None}), ctx)
    assert terminal_status(asked) is JobStatus.NEEDS_INPUT
    (choice,) = asked.needs_input.choices
    assert choice.field == "dataset.mapping" and len(choice.options) == 2
    assert [e.code for e in asked.errors] == [ErrorCode.COLUMN_MAPPING_REQUIRED]  # asked once
    assert "get_adapter" not in fakes.calls

    fakes = FakeModules(inspect_dataset=inspect).install(monkeypatch)
    hinted = analyze(example_request(**{"dataset.mapping": {"prompt": "question"}}), ctx)
    assert terminal_status(hinted) is JobStatus.COMPLETED
    applied = _scanned_mapping(fakes)
    assert applied.format is DatasetFormat.PREFERENCE
    assert (applied.system, applied.prompt, applied.chosen, applied.rejected) == (
        "system",
        "question",
        "chosen",
        "rejected",
    )
    assert hinted.dataset_scan.mapping_applied == applied


def test_policy_only_mapping_uses_the_detected_columns(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapping that only carries the empty-system policy is detection plus that policy, not a
    mapping without columns (which would fail every row)."""
    fakes = FakeModules().install(monkeypatch)
    request = example_request(**{"dataset.mapping": {"empty_system_policy": "keep"}})
    result = analyze(request, ctx)
    assert terminal_status(result) is JobStatus.COMPLETED
    applied = _scanned_mapping(fakes)
    assert applied.prompt == "question" and applied.chosen == "chosen"
    assert applied.empty_system_policy is EmptySystemPolicy.KEEP
    assert fakes.adapters[0].empty_system_policy is EmptySystemPolicy.KEEP


def test_explicit_mapping_is_scanned_as_given(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = {
        "format": "preference",
        "system": "system",
        "prompt": "question",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    fakes = FakeModules(inspect_dataset=mapping_aware(string_columns(*AMBIGUOUS_COLUMNS))).install(
        monkeypatch
    )
    result = analyze(example_request(**{"dataset.mapping": explicit}), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED
    assert _scanned_mapping(fakes) == ColumnMapping.model_validate(explicit)


@pytest.mark.parametrize(
    ("mapping", "fragment"),
    [
        (
            {"format": "preference", "prompt": "nope", "chosen": "chosen", "rejected": "rejected"},
            "nope",
        ),
        ({"format": "messages", "messages": "question"}, "question→messages"),
    ],
)
def test_explicit_mapping_that_cannot_match_the_columns_needs_input(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, mapping: dict, fragment: str
) -> None:
    """A missing column or a column of the wrong kind would fail every row: ask instead."""
    fakes = FakeModules().install(monkeypatch)
    result = analyze(example_request(**{"dataset.mapping": mapping}), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    (choice,) = result.needs_input.choices
    assert choice.field == "dataset.mapping" and fragment in choice.reason
    assert ColumnMapping.model_validate(mapping) not in result.needs_input.mapping_candidates
    assert result.needs_input.mapping_candidates  # the detected mappings are offered
    assert {e.code for e in result.errors} == {ErrorCode.COLUMN_MAPPING_REQUIRED}
    assert [e.user_message for e in result.errors].count(choice.reason) == 1  # not repeated
    assert "get_adapter" not in fakes.calls and "full_scan" not in fakes.calls


def test_explicit_mapping_is_checked_even_when_the_inspector_did_not(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(inspect_dataset=lambda *a, **k: dataset_inspection()).install(monkeypatch)
    request = example_request(**{"dataset.mapping": {"format": "text", "text": "missing_column"}})
    result = analyze(request, ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    assert "missing_column" in result.needs_input.choices[0].reason


def test_hint_contradicting_the_suggestion_needs_input(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(inspect_dataset=lambda *a, **k: dataset_inspection()).install(monkeypatch)
    result = analyze(example_request(**{"dataset.mapping": {"prompt": "system"}}), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    (choice,) = result.needs_input.choices
    assert "prompt" in choice.reason
    assert choice.suggested == "system=system, prompt=question, chosen=chosen, rejected=rejected"


def test_role_kinds_match_the_inspector() -> None:
    from vramforge_estimator.inspection.dataset_mapping import ROLE_KINDS

    assert pipeline._ROLE_KINDS == ROLE_KINDS


def test_missing_split_and_multiple_configs_need_input(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(
        inspect_dataset=lambda *a, **k: dataset_inspection(
            configs=["a", "b"],
            selected_config=None,
            splits=[DatasetSplitInfo(name="x"), DatasetSplitInfo(name="y")],
            selected_split=None,
        )
    ).install(monkeypatch)
    result = analyze(example_request(**{"dataset.split": None}), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    fields = [c.field for c in result.needs_input.choices]
    assert fields == ["dataset.config", "dataset.split"]
    assert result.needs_input.choices[1].options == ["x", "y"]


def test_requested_split_must_exist(ctx: CachingContext, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModules().install(monkeypatch)
    result = analyze(example_request(**{"dataset.split": "validation"}), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    assert result.needs_input.choices[0].field == "dataset.split"
    assert result.needs_input.choices[0].suggested == "train"


def test_inspector_choice_errors_become_needs_input(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(
        inspect_dataset=raising(
            issue(ErrorCode.DATASET_CONFIG_REQUIRED, "config를 선택해야 합니다.")
        )
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    assert result.needs_input.choices[0].field == "dataset.config"


def test_conversational_data_without_template_is_template_required(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules(
        chat_template=False,
        inspect_dataset=lambda *a, **k: dataset_inspection(
            columns=[DatasetColumn(name="messages", dtype="list", kind="messages")],
            detected_format=DatasetFormat.MESSAGES,
            suggested_mapping=ColumnMapping(format=DatasetFormat.MESSAGES, messages="messages"),
        ),
    ).install(monkeypatch)
    request = example_request(
        **{
            "training.objective": "sft",
            "dataset.mapping": {"format": "messages", "messages": "messages"},
        }
    )
    result = analyze(request, ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    stopped = pipeline.halting_issue(result)
    assert stopped.code is ErrorCode.TEMPLATE_REQUIRED
    assert stopped.stage is Stage.INSPECTING
    assert "get_adapter" not in fakes.calls and "full_scan" not in fakes.calls


def test_plain_string_columns_do_not_need_a_template(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(chat_template=False).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED


def test_unsupported_architecture_is_metadata_only(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = compat_report(readiness=TrainingReadiness.UNSUPPORTED, grade=None, adapter=None)
    fakes = FakeModules(resolve=lambda req, inv, tok: (None, report)).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED  # a result, not a failure
    assert "full_scan" in fakes.calls  # data statistics are still produced
    assert "plan_batches" not in fakes.calls and "estimate_memory" not in fakes.calls
    assert result.memory is None and result.batch_plan is None
    assert result.status.estimate_evidence is EvidenceLevel.METADATA_ONLY
    assert result.status.training_readiness is TrainingReadiness.UNSUPPORTED
    assert ErrorCode.UNSUPPORTED_ARCHITECTURE in {e.code for e in result.errors}


def test_partial_full_scan_never_reaches_estimation(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules(scan_coverage=ScanCoverage.PARTIAL).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    assert pipeline.halting_issue(result).code is ErrorCode.SCAN_PARTIAL
    assert result.status.scan_coverage is ScanCoverage.PARTIAL
    assert result.dataset_scan is not None and result.context_validation is not None
    assert "plan_batches" not in fakes.calls
    assert ctx.cache == {}  # partial scans are never cached


def test_cancellation_during_scan(ctx: CachingContext, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModules(full_scan=raising(CancelledError(Stage.TOKENIZING))).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.CANCELLED
    assert result.status.scan_coverage is ScanCoverage.PARTIAL


def test_cancellation_between_stages(ctx: CachingContext, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = FakeModules().install(monkeypatch)
    ctx.cancel_after_stage = JobStatus.INSPECTING
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.CANCELLED
    assert "full_scan" not in fakes.calls


def test_scan_cache_hit_reuses_the_artifact(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeModules().install(monkeypatch)
    first = analyze(example_request(), ctx)
    key = first.dataset_scan.preprocess_key
    cached = ctx.cache[key]

    other_dir = tmp_path / "artifacts" / "owner" / ("b" * 32)
    other_dir.mkdir(parents=True)
    second_ctx = CachingContext(artifact_dir=other_dir, analysis_id="b" * 32, cache={key: cached})
    fakes = FakeModules().install(monkeypatch)
    second = analyze(example_request(), second_ctx)
    assert "full_scan" not in fakes.calls and "open_rows" not in fakes.calls
    assert terminal_status(second) is JobStatus.COMPLETED
    adopted = other_dir / "lengths"
    assert (adopted / "part-00000.parquet").is_file()
    manifest = json.loads((other_dir / pipeline.ARTIFACTS_FILE).read_text())
    assert manifest["lengths"] == "lengths"


def test_stale_cache_entries_are_ignored(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fakes = FakeModules().install(monkeypatch)
    first = analyze(example_request(), ctx)
    key = first.dataset_scan.preprocess_key
    ctx.cache[key] = CachedScan(result=first.dataset_scan, artifact_path=tmp_path / "missing")
    fakes.calls.clear()
    analyze(example_request(), ctx)
    assert "full_scan" in fakes.calls


def test_context_exceeded_for_some_grpo_budgets(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan §8.3: a GRPO scenario's context is prompt + its budget. Unchosen candidate budgets
    that exceed the context withhold only their own fit; they do not make the data "violated"
    or the whole setup unsupported."""
    from pipeline_fakes import FakeModules as Modules

    from vramforge_estimator.schemas import PreservationAudit

    def by_budget(scan, **kwargs):
        return context_validation("exceeded" if kwargs["extra_tokens"] > 3000 else "ok")

    def audit(scan, *, context, batch_plan, **kwargs):  # the real audit fails on exceeded
        if context.status == "exceeded":
            return PreservationAudit(status=DataPreservation.VIOLATED)
        status = DataPreservation.PENDING if batch_plan is None else DataPreservation.VERIFIED
        return PreservationAudit(status=status)

    scenarios = [
        scenario(f"budget_{b}", fit=HardwareFit.EXPECTED_FIT).model_copy(
            update={"params": {"completion_budget": b}}
        )
        for b in (1024, 2048, 4096, 8192)
    ]
    Modules(
        validate_context=by_budget,
        audit_preservation=audit,
        estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary=None),
    ).install(monkeypatch)
    request = example_request(**{"hardware": {"mode": "custom", "device_total_bytes": 80 * GiB}})
    result = analyze(request, ctx)
    (exceeded,) = [w for w in result.warnings if w.code is ErrorCode.CONTEXT_EXCEEDED]
    assert exceeded.details["budgets"] == [4096, 8192]
    assert result.context_validation.status == "ok"  # the smallest offered budget fits
    assert result.status.data_preservation is DataPreservation.VERIFIED
    assert result.status.training_readiness is TrainingReadiness.READY
    fits = {s.scenario_id: s.hardware_fit.status for s in result.memory.scenarios}
    assert fits == {
        "budget_1024": HardwareFit.EXPECTED_FIT,
        "budget_2048": HardwareFit.EXPECTED_FIT,
        "budget_4096": HardwareFit.UNKNOWN,
        "budget_8192": HardwareFit.UNKNOWN,
    }
    assert result.status.hardware_fit is HardwareFit.UNKNOWN  # worst over the scenarios

    # an explicitly chosen budget that exceeds the context is a real violation
    explicit = analyze(example_request(**{"grpo.completion_budget": 4096}), ctx)
    assert explicit.context_validation.status == "exceeded"
    assert explicit.status.data_preservation is DataPreservation.VIOLATED
    assert explicit.status.training_readiness is TrainingReadiness.UNSUPPORTED


def test_context_exceeded_for_every_grpo_budget_is_an_error(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(validate_context=lambda scan, **k: context_validation("exceeded")).install(
        monkeypatch
    )
    result = analyze(example_request(), ctx)
    (exceeded,) = [e for e in result.errors if e.code is ErrorCode.CONTEXT_EXCEEDED]
    assert exceeded.details["budgets"] == [1024, 2048, 4096, 8192]
    assert result.context_validation.status == "exceeded"
    assert result.status.training_readiness is TrainingReadiness.UNSUPPORTED


def test_hardware_fit_uses_primary_or_worst_scenario(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenarios = [
        scenario("budget_1024", fit=HardwareFit.EXPECTED_FIT),
        scenario("budget_8192", high=40 * GiB, fit=HardwareFit.EXCEEDS),
    ]
    request = example_request(**{"hardware": {"mode": "custom", "device_total_bytes": 24 * GiB}})
    FakeModules(estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary=None)).install(
        monkeypatch
    )
    worst = analyze(request, ctx)
    assert worst.status.hardware_fit is HardwareFit.EXCEEDS
    assert worst.hardware_fit.status is HardwareFit.EXCEEDS

    FakeModules(
        estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary="budget_1024")
    ).install(monkeypatch)
    primary = analyze(request, ctx)
    assert primary.status.hardware_fit is HardwareFit.EXPECTED_FIT


def test_sample_scan_withholds_the_fit_verdict(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = example_request(
        **{
            "dataset.scan_mode": "sample",
            "dataset.sample_rows": 2,
            "hardware": {"mode": "custom", "device_total_bytes": 80 * GiB},
        }
    )
    seen_limits = []

    fakes = FakeModules(
        scan_coverage=ScanCoverage.PARTIAL,
        estimate_memory=lambda *a, **k: memory_estimate(
            [scenario("budget_1024", fit=HardwareFit.EXPECTED_FIT)]
        ),
    )

    original = fakes._full_scan

    def capture(stream, adapter, scan_ctx, **kwargs):
        seen_limits.append(scan_ctx.limits.max_rows)
        return original(stream, adapter, scan_ctx, **kwargs)

    fakes.full_scan = capture
    fakes.install(monkeypatch)
    result = analyze(request, ctx)
    assert seen_limits == [2]
    assert terminal_status(result) is JobStatus.COMPLETED
    assert result.status.scan_coverage is ScanCoverage.PARTIAL
    assert result.status.hardware_fit is HardwareFit.UNKNOWN
    assert result.status.training_readiness is TrainingReadiness.CONDITIONAL
    (card,) = result.memory.scenarios
    assert card.hardware_fit.reason == "scan_incomplete"
    assert card.hardware_fit.utilization_ratio is None
    assert result.hardware_fit.reason == "scan_incomplete"
    assert result.hardware_fit.message == pipeline.SCAN_INCOMPLETE_FIT_MESSAGE
    reasons = [w.details.get("reason") for w in result.warnings]
    assert reasons.count("scan_incomplete") == 1


def test_compatibility_blockers_and_warnings_are_surfaced(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    reward = issue(ErrorCode.GRPO_REWARD_UNSPECIFIED, "reward 미지정")
    report = compat_report(
        readiness=TrainingReadiness.CONDITIONAL,
        warnings=[reward.model_copy(update={"severity": "warning"})],
    )
    FakeModules(resolve=lambda req, inv, tok: (resolved_config(), report)).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED
    assert result.status.training_readiness is TrainingReadiness.CONDITIONAL
    assert ErrorCode.GRPO_REWARD_UNSPECIFIED in {w.code for w in result.warnings}
    assert any(w.code is ErrorCode.GRPO_REWARD_UNSPECIFIED for w in ctx.warnings)


def test_plain_context_without_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModules().install(monkeypatch)
    plain = FakeContext(artifact_dir=tmp_path)
    result = analyze(example_request(), plain)
    assert terminal_status(result) is JobStatus.COMPLETED


def test_tokenizer_manifest_without_template_is_reported(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(
        load_tokenizer=lambda s, a: TokenizerHandle(
            tokenizer=object(), manifest=tokenizer_manifest(chat_template=False)
        )
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert result.tokenizer_manifest.chat_template_present is False


def test_auto_selected_split_is_reported(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules().install(monkeypatch)
    explicit = analyze(example_request(), ctx)
    assert explicit.dataset_scan.split_auto_selected is False
    auto = analyze(example_request(**{"dataset.split": None}), ctx)  # cache hit path
    assert auto.dataset_scan.split == "train"
    assert auto.dataset_scan.split_auto_selected is True


@pytest.mark.parametrize(
    "stop",
    ["estimate_not_implemented", "estimate_error", "cancelled_before_estimate", "needs_input"],
)
def test_runs_that_stop_early_are_never_ready(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """A halted, cancelled or NEEDS_INPUT run did not verify the setup: the readiness axis (which
    gates the trainer-config export) must not say ready even when the profile itself is ready."""
    request = example_request()
    if stop == "estimate_not_implemented":
        FakeModules(estimate_memory=raising(NotImplementedError())).install(monkeypatch)
    elif stop == "estimate_error":
        failure = issue(ErrorCode.UNKNOWN_MEMORY_COMPONENT, "ledger를 만들 수 없습니다.")
        FakeModules(estimate_memory=raising(failure)).install(monkeypatch)
    elif stop == "cancelled_before_estimate":
        FakeModules().install(monkeypatch)
        ctx.cancel_after_stage = JobStatus.PLANNING_BATCHES
    else:
        FakeModules(
            inspect_dataset=lambda *a, **k: dataset_inspection(mapping_ambiguous=True)
        ).install(monkeypatch)
        request = example_request(**{"dataset.mapping": None})
    result = analyze(request, ctx)
    assert terminal_status(result) is not JobStatus.COMPLETED
    assert result.compatibility_report.readiness is TrainingReadiness.READY
    assert result.status.training_readiness is TrainingReadiness.CONDITIONAL
    assert result.memory is None


def test_unverified_lengths_withhold_every_scenario_fit(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rows that failed to tokenize leave the longest length unknown: no scenario may claim a
    fit (plan §10.3), not only the summary."""
    from pipeline_fakes import scan_result

    from vramforge_estimator.scan import ScanOutcome

    scenarios = [
        scenario("budget_1024", fit=HardwareFit.EXPECTED_FIT),
        scenario("budget_2048", fit=HardwareFit.LOW_MARGIN),
        scenario("budget_8192", high=90 * GiB, fit=HardwareFit.EXCEEDS),
    ]

    def failed_rows_scan(stream, adapter, scan_ctx, **kwargs):
        lengths = scan_ctx.artifact_dir / "lengths"
        lengths.mkdir(parents=True, exist_ok=True)
        (lengths / "part-00000.parquet").write_bytes(b"PAR1fakePAR1")
        result = scan_result(preprocess_key=kwargs["preprocess_key"], rows_failed=1)
        return ScanOutcome(result=result, artifact_path=lengths)

    FakeModules(
        full_scan=failed_rows_scan,
        estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary=None),
    ).install(monkeypatch)
    request = example_request(**{"hardware": {"mode": "custom", "device_total_bytes": 80 * GiB}})
    result = analyze(request, ctx)
    fits = [s.hardware_fit.status for s in result.memory.scenarios]
    assert fits == [HardwareFit.UNKNOWN, HardwareFit.UNKNOWN, HardwareFit.EXCEEDS]
    assert result.hardware_fit.status is HardwareFit.EXCEEDS  # the worst scenario still shows
    assert [w.details.get("reason") for w in result.warnings].count("lengths_unverified") == 1

    FakeModules(estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary=None)).install(
        monkeypatch
    )
    fresh = FakeContext(artifact_dir=tmp_path / "fresh")  # no cached failed-rows scan
    verified = analyze(request, fresh)  # complete scan without failures keeps the verdicts
    assert [s.hardware_fit.status for s in verified.memory.scenarios] == [
        HardwareFit.EXPECTED_FIT,
        HardwareFit.LOW_MARGIN,
        HardwareFit.EXCEEDS,
    ]


def test_a_cached_scan_that_cannot_be_adopted_is_scanned_again(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The owner scan cache is optional: if its artifact vanishes while being adopted (e.g. the
    source analysis is deleted in another tab), the analysis scans instead of failing."""
    import shutil

    FakeModules().install(monkeypatch)
    first = analyze(example_request(), ctx)
    key = first.dataset_scan.preprocess_key

    other_dir = tmp_path / "artifacts" / "owner" / ("b" * 32)
    other_dir.mkdir(parents=True)
    second_ctx = CachingContext(
        artifact_dir=other_dir, analysis_id="b" * 32, cache={key: ctx.cache[key]}
    )

    def vanished(src, dst, **kwargs):
        Path(dst).mkdir(parents=True)
        (Path(dst) / "part-00007.parquet").write_bytes(b"PAR1")  # copied before the source went
        raise FileNotFoundError("source deleted during copy")

    monkeypatch.setattr(shutil, "copytree", vanished)
    fakes = FakeModules().install(monkeypatch)
    second = analyze(example_request(), second_ctx)
    assert terminal_status(second) is JobStatus.COMPLETED
    assert "full_scan" in fakes.calls  # fell back to a real scan
    assert not (other_dir / "lengths" / "part-00007.parquet").exists()
    assert (other_dir / "lengths" / "part-00000.parquet").is_file()


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.DATASET_FORMAT_UNSUPPORTED,
        ErrorCode.SOURCE_ACCESS_DENIED,
        ErrorCode.INTERNAL_ERROR,
    ],
)
def test_blocking_dataset_inspection_errors_halt_instead_of_asking(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, code: ErrorCode
) -> None:
    """An inspection error no config/split/mapping choice can fix (unreadable file, denied access)
    stops the run with that issue; it never becomes NEEDS_INPUT with nothing to choose from."""
    unreadable = issue(code, "데이터 파일을 읽을 수 없습니다.", stage=Stage.INSPECTING)
    fakes = FakeModules(
        inspect_dataset=lambda *a, **k: dataset_inspection(
            columns=[],
            detected_format=None,
            mapping_candidates=[],
            suggested_mapping=None,
            issues=[unreadable],
        )
    ).install(monkeypatch)
    result = analyze(example_request(**{"dataset.mapping": None}), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    assert result.needs_input is None
    stopped = pipeline.halting_issue(result)
    assert stopped is not None and stopped.code is code
    assert stopped.user_message == unreadable.user_message
    assert [e.code for e in result.errors] == [code]  # reported once, as the halting issue
    assert "open_rows" not in fakes.calls and "full_scan" not in fakes.calls


def test_needs_input_issues_and_warnings_do_not_halt(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vramforge_estimator.schemas import Severity

    preview_note = issue(ErrorCode.SCAN_FAILED_ROWS, "미리보기에서 읽지 못한 레코드가 있습니다.")
    preview_note = preview_note.model_copy(update={"severity": Severity.WARNING})
    FakeModules(inspect_dataset=lambda *a, **k: dataset_inspection(issues=[preview_note])).install(
        monkeypatch
    )
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED
    assert preview_note.user_message in {w.user_message for w in result.warnings}


def test_raised_needs_input_issue_keeps_its_options(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(
        inspect_dataset=raising(
            issue(
                ErrorCode.DATASET_CONFIG_REQUIRED,
                "config를 선택해야 합니다.",
                field="dataset.config",
                options=["en", "ko"],
                suggested="en",
            )
        )
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.NEEDS_INPUT
    (choice,) = result.needs_input.choices
    assert (choice.field, choice.options, choice.suggested) == (
        "dataset.config",
        ["en", "ko"],
        "en",
    )


def _scan_with(**changes):
    """A full_scan fake whose result/issues are modified (e.g. zero rows)."""
    from pipeline_fakes import FakeModules as Modules

    base = Modules()._full_scan
    extra_issues = changes.pop("issues", [])

    def scan(stream, adapter, scan_ctx, **kwargs):
        outcome = base(stream, adapter, scan_ctx, **kwargs)
        outcome.result = outcome.result.model_copy(update=changes)
        outcome.issues.extend(extra_issues)
        return outcome

    return scan


def test_empty_split_halts_before_batching(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = FakeModules(
        full_scan=_scan_with(rows_seen=0, rows_ok=0, rows_expected=0, branches=[])
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL
    stopped = pipeline.halting_issue(result)
    assert stopped.code is ErrorCode.EMPTY_DATASET
    assert stopped.stage is Stage.TOKENIZING
    assert stopped.details["split"] == "train"
    assert result.dataset_scan is not None and result.dataset_scan.rows_seen == 0
    assert result.batch_plan is None and result.memory is None
    for name in ("validate_context", "plan_batches", "estimate_memory"):
        assert name not in fakes.calls


def test_reader_reported_empty_dataset_is_the_halting_issue(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vramforge_estimator.schemas import Severity

    empty = issue(
        ErrorCode.EMPTY_DATASET, "선택한 split에 row가 없습니다.", reason="no_rows"
    ).model_copy(update={"severity": Severity.WARNING, "stage": Stage.TOKENIZING})
    fakes = FakeModules(
        full_scan=_scan_with(rows_seen=0, rows_ok=0, branches=[], issues=[empty])
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    stopped = pipeline.halting_issue(result)
    assert stopped.code is ErrorCode.EMPTY_DATASET
    assert stopped.details["reason"] == "no_rows"
    assert stopped.severity.value == "error"  # a halt is always an error
    assert "plan_batches" not in fakes.calls


def test_unreadable_split_is_not_called_empty(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero rows because the first read failed: the read error explains it, not EMPTY_DATASET."""
    read_error = issue(ErrorCode.SCAN_PARTIAL, "데이터를 읽는 중 오류가 발생했습니다.")
    FakeModules(
        full_scan=_scan_with(
            rows_seen=0,
            rows_ok=0,
            branches=[],
            coverage=ScanCoverage.FAILED,
            issues=[read_error],
        )
    ).install(monkeypatch)
    result = analyze(example_request(), ctx)
    stopped = pipeline.halting_issue(result)
    assert stopped.code is ErrorCode.SCAN_PARTIAL
    assert stopped.user_message == read_error.user_message
    assert result.status.scan_coverage is ScanCoverage.FAILED


def _fit(status: HardwareFit, reason: str) -> HardwareFitResult:
    return HardwareFitResult(
        status=status, reason=reason, message="엔진 판정", capacity_bytes=80 * GiB
    )


def test_incomplete_scan_withholds_length_dependent_verdicts_only(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan §10.3: on partial data no expected fit, low margin or peak-over-capacity verdict; the
    resident floor (and loading budget) over capacity does not depend on lengths and stays."""
    verdicts = {
        "budget_1024": _fit(HardwareFit.EXPECTED_FIT, "fits_with_margin"),
        "budget_2048": _fit(HardwareFit.LOW_MARGIN, "margin_insufficient"),
        "budget_4096": _fit(HardwareFit.EXCEEDS, "high_exceeds_capacity"),
        "budget_8192": _fit(HardwareFit.EXCEEDS, "floor_exceeds_capacity"),
    }
    scenarios = [
        scenario(sid).model_copy(update={"hardware_fit": fit}) for sid, fit in verdicts.items()
    ]
    FakeModules(
        scan_coverage=ScanCoverage.PARTIAL,
        estimate_memory=lambda *a, **k: memory_estimate(scenarios, primary=None),
    ).install(monkeypatch)
    request = example_request(
        **{
            "dataset.scan_mode": "sample",
            "dataset.sample_rows": 2,
            "hardware": {"mode": "custom", "device_total_bytes": 80 * GiB},
        }
    )
    result = analyze(request, ctx)
    fits = {
        s.scenario_id: (s.hardware_fit.status, s.hardware_fit.reason)
        for s in result.memory.scenarios
    }
    assert fits == {
        "budget_1024": (HardwareFit.UNKNOWN, "scan_incomplete"),
        "budget_2048": (HardwareFit.UNKNOWN, "scan_incomplete"),
        "budget_4096": (HardwareFit.UNKNOWN, "scan_incomplete"),
        "budget_8192": (HardwareFit.EXCEEDS, "floor_exceeds_capacity"),
    }
    assert result.hardware_fit.status is HardwareFit.EXCEEDS  # worst: the floor never fits


def test_sample_scan_that_read_every_row_still_withholds_the_fit(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(
        estimate_memory=lambda *a, **k: memory_estimate(
            [
                scenario("budget_1024").model_copy(
                    update={"hardware_fit": _fit(HardwareFit.EXPECTED_FIT, "fits_with_margin")}
                )
            ]
        )
    ).install(monkeypatch)
    request = example_request(
        **{
            "dataset.scan_mode": "sample",
            "dataset.sample_rows": 100,
            "hardware": {"mode": "custom", "device_total_bytes": 80 * GiB},
        }
    )
    result = analyze(request, ctx)
    assert result.status.scan_coverage is ScanCoverage.COMPLETE
    assert result.hardware_fit.status is HardwareFit.UNKNOWN
    assert result.hardware_fit.reason == "scan_incomplete"
    assert result.hardware_fit.message == pipeline.SAMPLE_SCAN_FIT_MESSAGE


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        ({"mode": "custom", "device_total_bytes": 80 * GiB}, "scan_incomplete"),
        ({"mode": "capacity_only"}, None),
    ],
)
def test_run_stopped_by_an_incomplete_scan_says_why_the_fit_is_unknown(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch, hardware: dict, expected: str | None
) -> None:
    FakeModules(scan_coverage=ScanCoverage.PARTIAL).install(monkeypatch)
    result = analyze(example_request(hardware=hardware), ctx)
    assert terminal_status(result) is JobStatus.PARTIAL and result.memory is None
    if expected is None:
        assert result.hardware_fit is None
        assert result.status.hardware_fit is HardwareFit.NOT_EVALUATED
    else:
        assert result.hardware_fit.status is HardwareFit.UNKNOWN
        assert result.hardware_fit.reason == expected
        assert result.status.hardware_fit is HardwareFit.UNKNOWN


def test_memory_estimate_issues_are_reported(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vramforge_estimator.schemas import Severity

    budget = issue(
        ErrorCode.LOAD_BUDGET_EXCEEDED,
        "모델 로딩 단계의 임시 메모리가 가용량에 가깝습니다.",
        stage=Stage.ESTIMATING,
    ).model_copy(update={"severity": Severity.WARNING})
    unknown = issue(ErrorCode.UNKNOWN_MEMORY_COMPONENT, "커널 workspace를 산정할 수 없습니다.")

    def estimate(*args, **kwargs):
        return memory_estimate().model_copy(update={"issues": [budget, unknown]})

    FakeModules(estimate_memory=estimate).install(monkeypatch)
    result = analyze(example_request(), ctx)
    assert terminal_status(result) is JobStatus.COMPLETED
    assert budget in result.warnings
    assert unknown in result.errors  # severity decides where it is listed
    assert budget in ctx.warnings  # streamed as a warning event while running
