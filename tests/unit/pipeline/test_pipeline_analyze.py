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
    memory_estimate,
    raising,
    resolved_config,
    scenario,
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
    ErrorCode,
    EvidenceLevel,
    HardwareFit,
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
    assert {w.details.get("reason") for w in result.warnings} == {"not_implemented"}
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
    assert stopped.code is ErrorCode.INTERNAL_ERROR
    assert stopped.stage is Stage.TOKENIZING
    assert stopped.affected_component == "scan.full_scan"
    assert stopped.details["reason"] == "not_implemented"
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


def test_explicit_mapping_skips_the_ambiguity_question(
    ctx: CachingContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeModules(inspect_dataset=lambda *a, **k: dataset_inspection(mapping_ambiguous=True)).install(
        monkeypatch
    )
    result = analyze(example_request(), ctx)  # the example request carries a mapping
    assert terminal_status(result) is JobStatus.COMPLETED


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
    def by_budget(scan, **kwargs):
        return context_validation("exceeded" if kwargs["extra_tokens"] > 3000 else "ok")

    FakeModules(validate_context=by_budget).install(monkeypatch)
    result = analyze(example_request(), ctx)
    exceeded = [e for e in result.errors if e.code is ErrorCode.CONTEXT_EXCEEDED]
    assert exceeded and exceeded[0].details["budgets"] == [4096, 8192]
    assert result.context_validation.status == "exceeded"  # the largest budget


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
