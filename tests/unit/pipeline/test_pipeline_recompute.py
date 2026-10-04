"""Recompute vs re-analysis (plan §4.2, docs/architecture.md §3.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline_fakes import (
    CachingContext,
    FakeModules,
    compat_report,
    example_request,
    raising,
    resolved_config,
)

from vramforge_estimator.pipeline import analyze, reanalysis_reasons, recompute
from vramforge_estimator.schemas import (
    AnalysisResult,
    ErrorCode,
    ScanCoverage,
    TrainingReadiness,
)


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AnalysisResult, Path]:
    artifact_dir = tmp_path / "artifacts" / ("a" * 32)
    artifact_dir.mkdir(parents=True)
    FakeModules().install(monkeypatch)
    result = analyze(example_request(), CachingContext(artifact_dir=artifact_dir))
    assert result.memory is not None
    return result, artifact_dir


@pytest.mark.parametrize(
    "overrides",
    [
        {"training.lora.r": 64},
        {"training.lora.alpha": 8},
        {"training.optimizer": "paged_adamw_8bit"},
        {"training.gradient_accumulation_steps": 8},
        {"training.microbatch_per_device": 2},
        {"training.gradient_checkpointing": False},
        {"training.strategy": "lora", "training.quantization.enabled": False},
        {"grpo.completion_budget": 2048},
        {"grpo.num_generations": 8},
        {"hardware": {"mode": "custom", "device_total_bytes": 24 * 1024**3}},
        {"margin_policy": {"min_bytes": 0, "fraction": 0.3}},
        {"scope": {"include_evaluation": True, "include_checkpoint_save": True}},
        {"dataset.mapping": None},  # auto-detection would give the applied mapping again
        {"dataset.split": None},
        {"model.revision": "2367e865d009c13ac81713a2878291d33ab28177"},  # the pinned commit
    ],
)
def test_estimate_layer_changes_do_not_need_reanalysis(
    base: tuple[AnalysisResult, Path], overrides: dict
) -> None:
    result, _ = base
    assert reanalysis_reasons(result, example_request(**overrides)) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"training.objective": "dpo"}, "학습 방식"),
        ({"model.reference": "Qwen/Qwen3-8B"}, "모델이 바뀌어"),
        ({"model.revision": "main"}, "모델 revision"),
        ({"dataset.reference": "org/other"}, "데이터셋이 바뀌어"),
        ({"dataset.revision": "deadbeef"}, "데이터셋 revision"),
        ({"dataset.config": "other"}, "config"),
        ({"dataset.split": "test"}, "학습 split"),
        ({"dataset.eval_split": "test"}, "평가 split"),
        ({"dataset.scan_mode": "sample", "dataset.sample_rows": 10}, "스캔 범위"),
        ({"dataset.mapping.prompt": "system"}, "컬럼 매핑"),
        ({"dataset.mapping.empty_system_policy": "keep"}, "빈 system"),
        ({"dataset.mapping.format": "prompt_only"}, "데이터 형식"),
        ({"training.template": {"enable_thinking": False}}, "chat template"),
        ({"training.packing": True}, "packing"),
    ],
)
def test_preprocessing_layer_changes_need_reanalysis(
    base: tuple[AnalysisResult, Path], overrides: dict, fragment: str
) -> None:
    result, artifact_dir = base
    response = recompute(result, example_request(**overrides), artifact_dir)
    assert response.requires_reanalysis is True
    assert response.result is None
    assert any(fragment in reason for reason in response.reanalysis_reasons)


def test_recompute_reuses_artifacts_without_touching_sources(
    base: tuple[AnalysisResult, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result, artifact_dir = base
    forbidden = raising(AssertionError("re-downloaded or re-tokenized"))
    fakes = FakeModules(
        resolve_model=forbidden,
        resolve_dataset=forbidden,
        inspect_model=forbidden,
        load_tokenizer=forbidden,
        inspect_dataset=forbidden,
        open_rows=forbidden,
        full_scan=forbidden,
        get_adapter=forbidden,
    ).install(monkeypatch)
    request = example_request(**{"training.lora.r": 64})
    response = recompute(result, request, artifact_dir)
    assert response.requires_reanalysis is False
    new = response.result
    assert new is not None
    assert new.analysis_id == result.analysis_id
    assert new.analysis_fingerprint == response.fingerprint != result.analysis_fingerprint
    assert new.requested_config.training.lora.r == 64
    assert new.dataset_scan == result.dataset_scan
    assert new.memory is not None and new.batch_plan is not None
    assert fakes.calls.count("estimate_memory") == 1
    assert {"resolve", "load_lengths", "plan_batches"} <= set(fakes.calls)
    assert new.status.scan_coverage is ScanCoverage.COMPLETE


def test_recompute_without_artifacts_needs_reanalysis(
    base: tuple[AnalysisResult, Path], tmp_path: Path
) -> None:
    result, _ = base
    response = recompute(result, example_request(), tmp_path / "empty")
    assert response.requires_reanalysis is True
    assert response.reanalysis_reasons


def test_recompute_of_a_partial_scan_needs_reanalysis(
    base: tuple[AnalysisResult, Path],
) -> None:
    result, artifact_dir = base
    partial = result.model_copy(
        update={
            "dataset_scan": result.dataset_scan.model_copy(
                update={"coverage": ScanCoverage.PARTIAL}
            )
        }
    )
    response = recompute(partial, example_request(), artifact_dir)
    assert response.requires_reanalysis is True


def test_recompute_detects_a_changed_preprocessing_profile(
    base: tuple[AnalysisResult, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result, artifact_dir = base
    FakeModules(
        resolve=lambda req, inv, tok: (
            resolved_config(dependency_lock_digest="lock_other"),
            compat_report(),
        )
    ).install(monkeypatch)
    response = recompute(result, example_request(**{"training.lora.r": 8}), artifact_dir)
    assert response.requires_reanalysis is True
    assert "dependency lock" in response.reanalysis_reasons[0]


def test_recompute_reports_module_failures_as_a_partial_result(
    base: tuple[AnalysisResult, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result, artifact_dir = base
    FakeModules(estimate_memory=raising(NotImplementedError())).install(monkeypatch)
    response = recompute(result, example_request(**{"training.lora.r": 8}), artifact_dir)
    assert response.requires_reanalysis is False
    assert response.result is not None and response.result.memory is None
    assert response.result.errors[-1].code is ErrorCode.NOT_IMPLEMENTED


def test_recompute_can_newly_support_a_combination(
    base: tuple[AnalysisResult, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result, artifact_dir = base
    FakeModules(
        resolve=lambda req, inv, tok: (
            None,
            compat_report(readiness=TrainingReadiness.UNSUPPORTED, grade=None),
        )
    ).install(monkeypatch)
    response = recompute(result, example_request(**{"training.strategy": "full"}), artifact_dir)
    assert response.requires_reanalysis is False
    assert response.result.memory is None
    assert response.result.status.training_readiness is TrainingReadiness.UNSUPPORTED


def test_recompute_keeps_dataset_and_scan_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data did not change, so its inspection and scan findings must still be reported after a
    recompute (they are not re-run and must not silently disappear)."""
    from pipeline_fakes import dataset_inspection

    from vramforge_estimator.errors import make_issue
    from vramforge_estimator.schemas import ErrorCode, Severity, Stage

    failed_rows = make_issue(
        ErrorCode.SCAN_FAILED_ROWS,
        "3개 row를 토큰화하지 못했습니다.",
        severity=Severity.WARNING,
        stage=Stage.TOKENIZING,
    )
    unlisted = make_issue(
        ErrorCode.DATASET_FORMAT_UNSUPPORTED,
        "목록에 없는 데이터 파일이 있습니다.",
        severity=Severity.WARNING,
        stage=Stage.INSPECTING,
    )
    modules = FakeModules(inspect_dataset=lambda *a, **k: dataset_inspection(issues=[unlisted]))
    plain_scan = modules._full_scan

    def scan_with_finding(stream, adapter, ctx, **kwargs):
        outcome = plain_scan(stream, adapter, ctx, **kwargs)
        outcome.issues.append(failed_rows)
        return outcome

    modules.full_scan = scan_with_finding
    modules.install(monkeypatch)
    artifact_dir = tmp_path / "artifacts" / ("c" * 32)
    artifact_dir.mkdir(parents=True)
    base = analyze(example_request(), CachingContext(artifact_dir=artifact_dir))
    assert {failed_rows.user_message, unlisted.user_message} <= {
        w.user_message for w in base.warnings
    }

    response = recompute(base, example_request(**{"training.lora.r": 64}), artifact_dir)
    assert response.requires_reanalysis is False
    messages = [w.user_message for w in response.result.warnings]
    assert messages.count(failed_rows.user_message) == 1
    assert messages.count(unlisted.user_message) == 1


def test_recompute_needs_reanalysis_when_the_base_had_no_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan keyed without a backend profile is not the tokenization a newly resolved profile
    would run, so it cannot be reused."""
    unsupported = compat_report(readiness=TrainingReadiness.UNSUPPORTED, grade=None, adapter=None)
    FakeModules(resolve=lambda req, inv, tok: (None, unsupported)).install(monkeypatch)
    artifact_dir = tmp_path / "artifacts" / ("d" * 32)
    artifact_dir.mkdir(parents=True)
    base = analyze(example_request(), CachingContext(artifact_dir=artifact_dir))
    assert base.resolved_config is None and base.dataset_scan is not None

    FakeModules().install(monkeypatch)  # the profile now resolves
    response = recompute(base, example_request(**{"training.lora.r": 8}), artifact_dir)
    assert response.requires_reanalysis is True
    assert "backend profile" in response.reanalysis_reasons[0]
