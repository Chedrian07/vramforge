"""Exports: redaction, plan YAML, escaped Korean report and the gated trainer config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from exports_testkit import GRPO_BUDGET, build_result, fakes, grpo_resolved

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.exports import (
    export_json,
    export_plan_yaml,
    export_report_md,
    export_trainer_config,
)
from vramforge_estimator.exports.sanitize import REDACTED_PATH, REDACTED_URL, redact_text
from vramforge_estimator.exports.trainer_config import FIELDS, FORBIDDEN_FIELDS
from vramforge_estimator.schemas import (
    AnalysisResult,
    ErrorCode,
    Objective,
    RewardKind,
    Severity,
    TrainingReadiness,
)

SECRET = "hf_" + "Q" * 30


def _with_dirty_strings(result: AnalysisResult) -> AnalysisResult:
    dirty = fakes.issue(
        ErrorCode.SCAN_FAILED_ROWS,
        f"row 처리 실패: /home/alice/data/x.jsonl {SECRET} http://10.1.2.3/internal?sig=abc",
    ).model_copy(update={"severity": Severity.WARNING})
    return result.model_copy(update={"warnings": [*result.warnings, dirty]})


@pytest.fixture
def grpo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AnalysisResult:
    return build_result("grpo", tmp_path, monkeypatch)


def test_redaction_rules() -> None:
    text = redact_text(f"see /Users/x/a.json {SECRET} https://user:pw@huggingface.co/a?token=1")
    assert REDACTED_PATH in text and SECRET not in text
    assert "https://huggingface.co/a" in text and "pw@" not in text and "token=1" not in text
    assert redact_text("http://localhost:8000/x") == REDACTED_URL
    assert redact_text("text/event-stream P50/P90 local:models/x") == (
        "text/event-stream P50/P90 local:models/x"
    )


def test_json_export_round_trips_and_is_redacted(grpo: AnalysisResult) -> None:
    raw = export_json(_with_dirty_strings(grpo))
    text = raw.decode()
    assert SECRET not in text and "/home/alice" not in text and "10.1.2.3" not in text
    again = AnalysisResult.model_validate(json.loads(raw))
    assert again.analysis_id == grpo.analysis_id
    assert again.memory == grpo.memory  # numbers are untouched


def test_plan_yaml_is_reproducible_and_not_trl(grpo: AnalysisResult) -> None:
    text = export_plan_yaml(_with_dirty_strings(grpo)).decode()
    assert text.startswith("# VRAMForge resolved plan")
    assert "TRL 인자와 1:1로 대응하지 않습니다" in text
    assert SECRET not in text and "/home/alice" not in text
    data = yaml.safe_load(text)
    assert data["kind"] == "vramforge.resolved-plan"
    assert data["sources"]["model"]["resolved_revision"] == fakes.MODEL_SHA
    assert data["sources"]["dataset"]["resolved_revision"] == fakes.DATASET_SHA
    assert data["dataset_plan"]["mapping"]["prompt"] == "question"
    assert data["dataset_plan"]["truncation"].startswith("none")
    assert data["resolved"]["objective"] == "grpo"
    assert data["memory"]["scenarios"][0]["devices"][0]["scenario_high_bytes"] == 10 * 1024**3


def test_report_is_korean_escaped_and_html_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = build_result("grpo", tmp_path, monkeypatch)
    manifests = result.source_manifests.model_copy(
        update={
            "model": result.source_manifests.model.model_copy(
                update={"reference": "hf:<script>alert(1)</script>|x"}
            )
        }
    )
    evil = fakes.issue(
        ErrorCode.SCAN_FAILED_ROWS, "[click](javascript:alert(1)) <img src=x onerror=1> **bold**"
    ).model_copy(update={"severity": Severity.WARNING})
    result = result.model_copy(
        update={"source_manifests": manifests, "warnings": [*result.warnings, evil]}
    )
    text = export_report_md(_with_dirty_strings(result)).decode()
    assert text.startswith("# VRAMForge 분석 보고서")
    for section in ("## 요약", "## 상태", "## 데이터", "## 시나리오 비교", "## 단계별 피크"):
        assert section in text
    assert "<script>" not in text and "<img" not in text
    assert "&lt;script&gt;" in text
    assert "\\[click\\](javascript:alert(1))" in text  # no link can form
    assert "\\*\\*bold\\*\\*" in text
    assert "hf:&lt;script&gt;alert(1)&lt;/script&gt;\\|x" in text  # pipe cannot split a cell
    assert SECRET not in text and "/home/alice" not in text
    assert "10.0 GiB (10,737,418,240 bytes)" in text
    assert "정적 추정 결과입니다" in text


def test_report_without_estimate_says_so(tmp_path: Path, monkeypatch) -> None:
    fakes.FakeModules(full_scan=fakes.raising(NotImplementedError())).install(monkeypatch)
    artifact_dir = tmp_path / "a"
    artifact_dir.mkdir()
    from vramforge_estimator.pipeline import analyze

    result = analyze(fakes.example_request(), fakes.FakeContext(artifact_dir=artifact_dir))
    text = export_report_md(result).decode()
    assert "메모리 산정 결과 없음" in text
    assert "## 오류와 경고" in text


@pytest.mark.parametrize("objective", ["sft", "dpo", "grpo"])
def test_trainer_config_for_ready_results(
    objective: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = build_result(objective, tmp_path, monkeypatch)
    assert result.status.training_readiness is TrainingReadiness.READY
    text = export_trainer_config(result).decode()
    assert text.startswith("# VRAMForge trainer config for TRL 1.14.1")
    data = yaml.safe_load(text)
    args = data["trl"]["args"]
    obj = Objective(objective)
    assert set(args) <= FIELDS[obj]
    assert not set(args) & FORBIDDEN_FIELDS
    assert data["trainer"]["version"] == "1.14.1"
    assert data["model"]["revision"] == fakes.MODEL_SHA
    assert data["model"]["model_init_kwargs"]["dtype"] == "bfloat16"
    assert args["model_init_kwargs"]["dtype"] == "bfloat16"  # TRL loads float32 otherwise
    assert data["quantization"]["load_in_4bit"] is True
    assert data["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert data["peft"]["r"] == 16 and data["peft"]["task_type"] == "CAUSAL_LM"
    assert data["dataset"]["revision"] == fakes.DATASET_SHA
    assert data["dataset"]["remove_original_columns"] is True
    assert data["processing_class"]["class"] == "AutoTokenizer"
    assert args["dataloader_drop_last"] is False and args["max_steps"] == -1
    if obj in (Objective.SFT, Objective.DPO):
        assert "max_length" in args and args["max_length"] is None
        assert args["padding_free"] is False
    if obj is Objective.SFT:
        assert args["packing"] is False and args["eval_packing"] is False
        assert args["loss_type"] == "chunked_nll"
    if obj is Objective.DPO:
        assert args["loss_type"] == ["sigmoid"] and args["precompute_ref_log_probs"] is False
    if obj is Objective.GRPO:
        assert args["max_completion_length"] == GRPO_BUDGET
        assert "max_prompt_length" not in args and "steps_per_generation" not in args
        assert args["generation_batch_size"] == 4 and args["num_generations"] == 4
        assert data["processing_class"]["from_pretrained"]["padding_side"] == "left"
        assert data["reward"]["kind"] == "cpu_rule"
    assert "/Users/" not in text and "hf_" not in text


def test_trainer_config_is_refused_unless_ready(tmp_path: Path, monkeypatch) -> None:
    reward = fakes.issue(ErrorCode.GRPO_REWARD_UNSPECIFIED, "reward가 지정되지 않았습니다.")
    conditional = build_result(
        "grpo",
        tmp_path,
        monkeypatch,
        readiness=TrainingReadiness.CONDITIONAL,
        report_kwargs={"blockers": [reward]},
    )
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(conditional)
    issue = exc.value.issue
    assert issue.code is ErrorCode.GRPO_REWARD_UNSPECIFIED
    assert "ready일 때만" in issue.user_message
    assert issue.details["readiness"] == "conditional"
    # the plan and the report are still available for a conditional result
    assert export_plan_yaml(conditional) and export_report_md(conditional)


@pytest.mark.parametrize(
    ("resolved", "code"),
    [
        (grpo_resolved(budget_explicit=False), ErrorCode.GRPO_BUDGET_UNSPECIFIED),
        (grpo_resolved(reward=RewardKind.UNSPECIFIED), ErrorCode.GRPO_REWARD_UNSPECIFIED),
    ],
)
def test_grpo_needs_an_explicit_budget_and_reward(
    tmp_path: Path, monkeypatch, resolved, code
) -> None:
    result = build_result("grpo", tmp_path, monkeypatch, resolved_overrides={"grpo": resolved})
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(result)
    assert exc.value.issue.code is code


def test_unsupported_result_has_no_trainer_config(tmp_path: Path, monkeypatch) -> None:
    report = fakes.compat_report(readiness=TrainingReadiness.UNSUPPORTED, grade=None, adapter=None)
    fakes.FakeModules(resolve=lambda req, inv, tok: (None, report)).install(monkeypatch)
    artifact_dir = tmp_path / "u"
    artifact_dir.mkdir()
    from vramforge_estimator.pipeline import analyze

    result = analyze(fakes.example_request(), fakes.FakeContext(artifact_dir=artifact_dir))
    with pytest.raises(EstimatorError):
        export_trainer_config(result)
    assert yaml.safe_load(export_plan_yaml(result).decode())["memory"] is None


@pytest.mark.parametrize(
    ("loss_path", "loss_type"), [("trl_chunked_nll", "chunked_nll"), ("hf_ce", "nll")]
)
def test_sft_loss_path_maps_to_trl_loss_type(
    tmp_path: Path, monkeypatch, loss_path: str, loss_type: str
) -> None:
    result = build_result("sft", tmp_path, monkeypatch, resolved_overrides={"loss_path": loss_path})
    args = yaml.safe_load(export_trainer_config(result))["trl"]["args"]
    assert args["loss_type"] == loss_type


def test_unmappable_sft_loss_path_is_refused(tmp_path: Path, monkeypatch) -> None:
    result = build_result(
        "sft", tmp_path, monkeypatch, resolved_overrides={"loss_path": "liger_fused"}
    )
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(result)
    assert "loss" in exc.value.issue.user_message


def test_conditional_refusal_explains_with_report_warnings(tmp_path: Path, monkeypatch) -> None:
    reward = fakes.issue(ErrorCode.GRPO_REWARD_UNSPECIFIED, "reward가 지정되지 않았습니다.")
    result = build_result(
        "grpo",
        tmp_path,
        monkeypatch,
        readiness=TrainingReadiness.CONDITIONAL,
        report_kwargs={"warnings": [reward.model_copy(update={"severity": Severity.WARNING})]},
    )
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(result)
    assert exc.value.issue.code is ErrorCode.GRPO_REWARD_UNSPECIFIED
    assert "reward가 지정되지 않았습니다." in exc.value.issue.user_message


def test_trainer_config_is_refused_for_halted_or_running_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a run that finished every stage may produce the execution config."""
    ready = build_result("sft", tmp_path, monkeypatch)
    assert export_trainer_config(ready)

    halt = fakes.issue(ErrorCode.UNKNOWN_MEMORY_COMPONENT, "ledger를 만들 수 없습니다.")
    halted = build_result("sft", tmp_path / "halted", monkeypatch, estimate_error=halt)
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(halted)
    assert exc.value.issue.code is ErrorCode.UNKNOWN_MEMORY_COMPONENT
    assert "ledger를 만들 수 없습니다." in exc.value.issue.user_message

    # a result stored while the job is still running (or by an older build) has no estimate yet
    running = ready.model_copy(update={"memory": None})
    with pytest.raises(EstimatorError) as exc:
        export_trainer_config(running)
    assert "메모리 산정" in exc.value.issue.user_message


@pytest.mark.parametrize(
    ("strategy", "lora", "separate", "passed"),
    [
        ("frozen_base_switch", True, None, None),
        ("precomputed_log_probs", True, None, None),
        ("standalone_model", False, None, None),
        ("standalone_model", True, None, "policy"),
        ("standalone_model", False, "org/reference-model", "org/reference-model"),
    ],
)
def test_dpo_reference_setup_is_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    lora: bool,
    separate: str | None,
    passed: str | None,
) -> None:
    """TRL builds the analyzed reference itself only for ref_model=None cases; otherwise the
    launcher must pass `ref_model` or the run differs from the estimate."""
    from vramforge_estimator.schemas import ConfigResolution, DpoResolved, ReferenceStrategy

    overrides: dict = {
        "dpo": DpoResolved(
            reference_strategy=ReferenceStrategy(strategy), beta=0.1, loss_type="sigmoid"
        ),
        "resolutions": [
            ConfigResolution(
                field="dpo.reference_model",
                requested=separate,
                resolved=separate,
                reason="test",
            )
        ],
    }
    if not lora:
        overrides["lora"] = None
    result = build_result("dpo", tmp_path, monkeypatch, resolved_overrides=overrides)
    reference = yaml.safe_load(export_trainer_config(result))["reference"]
    assert reference["strategy"] == strategy
    assert reference["pass_ref_model"] is (passed is not None)
    if passed == "policy":
        assert reference["ref_model"] == {
            "id": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
            "revision": fakes.MODEL_SHA,
        }
    elif passed is not None:
        assert reference["ref_model"]["id"] == passed
    else:
        assert reference["ref_model"] is None and reference["note"].startswith("ref_model=None")


@pytest.mark.parametrize("objective", ["sft", "dpo", "grpo"])
def test_analyzed_template_options_reach_the_trainer(
    objective: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run must render the chat template with the options the analysis tokenized with."""
    kwargs = {"enable_thinking": False}
    result = build_result(
        objective, tmp_path, monkeypatch, resolved_overrides={"template_kwargs": kwargs}
    )
    data = yaml.safe_load(export_trainer_config(result))
    if objective == "grpo":
        assert data["trl"]["args"]["chat_template_kwargs"] == kwargs
        assert "chat_template_kwargs" not in data["dataset"]
    else:  # TRL 1.14.1 SFT/DPO read the options per row
        assert data["dataset"]["chat_template_kwargs"] == kwargs
        assert data["dataset"]["output_columns"][-1] == "chat_template_kwargs"
        assert "chat_template_kwargs" not in data["trl"]["args"]

    plain = build_result(objective, tmp_path / "plain", monkeypatch)
    assert "chat_template_kwargs" not in yaml.safe_load(export_trainer_config(plain))["dataset"]
