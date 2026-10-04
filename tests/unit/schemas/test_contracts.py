"""Contract tests: the plan's example request and the status/state enums (plan §12.1, §15.2, §16.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vramforge_estimator.schemas import (
    PIPELINE_STAGES,
    TERMINAL_JOB_STATUSES,
    AnalysisRequest,
    AnalysisResult,
    DataPreservation,
    EvidenceLevel,
    HardwareFit,
    JobStatus,
    Objective,
    ScanCoverage,
    StatusAxes,
    TrainingReadiness,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "requests" / "plan_example_grpo.json"


def load_example() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_plan_example_request_validates() -> None:
    req = AnalysisRequest.model_validate(load_example())
    assert req.training.objective is Objective.GRPO
    assert req.training.quantization.enabled is True
    assert req.grpo.completion_budget is None
    assert req.grpo.completion_budget_candidates == [1024, 2048, 4096, 8192]
    assert req.dataset.mapping is not None and req.dataset.mapping.prompt == "question"


def test_unknown_fields_are_rejected() -> None:
    data = load_example()
    data["training"]["max_length"] = 4096  # a truncation knob must never be silently accepted
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(data)


def test_request_round_trips_through_json() -> None:
    req = AnalysisRequest.model_validate(load_example())
    again = AnalysisRequest.model_validate_json(req.model_dump_json())
    assert again == req


def test_status_axes_are_independent_and_default_to_unknown_states() -> None:
    axes = StatusAxes()
    assert axes.scan_coverage is ScanCoverage.NOT_STARTED
    assert axes.data_preservation is DataPreservation.PENDING
    assert axes.training_readiness is None
    assert axes.estimate_evidence is None
    assert axes.hardware_fit is HardwareFit.NOT_EVALUATED
    assert {e.value for e in TrainingReadiness} == {"ready", "conditional", "unsupported"}
    assert {e.value for e in EvidenceLevel} == {
        "metadata_only",
        "analytic",
        "calibrated",
        "measured",
    }


def test_job_state_machine_names_match_plan() -> None:
    assert [s.value for s in PIPELINE_STAGES] == [
        "RESOLVING",
        "INSPECTING",
        "TOKENIZING",
        "VALIDATING_DATA",
        "PLANNING_BATCHES",
        "ESTIMATING",
    ]
    assert JobStatus.COMPLETED in TERMINAL_JOB_STATUSES
    assert JobStatus.CANCEL_REQUESTED not in TERMINAL_JOB_STATUSES


def test_result_requires_request_and_has_no_fake_numbers_by_default() -> None:
    from datetime import UTC, datetime

    result = AnalysisResult(
        analysis_id="a1",
        created_at=datetime.now(UTC),
        analysis_fingerprint="f",
        estimator_version="0.1.0",
        requested_config=AnalysisRequest.model_validate(load_example()),
    )
    assert result.memory is None
    assert result.hardware_fit is None
    assert result.status.hardware_fit is HardwareFit.NOT_EVALUATED
