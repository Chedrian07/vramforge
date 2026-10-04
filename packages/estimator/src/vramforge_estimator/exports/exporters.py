"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import AnalysisResult


def export_json(result: AnalysisResult) -> bytes:
    raise NotImplementedError


def export_plan_yaml(result: AnalysisResult) -> bytes:
    """`resolved-plan.yaml`: reproducible plan of this app (not claimed 1:1 with TRL args)."""
    raise NotImplementedError


def export_report_md(result: AnalysisResult) -> bytes:
    raise NotImplementedError


def export_trainer_config(result: AnalysisResult) -> bytes:
    """Raises `EstimatorError` unless the result is ready for execution."""
    raise NotImplementedError
