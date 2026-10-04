"""Result exports (plan.md §12.4).

Exports exclude raw data, HF tokens, absolute local paths and private URLs by default.
`trainer-config.yaml` is produced only when readiness is `ready`.

Owner: api agent (docs/architecture.md §8). Signatures are the contract.
"""

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


__all__ = ["export_json", "export_plan_yaml", "export_report_md", "export_trainer_config"]
