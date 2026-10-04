"""Result exports (plan.md §12.4).

- `analysis.json`: the full result (schema version, assumptions, evidence) after redaction.
- `resolved-plan.yaml`: the app's reproducible plan (not claimed 1:1 with TRL arguments).
- `report.md`: Korean, escaped, no raw HTML.
- `trainer-config.yaml`: only for a `ready` result; raises `EstimatorError` with the reason.

Raw data, HF tokens, absolute local paths and private URLs are excluded (`sanitize`).
"""

from __future__ import annotations

import json

import yaml

from vramforge_estimator.schemas import AnalysisResult

from .plan_yaml import HEADER as PLAN_HEADER
from .plan_yaml import build_plan
from .report_md import render_report
from .sanitize import sanitize
from .trainer_config import HEADER as TRAINER_HEADER
from .trainer_config import build_trainer_config


class _Dumper(yaml.SafeDumper):
    """Safe YAML without aliases (anchors would make the file harder to read)."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def _yaml(data: object) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100, indent=2)


def export_json(result: AnalysisResult) -> bytes:
    data = sanitize(result.model_dump(mode="json"))
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def export_plan_yaml(result: AnalysisResult) -> bytes:
    """`resolved-plan.yaml`: reproducible plan of this app (not claimed 1:1 with TRL args)."""
    return (PLAN_HEADER + _yaml(sanitize(build_plan(result)))).encode("utf-8")


def export_report_md(result: AnalysisResult) -> bytes:
    return render_report(result).encode("utf-8")


def export_trainer_config(result: AnalysisResult) -> bytes:
    """Raises `EstimatorError` unless the result is ready for execution."""
    return (TRAINER_HEADER + _yaml(sanitize(build_trainer_config(result)))).encode("utf-8")


__all__ = ["export_json", "export_plan_yaml", "export_report_md", "export_trainer_config"]
