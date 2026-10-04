"""Analysis orchestration (plan.md §14.2, docs/architecture.md §3).

`analyze` runs the stages RESOLVING → … → ESTIMATING and always returns an `AnalysisResult`:
when a stage fails, the result contains everything verified so far plus the blocking issue, and
later stages are not executed (no invented lengths or numbers).

The signatures are owned by the orchestrator; the body is implemented by the api agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vramforge_estimator.scan import ScanContext
from vramforge_estimator.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Issue,
    JobStatus,
    ScenarioResponse,
)
from vramforge_estimator.sources import SourceAccess


class JobContext(ScanContext, Protocol):
    """Runtime services the worker provides to the pipeline (no DB/Redis inside the core)."""

    analysis_id: str
    access: SourceAccess

    def set_stage(self, stage: JobStatus) -> None: ...

    def warn(self, issue: Issue) -> None: ...


def analyze(request: AnalysisRequest, ctx: JobContext) -> AnalysisResult:
    raise NotImplementedError


def recompute(
    base: AnalysisResult, request: AnalysisRequest, artifact_dir: Path
) -> ScenarioResponse:
    """Re-plan batches and re-estimate memory from stored artifacts. Returns
    `requires_reanalysis=True` (with reasons) when `request` changes the preprocessing layer."""
    raise NotImplementedError


__all__ = ["JobContext", "analyze", "recompute"]
