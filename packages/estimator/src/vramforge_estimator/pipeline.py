"""Analysis orchestration (plan.md §14.2, docs/architecture.md §3).

`analyze` runs the stages RESOLVING → … → ESTIMATING and always returns an `AnalysisResult`:
when a stage fails, the result contains everything verified so far plus the blocking issue, and
later stages are not executed (no invented lengths or numbers).

Stopping rules:

- An `EstimatorError` or `NotImplementedError` (NOT_IMPLEMENTED) from a stage halts the pipeline.
  The issue is recorded in `result.errors` with ``details["pipeline_halted"] = True``;
  `terminal_status` turns that into PARTIAL (something was verified) or FAILED (nothing was).
  Other exceptions propagate to the worker, which marks the job FAILED with INTERNAL_ERROR.
- Ambiguous config/split/mapping that the request leaves unspecified ends the job with
  `needs_input` (NEEDS_INPUT); conversational data whose model has no chat template ends it with
  TEMPLATE_REQUIRED. A dataset-inspection error that no choice can fix (e.g. an unreadable file
  format, denied access) halts with that issue instead of asking.
- A split without rows (the reader's EMPTY_DATASET, or a scan that read to the end without a row)
  halts with EMPTY_DATASET after the scan, before validation and batch planning.
- Column mapping (plan §7.2): `format=auto` — including a mapping with only some role hints or only
  the empty-system policy — and no mapping use the inspection's suggestion, which the inspector
  computed with those hints (the requested policy is applied to it); an explicit format is checked
  against the inspected columns. A missing column or a column of the wrong kind asks for the
  mapping (COLUMN_MAPPING_REQUIRED) instead of scanning with a mapping that fails every row.
- An unsupported architecture/combination is a result, not a failure: the scan still runs (data
  statistics are useful), batch planning and memory estimation are skipped, and the estimate
  evidence is `metadata_only`.
- A full scan that did not reach COMPLETE stops before batch planning: a partial maximum length
  is never presented as the dataset maximum (plan §7.7, §19.2).

Status axes (plan §12.1): `scan_coverage` from the scan, `data_preservation` from the audit,
`training_readiness` from the compatibility report combined with data findings (context exceeded
or preservation violated → unsupported; preservation unknown → conditional; a run that halted,
was cancelled or needs input is never `ready`), `estimate_evidence` from the memory estimate
(`metadata_only` when only the inventory exists) and `hardware_fit` from the primary scenario —
or, when there is no primary scenario (e.g. GRPO without an explicit completion budget), the worst
outcome over all scenarios (exceeds > unknown > low_margin > expected_fit). No scenario keeps a
length-dependent verdict while row lengths are unverified (plan §10.3): an incomplete or sample
scan turns them into unknown with reason `scan_incomplete` (a run that stopped on an incomplete
scan reports that reason as its summary fit), failed rows withhold positive verdicts; verdicts that
do not depend on lengths (resident floor or loading budget over capacity) stay. Issues of the memory
estimate are reported in the result like every other stage's. GRPO context is checked per scenario
(prompt + budget, plan §8.3): the result-level check uses the chosen budget or the smallest
candidate, and a larger candidate that exceeds the context only withholds its own scenario's fit.

The signatures are owned by the orchestrator; the body is implemented by the api agent.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from vramforge_estimator import (
    __version__,
    batching,
    compatibility,
    inspection,
    memory,
    preprocessing,
    scan,
    sources,
)
from vramforge_estimator.errors import CancelledError, EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.keys import preprocess_key as make_preprocess_key
from vramforge_estimator.keys import request_fingerprint, source_key
from vramforge_estimator.scan import ScanContext, ScanLimits
from vramforge_estimator.schemas import (
    AnalysisRequest,
    AnalysisResult,
    BatchPlan,
    ColumnMapping,
    CompatibilityReport,
    ContextValidation,
    DataPreservation,
    DatasetFormat,
    DatasetInspection,
    DatasetScanResult,
    ErrorCode,
    EvidenceLevel,
    ExcludedComponent,
    HardwareFit,
    HardwareFitResult,
    HardwareMode,
    Issue,
    JobProgress,
    JobStatus,
    MeasurementScope,
    MemoryEstimate,
    ModelInventory,
    ModelInventorySummary,
    NeedsInput,
    NeedsInputChoice,
    Objective,
    PreservationAudit,
    ResolvedConfig,
    ScanCoverage,
    ScanMode,
    ScenarioResponse,
    Severity,
    SourceManifests,
    Stage,
    StatusAxes,
    TokenizerManifest,
    TrainingReadiness,
    UnknownComponent,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

log = logging.getLogger(__name__)

T = TypeVar("T")

# Artifact layout under the analysis artifact directory (docs/architecture.md §7).
INVENTORY_FILE = "model_inventory.json"
ARTIFACTS_FILE = "artifacts.json"
LENGTHS_DIR = "lengths"
# preprocess_key component when no backend profile was resolved (unsupported combination).
UNRESOLVED_LOCK = "unresolved"
HALTED = "pipeline_halted"
# artifacts.json key: issues of the stages `recompute` does not re-run (dataset inspection, scan).
DATA_ISSUES_KEY = "data_issues"

_STAGE_FOR_STATUS: dict[JobStatus, Stage] = {
    JobStatus.RESOLVING: Stage.RESOLVING,
    JobStatus.INSPECTING: Stage.INSPECTING,
    JobStatus.TOKENIZING: Stage.TOKENIZING,
    JobStatus.VALIDATING_DATA: Stage.VALIDATING_DATA,
    JobStatus.PLANNING_BATCHES: Stage.PLANNING_BATCHES,
    JobStatus.ESTIMATING: Stage.ESTIMATING,
}
_READINESS_RANK = {
    TrainingReadiness.UNSUPPORTED: 0,
    TrainingReadiness.CONDITIONAL: 1,
    TrainingReadiness.READY: 2,
}
_FIT_SEVERITY = {
    HardwareFit.EXCEEDS: 4,
    HardwareFit.UNKNOWN: 3,
    HardwareFit.LOW_MARGIN: 2,
    HardwareFit.EXPECTED_FIT: 1,
    HardwareFit.NOT_EVALUATED: 0,
}
_NEEDS_INPUT_CODES = frozenset(
    {
        ErrorCode.DATASET_CONFIG_REQUIRED,
        ErrorCode.DATASET_SPLIT_REQUIRED,
        ErrorCode.COLUMN_MAPPING_REQUIRED,
    }
)
_ROLE_FIELDS = ("system", "prompt", "chosen", "rejected", "completion", "messages", "text")
# Fit verdicts that do not depend on row lengths: the resident floor, the loading budget, an
# unsupported setup and "no hardware selected" stay when lengths are unverified (plan §10.3).
_LENGTH_INDEPENDENT_FITS = frozenset(
    {"not_evaluated", "floor_exceeds_capacity", "load_budget_insufficient", "unsupported"}
)
SCAN_INCOMPLETE_FIT_MESSAGE = (
    "데이터셋 전체를 분석하지 않아(샘플 분석 또는 끝나지 않은 스캔) 적합 판정을 보류합니다. 확인한 "
    "row의 최대 길이는 데이터셋 전체의 최대 길이가 아닐 수 있습니다."
)
# Column kinds (`DatasetColumn.kind`) each role accepts; "other" (e.g. all-null preview) passes.
# Same rule as the inspector's mapping analysis (inspection.dataset_mapping.ROLE_KINDS).
_ROLE_KINDS: dict[str, frozenset[str]] = {
    "system": frozenset({"string"}),
    "prompt": frozenset({"string", "messages"}),
    "chosen": frozenset({"string", "messages"}),
    "rejected": frozenset({"string", "messages"}),
    "completion": frozenset({"string", "messages"}),
    "messages": frozenset({"messages"}),
    "text": frozenset({"string"}),
}


def inventory_summary(inventory: ModelInventory) -> ModelInventorySummary:
    """The part of a full inventory embedded in results and API responses."""
    return ModelInventorySummary(
        facts=inventory.facts,
        tensor_count=len(inventory.tensors),
        linear_module_count=len(inventory.linear_modules),
        params_total=inventory.params_total,
        bytes_serialized_total=inventory.bytes_serialized_total,
        by_component=inventory.by_component,
        tied_groups=inventory.tied_groups,
        quantized_checkpoint_format=inventory.quantized_checkpoint_format,
        inventory_hash=inventory.inventory_hash,
    )


class JobContext(ScanContext, Protocol):
    """Runtime services the worker provides to the pipeline (no DB/Redis inside the core)."""

    analysis_id: str
    access: SourceAccess

    def set_stage(self, stage: JobStatus) -> None: ...

    def warn(self, issue: Issue) -> None: ...


# ---------------------------------------------------------------- optional context extensions


@dataclass(frozen=True)
class CachedScan:
    """A COMPLETE scan kept by the worker's owner-scoped scan cache (plan §16.3)."""

    result: DatasetScanResult
    artifact_path: Path  # the `lengths/` directory written by `scan.full_scan`
    template_content_loss_rows: int = 0
    issues: tuple[Issue, ...] = ()


@runtime_checkable
class ScanCacheContext(Protocol):
    """Implemented by the worker (DB-backed); the core only calls these two methods."""

    def lookup_scan_cache(self, preprocess_key: str) -> CachedScan | None: ...

    def store_scan_cache(self, preprocess_key: str, scan: CachedScan) -> None: ...


@runtime_checkable
class PartialResultContext(Protocol):
    """Receives the result-so-far after each stage (shown while the job runs)."""

    def publish_partial(self, result: AnalysisResult) -> None: ...


# ---------------------------------------------------------------- public helpers


def terminal_status(result: AnalysisResult) -> JobStatus:
    """The job status a finished `analyze` result maps to (plan §16.1)."""
    if result.needs_input is not None:
        return JobStatus.NEEDS_INPUT
    halted = [e for e in result.errors if e.details.get(HALTED)]
    if any(e.code is ErrorCode.CANCELLED for e in halted):
        return JobStatus.CANCELLED
    if halted:
        verified = (
            result.source_manifests.model is not None or result.source_manifests.dataset is not None
        )
        return JobStatus.PARTIAL if verified else JobStatus.FAILED
    return JobStatus.COMPLETED


def halting_issue(result: AnalysisResult) -> Issue | None:
    """The issue that stopped the pipeline, if any."""
    for issue in result.errors:
        if issue.details.get(HALTED):
            return issue
    return None


def load_inventory(artifact_dir: Path) -> ModelInventory | None:
    path = artifact_dir / INVENTORY_FILE
    if not path.is_file():
        return None
    return ModelInventory.model_validate_json(path.read_text(encoding="utf-8"))


def read_artifacts_manifest(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / ARTIFACTS_FILE
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------- internals


class _Halt(Exception):
    """Stops the pipeline; the reason is already recorded in the result."""


class _NeedsReanalysis(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _now() -> datetime:
    return datetime.now(UTC)


def _not_implemented(stage: Stage, component: str, *, halts: bool) -> Issue:
    message = (
        "이 분석 단계는 현재 빌드에서 아직 구현되지 않아 이후 단계를 진행하지 않았습니다."
        if halts
        else "이 계산은 현재 빌드에서 아직 구현되지 않아 결과에 포함하지 않았습니다."
    )
    return make_issue(ErrorCode.NOT_IMPLEMENTED, message, stage=stage, component=component)


def _blocking_dataset_issues(ds: DatasetInspection | None) -> list[Issue]:
    """Errors of the dataset inspection that no config/split/mapping choice can resolve (e.g. an
    unreadable file format, denied access, an internal error)."""
    if ds is None:
        return []
    return [
        issue
        for issue in ds.issues
        if issue.severity is Severity.ERROR and issue.code not in _NEEDS_INPUT_CODES
    ]


def _empty_split_issue(scan_result: DatasetScanResult, issues: list[Issue]) -> Issue | None:
    """Why a scanned split gives nothing to plan, or None when it has rows.

    The reader's own EMPTY_DATASET report wins; a split that was read to the end without a single
    row is empty; a read that failed before the first row is reported with its own error rather
    than claiming the split is empty."""
    reported = next((i for i in issues if i.code is ErrorCode.EMPTY_DATASET), None)
    if reported is not None:
        return reported
    if scan_result.rows_seen > 0:
        return None
    if scan_result.coverage is not ScanCoverage.COMPLETE:
        failure = next((i for i in issues if i.severity is Severity.ERROR), None)
        if failure is not None:
            return failure
    return make_issue(
        ErrorCode.EMPTY_DATASET,
        "선택한 split에 row가 하나도 없어 batch 계획과 메모리 산정을 할 수 없습니다. "
        "config와 split을 확인하세요.",
        stage=Stage.TOKENIZING,
        config=scan_result.config,
        split=scan_result.split,
    )


def _write_json_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _effective_format(mapping: ColumnMapping, ds: DatasetInspection | None) -> DatasetFormat:
    if mapping.format is not DatasetFormat.AUTO:
        return mapping.format
    if ds is not None and ds.detected_format not in (None, DatasetFormat.AUTO):
        assert ds.detected_format is not None
        return ds.detected_format
    if mapping.messages:
        return DatasetFormat.MESSAGES
    if mapping.chosen and mapping.rejected:
        return DatasetFormat.PREFERENCE
    if mapping.completion:
        return DatasetFormat.PROMPT_COMPLETION
    if mapping.prompt:
        return DatasetFormat.PROMPT_ONLY
    if mapping.text:
        return DatasetFormat.TEXT
    return DatasetFormat.AUTO


def _needs_chat_template(
    fmt: DatasetFormat, mapping: ColumnMapping, ds: DatasetInspection | None
) -> bool:
    """Conversational data (message lists) cannot be rendered without the model's template.

    String columns are rendered as plain text when the model has no template (TRL's
    non-conversational path), so only message-list data is checked here; per-row cases (e.g. a
    non-empty system value) are reported by the preprocessing adapter.
    """
    if fmt is DatasetFormat.MESSAGES:
        return True
    kinds = {c.name: c.kind for c in ds.columns} if ds is not None else {}
    roles = (mapping.messages, mapping.prompt, mapping.chosen, mapping.rejected, mapping.completion)
    return any(col and kinds.get(col) == "messages" for col in roles)


def _choice_from_issue(issue: Issue) -> NeedsInputChoice:
    """The question an inspector's needs-input error asks (its options when it lists them)."""
    default_field = {
        ErrorCode.DATASET_CONFIG_REQUIRED: "dataset.config",
        ErrorCode.DATASET_SPLIT_REQUIRED: "dataset.split",
    }.get(issue.code, "dataset.mapping")
    details = issue.details
    field_name = details.get("field")
    if not (isinstance(field_name, str) and field_name.startswith("dataset.")):
        field_name = default_field
    options = details.get("options")
    suggested = details.get("suggested")
    return NeedsInputChoice(
        field=field_name,
        options=[str(o) for o in options] if isinstance(options, list) else [],
        suggested=suggested if isinstance(suggested, str) else None,
        reason=issue.user_message,
    )


def _inspection_issue(
    ds: DatasetInspection, code: ErrorCode, field_name: str | None
) -> Issue | None:
    """The inspector's needs-input error with `code` (for `field_name` when it names one)."""
    for issue in ds.issues:
        if issue.code is not code or issue.severity is not Severity.ERROR:
            continue
        named = issue.details.get("field")
        if field_name is None or named in (None, field_name):
            return issue
    return None


def _with_issue(choice: NeedsInputChoice, issue: Issue | None) -> NeedsInputChoice:
    """Prefer the inspector's wording and suggestion for the same question."""
    if issue is None:
        return choice
    from_issue = _choice_from_issue(issue)
    return choice.model_copy(
        update={
            "reason": from_issue.reason,
            "options": choice.options or from_issue.options,
            "suggested": choice.suggested or from_issue.suggested,
        }
    )


def _explicit_mapping_problem(mapping: ColumnMapping, ds: DatasetInspection) -> str | None:
    """Why an explicit-format mapping cannot be scanned against the inspected columns."""
    roles = {role: column for role in _ROLE_FIELDS if (column := getattr(mapping, role))}
    if ds.columns:
        kinds = {c.name: c.kind for c in ds.columns}
        missing = sorted({column for column in roles.values() if column not in kinds})
        if missing:
            return "지정한 컬럼이 데이터셋에 없습니다: " + ", ".join(missing)
        wrong = [
            f"{column}→{role}"
            for role, column in roles.items()
            if kinds[column] != "other" and kinds[column] not in _ROLE_KINDS[role]
        ]
        if wrong:
            return "컬럼 값의 형식이 지정한 역할에 맞지 않습니다: " + ", ".join(wrong)
    # The inspector validated this same mapping (e.g. roles that do not fit the format, or a
    # format the objective cannot use).
    reported = _inspection_issue(ds, ErrorCode.COLUMN_MAPPING_REQUIRED, None)
    return reported.user_message if reported is not None else None


def _hint_conflict(requested: ColumnMapping | None, suggested: ColumnMapping) -> str | None:
    if requested is None:
        return None
    differing = [
        role
        for role in _ROLE_FIELDS
        if (hint := getattr(requested, role)) is not None and hint != getattr(suggested, role)
    ]
    if not differing:
        return None
    return (
        f"지정한 컬럼 역할({', '.join(differing)})이 데이터셋에서 찾은 매핑과 맞지 않습니다. "
        "사용할 매핑을 선택해 주세요."
    )


def _mapping_choice(
    candidates: list[ColumnMapping], suggested: ColumnMapping | None, reason: str
) -> NeedsInputChoice:
    return NeedsInputChoice(
        field="dataset.mapping",
        options=[_mapping_label(m) for m in candidates],
        suggested=_mapping_label(suggested) if suggested is not None else None,
        reason=reason,
    )


def _mapping_label(mapping: ColumnMapping) -> str:
    parts = [f"{role}={getattr(mapping, role)}" for role in _ROLE_FIELDS if getattr(mapping, role)]
    return ", ".join(parts) or mapping.format.value


def _request_template_kwargs(request: AnalysisRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if request.training.template.enable_thinking is not None:
        kwargs["enable_thinking"] = request.training.template.enable_thinking
    return kwargs


def _combine_readiness(*values: TrainingReadiness | None) -> TrainingReadiness | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return min(present, key=lambda v: _READINESS_RANK[v])


def _data_readiness(
    audit: PreservationAudit | None, context: ContextValidation | None
) -> TrainingReadiness:
    if context is not None and context.status == "exceeded":
        return TrainingReadiness.UNSUPPORTED
    if audit is None:
        return TrainingReadiness.CONDITIONAL
    if audit.status is DataPreservation.VIOLATED:
        return TrainingReadiness.UNSUPPORTED
    if audit.status is DataPreservation.VERIFIED:
        return TrainingReadiness.READY
    return TrainingReadiness.CONDITIONAL


def _summary_fit(estimate: MemoryEstimate) -> HardwareFitResult | None:
    """Primary scenario's fit, else the worst fit over all scenarios (see module docstring)."""
    if not estimate.scenarios:
        return None
    if estimate.primary_scenario_id is not None:
        for scenario in estimate.scenarios:
            if scenario.scenario_id == estimate.primary_scenario_id:
                return scenario.hardware_fit
    return max((s.hardware_fit for s in estimate.scenarios), key=lambda f: _FIT_SEVERITY[f.status])


def _completion_budgets(request: AnalysisRequest) -> list[int]:
    if request.training.objective is not Objective.GRPO:
        return []
    if request.grpo.completion_budget is not None:
        return [request.grpo.completion_budget]
    return sorted(set(request.grpo.completion_budget_candidates))


def _artifact_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class _LimitedScanContext:
    """Delegates to the job context with a smaller row limit (sample scans)."""

    def __init__(self, inner: ScanContext, limits: ScanLimits) -> None:
        self._inner = inner
        self.limits = limits
        self.artifact_dir = inner.artifact_dir

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        self._inner.report(progress, partial)

    def cancelled(self) -> bool:
        return self._inner.cancelled()

    def load_checkpoint(self) -> dict[str, Any] | None:
        return self._inner.load_checkpoint()

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        self._inner.save_checkpoint(data)


class _RecomputeContext:
    """Minimal context for `recompute` (no job, no progress, never cancelled)."""

    analysis_id = ""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.access = SourceAccess()
        self.limits = ScanLimits()

    def set_stage(self, stage: JobStatus) -> None:
        del stage

    def warn(self, issue: Issue) -> None:
        del issue

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        del progress, partial

    def cancelled(self) -> bool:
        return False

    def load_checkpoint(self) -> dict[str, Any] | None:
        return None

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        del data


@dataclass
class _Run:
    request: AnalysisRequest
    ctx: JobContext
    result: AnalysisResult
    model_src: ResolvedSource | None = None
    dataset_src: ResolvedSource | None = None
    inventory: ModelInventory | None = None
    tokenizer: TokenizerHandle | None = None
    tokenizer_manifest: TokenizerManifest | None = None
    ds_inspection: DatasetInspection | None = None
    config: str | None = None
    split: str | None = None
    mapping: ColumnMapping | None = None
    resolved: ResolvedConfig | None = None
    report: CompatibilityReport | None = None
    lengths_path: Path | None = None
    template_loss_rows: int = 0
    readiness: TrainingReadiness | None = None
    stage: Stage | None = None
    halted_status: JobStatus | None = None
    _seen: set[tuple[str, str, str, str]] = field(default_factory=set)

    scan_issues: list[Issue] = field(default_factory=list)
    context_exceeded_budgets: list[int] = field(default_factory=list)

    # -- issue bookkeeping ---------------------------------------------------------
    @staticmethod
    def _key(issue: Issue) -> tuple[str, str, str, str]:
        return (
            issue.code.value,
            issue.user_message,
            str(issue.stage or ""),
            issue.affected_component or "",
        )

    def note(self, issue: Issue, *, live: bool = True) -> None:
        key = self._key(issue)
        if key in self._seen:
            return
        self._seen.add(key)
        target = self.result.errors if issue.severity is Severity.ERROR else self.result.warnings
        target.append(issue)
        if live:
            self.ctx.warn(issue)

    def halt(self, issue: Issue) -> None:
        details = {**issue.details, HALTED: True}
        update: dict[str, Any] = {"details": details}
        if issue.stage is None and self.stage is not None:
            update["stage"] = self.stage
        if issue.severity is not Severity.ERROR and issue.code is not ErrorCode.CANCELLED:
            update["severity"] = Severity.ERROR
        stopped = issue.model_copy(update=update)
        key = self._key(stopped)
        self._seen.add(key)
        self.result.errors = [e for e in self.result.errors if self._key(e) != key]
        self.result.warnings = [w for w in self.result.warnings if self._key(w) != key]
        self.result.errors.append(stopped)
        raise _Halt

    def attempt(
        self,
        component: str,
        fn: Callable[..., T],
        *args: Any,
        _halts: bool = True,
        **kwargs: Any,
    ) -> tuple[T | None, Issue | None]:
        """Run one library call; failures become an issue (cancellation always halts).

        `_halts=False` marks optional outputs whose failure does not stop the pipeline (it only
        changes the wording of a NOT_IMPLEMENTED issue)."""
        stage = self.stage or Stage.REQUEST
        try:
            return fn(*args, **kwargs), None
        except CancelledError as exc:
            self.halt(exc.issue)
        except EstimatorError as exc:
            issue = exc.issue
            if issue.stage is None:
                issue = issue.model_copy(update={"stage": stage})
            return None, issue
        except NotImplementedError:
            return None, _not_implemented(stage, component, halts=_halts)
        raise AssertionError("unreachable")  # pragma: no cover

    def call(self, component: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        value, issue = self.attempt(component, fn, *args, **kwargs)
        if issue is not None:
            self.halt(issue)
        assert value is not None
        return value

    def enter(self, status: JobStatus) -> None:
        self.stage = _STAGE_FOR_STATUS[status]
        if self.ctx.cancelled():
            self.halt(CancelledError(self.stage).issue)
        self.ctx.set_stage(status)

    def publish(self) -> None:
        if isinstance(self.ctx, PartialResultContext):
            try:
                self.ctx.publish_partial(self.snapshot())
            except Exception:  # display-only: never fail the analysis for it
                log.warning("publishing a partial result failed", exc_info=True)

    def snapshot(self) -> AnalysisResult:
        self.result.status = self.status_axes()
        return self.result.model_copy(deep=True)

    # -- stages ----------------------------------------------------------------------
    def resolve_sources(self) -> None:
        self.enter(JobStatus.RESOLVING)
        access = self.ctx.access
        model_src, model_issue = self.attempt(
            "sources.resolve_model", sources.resolve_model, self.request.model, access
        )
        dataset_src, dataset_issue = self.attempt(
            "sources.resolve_dataset", sources.resolve_dataset, self.request.dataset, access
        )
        self.model_src, self.dataset_src = model_src, dataset_src
        self.result.source_manifests = SourceManifests(
            model=model_src.manifest if model_src else None,
            dataset=dataset_src.manifest if dataset_src else None,
        )
        self.publish()
        if model_issue and dataset_issue:
            self.note(model_issue)
            self.halt(dataset_issue)
        if model_issue or dataset_issue:
            self.halt(model_issue or dataset_issue)  # type: ignore[arg-type]

    def inspect(self) -> None:
        self.enter(JobStatus.INSPECTING)
        access = self.ctx.access
        assert self.model_src is not None and self.dataset_src is not None
        inventory, inv_issue = self.attempt(
            "inspection.inspect_model", inspection.inspect_model, self.model_src, access
        )
        if inventory is not None:
            self.inventory = inventory
            self.result.model_inventory_summary = inventory_summary(inventory)
            _write_json_atomic(self.ctx.artifact_dir / INVENTORY_FILE, inventory.model_dump_json())
        handle, tok_issue = self.attempt(
            "inspection.load_tokenizer", inspection.load_tokenizer, self.model_src, access
        )
        if handle is not None:
            self.tokenizer = handle
            self.tokenizer_manifest = handle.manifest
            self.result.tokenizer_manifest = handle.manifest
        ds, ds_issue = self.attempt(
            "inspection.inspect_dataset",
            inspection.inspect_dataset,
            self.dataset_src,
            self.request.dataset,
            access,
            self.request.training.objective,
        )
        self.ds_inspection = ds
        if ds is not None:
            for issue in ds.issues:
                self.note(issue)
        failures = [i for i in (inv_issue, tok_issue) if i is not None]
        needs_input: Issue | None = None
        if ds_issue is not None:
            if ds_issue.code in _NEEDS_INPUT_CODES:
                needs_input = ds_issue
            else:
                failures.append(ds_issue)
        # e.g. an unreadable file or denied access: no config/split/mapping choice can fix it, so
        # it stops the run instead of asking the user to choose from nothing.
        failures.extend(_blocking_dataset_issues(ds))
        if inventory is not None:
            compat_issue = self.resolve_compatibility(halt=False)
            if compat_issue is not None:
                failures.append(compat_issue)
        self.publish()
        if failures:
            for issue in failures[1:]:
                self.note(issue)
            if needs_input is not None:
                self.note(needs_input)
            self.halt(failures[0])
        if needs_input is not None:
            self.ask_for_input([], extra_issue=needs_input)
        assert ds is not None
        self.choose_dataset(ds)

    def resolve_compatibility(self, *, halt: bool = True) -> Issue | None:
        assert self.inventory is not None
        value, issue = self.attempt(
            "compatibility.resolve",
            compatibility.resolve,
            self.request,
            self.inventory,
            self.tokenizer_manifest,
        )
        if issue is not None:
            if halt:
                self.halt(issue)
            return issue
        assert value is not None
        resolved, report = value
        self.resolved, self.report = resolved, report
        self.result.resolved_config = resolved
        self.result.compatibility_report = report
        self.result.profile_id = (resolved.profile_id if resolved else None) or report.profile_id
        self.result.dependency_lock_digest = resolved.dependency_lock_digest if resolved else None
        for blocker in report.blockers:
            self.note(blocker)
        for warning in report.warnings:
            self.note(warning)
        reported = {i.code for i in (*report.blockers, *report.warnings)}
        if report.architecture_adapter is None and (
            ErrorCode.UNSUPPORTED_ARCHITECTURE not in reported
        ):
            self.note(
                make_issue(
                    ErrorCode.UNSUPPORTED_ARCHITECTURE,
                    "등록된 구조 adapter가 없어 모델 메타데이터와 데이터 분석까지만 제공합니다. "
                    "메모리 산정은 하지 않습니다.",
                    stage=Stage.INSPECTING,
                )
            )
        self.readiness = report.readiness
        return None

    def ask_for_input(
        self,
        choices: list[NeedsInputChoice],
        extra_issue: Issue | None,
        *,
        candidates: list[ColumnMapping] | None = None,
    ) -> None:
        ds = self.ds_inspection
        if extra_issue is not None and not choices:
            choices = [_choice_from_issue(extra_issue)]
        if candidates is None:
            candidates = list(ds.mapping_candidates) if ds else []
        self.result.needs_input = NeedsInput(
            choices=choices,
            columns=[c.name for c in ds.columns] if ds else [],
            mapping_candidates=candidates,
        )
        codes = {
            "dataset.config": ErrorCode.DATASET_CONFIG_REQUIRED,
            "dataset.split": ErrorCode.DATASET_SPLIT_REQUIRED,
            "dataset.mapping": ErrorCode.COLUMN_MAPPING_REQUIRED,
        }
        if extra_issue is not None:
            self.note(extra_issue)
        for choice in choices:
            code = codes.get(choice.field, ErrorCode.COLUMN_MAPPING_REQUIRED)
            if any(e.code is code and e.user_message == choice.reason for e in self.result.errors):
                continue  # the inspector already reported this question
            self.note(make_issue(code, choice.reason, stage=Stage.INSPECTING, field=choice.field))
        self.halted_status = JobStatus.NEEDS_INPUT
        raise _Halt

    def choose_dataset(self, ds: DatasetInspection) -> None:
        ref = self.request.dataset
        choices: list[NeedsInputChoice] = []
        # The inspector's own questions carry its wording, options and suggestion.
        config_issue = _inspection_issue(ds, ErrorCode.DATASET_CONFIG_REQUIRED, "dataset.config")
        split_issue = _inspection_issue(ds, ErrorCode.DATASET_SPLIT_REQUIRED, "dataset.split")
        config = ref.config or ds.selected_config
        if ref.config and ds.configs and ref.config not in ds.configs:
            choices.append(
                NeedsInputChoice(
                    field="dataset.config",
                    options=list(ds.configs),
                    suggested=ds.selected_config,
                    reason="요청한 config가 데이터셋에 없습니다. config를 다시 선택하세요.",
                )
            )
        elif config is None and (len(ds.configs) > 1 or config_issue is not None):
            choices.append(
                _with_issue(
                    NeedsInputChoice(
                        field="dataset.config",
                        options=list(ds.configs),
                        reason="데이터셋에 config가 여러 개 있어 분석할 config를 선택해야 합니다.",
                    ),
                    config_issue,
                )
            )
        split_names = [s.name for s in ds.splits]
        split = ref.split or ds.selected_split
        if ref.split and split_names and ref.split not in split_names:
            choices.append(
                NeedsInputChoice(
                    field="dataset.split",
                    options=split_names,
                    suggested=ds.selected_split,
                    reason="요청한 split이 데이터셋에 없습니다. 학습 split을 다시 선택하세요.",
                )
            )
        elif split is None:
            choices.append(
                _with_issue(
                    NeedsInputChoice(
                        field="dataset.split",
                        options=split_names,
                        reason="학습에 사용할 split을 자동으로 정할 수 없어 선택이 필요합니다.",
                    ),
                    split_issue,
                )
            )
        mapping, mapping_choice, candidates = self.resolve_mapping(ds)
        if mapping_choice is not None:
            choices.append(mapping_choice)
        if choices:
            self.ask_for_input(choices, None, candidates=candidates)
        assert mapping is not None
        self.config, self.split, self.mapping = config, split, mapping
        if ref.eval_split and self.request.scope.include_evaluation:
            self.note(
                make_issue(
                    ErrorCode.SCAN_PARTIAL,
                    "평가 split 분석은 아직 지원하지 않아 학습 split만 분석했습니다.",
                    severity=Severity.WARNING,
                    stage=Stage.INSPECTING,
                    reason="eval_split_not_scanned",
                )
            )
        fmt = _effective_format(mapping, ds)
        manifest = self.tokenizer_manifest
        if (
            manifest is not None
            and not manifest.chat_template_present
            and _needs_chat_template(fmt, mapping, ds)
        ):
            self.halt(
                make_issue(
                    ErrorCode.TEMPLATE_REQUIRED,
                    "대화형(메시지 목록) 데이터를 학습 형식으로 렌더링하려면 모델의 "
                    "chat template이 필요하지만 이 모델의 tokenizer에는 template이 없습니다. "
                    "다른 모델의 template으로 대신하지 않습니다.",
                    stage=Stage.INSPECTING,
                    component="tokenizer.chat_template",
                    format=fmt.value,
                )
            )

    def resolve_mapping(
        self, ds: DatasetInspection
    ) -> tuple[ColumnMapping | None, NeedsInputChoice | None, list[ColumnMapping]]:
        """The column mapping to scan with, or the question to ask (plan §7.2).

        - An explicit `format` is used as given once it is checked against the inspected columns:
          a missing column or a column holding the wrong kind of value would fail every row, so
          it is COLUMN_MAPPING_REQUIRED instead of a scan.
        - `format=auto` (also a mapping carrying only some role hints or only the empty-system
          policy) and no mapping use the inspection's suggestion, which the inspector computed
          with those hints; the requested empty-system policy is applied to it. Ambiguity, no
          candidate or a suggestion that contradicts a hint asks the user.

        Returns (mapping, choice, candidates offered in NEEDS_INPUT).
        """
        requested = self.request.dataset.mapping
        candidates = list(ds.mapping_candidates)
        if requested is not None and requested.format is not DatasetFormat.AUTO:
            problem = _explicit_mapping_problem(requested, ds)
            if problem is None:
                return requested, None, candidates
            candidates = [m for m in candidates if m != requested]
            return None, _mapping_choice(candidates, None, problem), candidates
        suggested = ds.suggested_mapping
        if suggested is not None and not ds.mapping_ambiguous:
            conflict = _hint_conflict(requested, suggested)
            if conflict is None:
                if requested is not None:
                    policy = requested.empty_system_policy
                    suggested = suggested.model_copy(update={"empty_system_policy": policy})
                return suggested, None, candidates
            return None, _mapping_choice(candidates, suggested, conflict), candidates
        reported = _inspection_issue(ds, ErrorCode.COLUMN_MAPPING_REQUIRED, None)
        reason = (
            reported.user_message
            if reported is not None
            else "컬럼 매핑을 자동으로 확정할 수 없어 학습에 쓸 컬럼을 선택해야 합니다."
        )
        return None, _mapping_choice(candidates, suggested, reason), candidates

    def context_limit(self) -> int | None:
        facts = self.inventory.facts if self.inventory is not None else None
        if facts is not None and facts.max_position_embeddings:
            return facts.max_position_embeddings
        tok = self.tokenizer_manifest
        if tok is not None and tok.model_max_length and not tok.model_max_length_is_sentinel:
            return tok.model_max_length
        return None

    def tokenize(self) -> None:
        self.enter(JobStatus.TOKENIZING)
        assert self.tokenizer is not None and self.mapping is not None
        assert self.dataset_src is not None
        request = self.request
        objective = request.training.objective
        template_kwargs = (
            dict(self.resolved.template_kwargs)
            if self.resolved is not None
            else _request_template_kwargs(request)
        )
        adapter = self.call(
            "preprocessing.get_adapter",
            preprocessing.get_adapter,
            objective,
            self.tokenizer,
            self.mapping,
            template_kwargs=template_kwargs,
            empty_system_policy=self.mapping.empty_system_policy,
        )
        manifest = self.dataset_src.manifest
        tok = self.tokenizer.manifest
        pkey = make_preprocess_key(
            source_key=source_key(
                manifest.source_type.value, f"{manifest.reference}@{manifest.resolved_revision}"
            ),
            config=self.config,
            split=self.split,
            mapping=self.mapping,
            objective=objective.value,
            tokenizer_fingerprint=tok.fingerprint,
            chat_template_sha256=tok.chat_template_sha256,
            template_kwargs=template_kwargs,
            adapter_name=adapter.name,
            adapter_version=adapter.version,
            dependency_lock_digest=(self.resolved.dependency_lock_digest if self.resolved else None)
            or UNRESOLVED_LOCK,
        )
        cached = self.lookup_cache(pkey)
        adopted: Path | None = None
        if cached is not None:
            try:
                adopted = self.adopt_artifact(cached.artifact_path)
            except OSError:  # e.g. the source analysis was deleted meanwhile: scan instead
                log.warning("could not reuse a cached scan; scanning again", exc_info=True)
                # A half-copied lengths/ must not pass for checkpointed parts of this analysis.
                shutil.rmtree(self.ctx.artifact_dir / LENGTHS_DIR, ignore_errors=True)
                cached = None
        if cached is not None:
            # Same owner, same immutable source and preprocessing identity (plan §16.3).
            log.info("reusing a cached complete scan for analysis %s", self.ctx.analysis_id)
            outcome_result = cached.result
            self.lengths_path = adopted
            self.template_loss_rows = cached.template_content_loss_rows
            issues: list[Issue] = list(cached.issues)
        else:
            stream = self.call(
                "inspection.open_rows",
                inspection.open_rows,
                self.dataset_src,
                config=self.config,
                split=self.split,
                access=self.ctx.access,
            )
            outcome = self.call(
                "scan.full_scan",
                scan.full_scan,
                stream,
                adapter,
                self.scan_context(),
                objective=objective,
                preprocess_key=pkey,
                tokenizer_fingerprint=tok.fingerprint,
                template_fingerprint=tok.chat_template_sha256,
                context_limit=self.context_limit(),
            )
            outcome_result = outcome.result
            issues = list(outcome.issues)
            self.template_loss_rows = outcome.template_content_loss_rows
            if outcome.artifact_path is not None:
                self.lengths_path = self.adopt_artifact(outcome.artifact_path)
            if outcome_result.coverage is ScanCoverage.COMPLETE and self.lengths_path:
                self.store_cache(
                    pkey,
                    CachedScan(
                        result=outcome_result,
                        artifact_path=self.lengths_path,
                        template_content_loss_rows=self.template_loss_rows,
                        issues=tuple(issues),
                    ),
                )
        # The scanner cannot know whether the split was chosen by the user (plan §7.1); the
        # inspector does (a split named in a dataset viewer URL is a user choice too).
        ds = self.ds_inspection
        auto_split = self.request.dataset.split is None and (ds is None or ds.split_auto_selected)
        self.result.dataset_scan = outcome_result.model_copy(
            update={"split_auto_selected": auto_split}
        )
        self.scan_issues = issues
        self.write_artifacts_manifest(pkey)
        for issue in issues:
            self.note(issue)
        self.publish()
        if self.ctx.cancelled():
            self.halt(CancelledError(Stage.TOKENIZING).issue)
        # A split without rows has no length to plan with: stop before validation and batching.
        empty = _empty_split_issue(outcome_result, issues)
        if empty is not None:
            self.halt(empty)

    def scan_context(self) -> ScanContext:
        dataset = self.request.dataset
        if dataset.scan_mode is ScanMode.SAMPLE and dataset.sample_rows:
            limits = self.ctx.limits
            sample = dataclasses.replace(limits, max_rows=min(limits.max_rows, dataset.sample_rows))
            return _LimitedScanContext(self.ctx, sample)
        return self.ctx

    def lookup_cache(self, pkey: str) -> CachedScan | None:
        if not isinstance(self.ctx, ScanCacheContext):
            return None
        try:
            cached = self.ctx.lookup_scan_cache(pkey)
        except Exception:
            log.warning("scan cache lookup failed", exc_info=True)
            return None
        if (
            cached is None
            or cached.result.coverage is not ScanCoverage.COMPLETE
            or cached.result.preprocess_key != pkey
            or not cached.artifact_path.exists()
        ):
            return None
        return cached

    def store_cache(self, pkey: str, cached: CachedScan) -> None:
        if not isinstance(self.ctx, ScanCacheContext):
            return
        try:
            self.ctx.store_scan_cache(pkey, cached)
        except Exception:
            log.warning("scan cache store failed", exc_info=True)

    def adopt_artifact(self, path: Path) -> Path:
        """Keep the length artifact inside this analysis' directory (self-contained deletion
        and recompute). Hard links are used when possible."""
        own = self.ctx.artifact_dir.resolve()
        source = path.resolve()
        if source.is_relative_to(own):
            return source
        dest = own / LENGTHS_DIR
        if dest.exists():
            shutil.rmtree(dest)
        if source.is_dir():
            shutil.copytree(source, dest, copy_function=_link_or_copy)
        else:
            dest.mkdir(parents=True)
            _link_or_copy(str(source), str(dest / source.name))
        return dest

    def write_artifacts_manifest(self, pkey: str) -> None:
        own = self.ctx.artifact_dir.resolve()
        lengths = None
        if self.lengths_path is not None and self.lengths_path.resolve().is_relative_to(own):
            lengths = str(self.lengths_path.resolve().relative_to(own))
        # Dataset-inspection and scan findings: `recompute` does not re-run those stages, so it
        # re-reports them from here instead of silently dropping them.
        data_issues = [
            *(self.ds_inspection.issues if self.ds_inspection is not None else []),
            *self.scan_issues,
        ]
        payload = {
            "version": 1,
            "inventory": INVENTORY_FILE if self.inventory is not None else None,
            "lengths": lengths,
            "preprocess_key": pkey,
            "template_content_loss_rows": self.template_loss_rows,
            DATA_ISSUES_KEY: [i.model_dump(mode="json") for i in data_issues],
        }
        _write_json_atomic(self.ctx.artifact_dir / ARTIFACTS_FILE, json.dumps(payload))

    def validate(self, *, halt_on_partial_scan: bool) -> None:
        self.enter(JobStatus.VALIDATING_DATA)
        scan_result = self.result.dataset_scan
        assert scan_result is not None
        facts = self.inventory.facts if self.inventory is not None else None
        # GRPO: a scenario's context is prompt + its completion budget (plan §8.3). The result-level
        # check (and so the audit and readiness) uses the smallest offered budget; candidates the
        # user did not choose that exceed the context only withhold their own scenario's fit.
        budgets = self.grpo_budgets()
        context = self.call(
            "scan.validate_context",
            scan.validate_context,
            scan_result,
            model_declared_max=facts.max_position_embeddings if facts else None,
            tokenizer=self.tokenizer_manifest,
            backend_verified_max=None,
            extra_tokens=budgets[0] if budgets else 0,
        )
        self.result.context_validation = context
        if len(budgets) > 1:
            exceeded = [budgets[0]] if context.status == "exceeded" else []
            for budget in budgets[1:]:
                per_budget, _ = self.attempt(
                    "scan.validate_context",
                    scan.validate_context,
                    scan_result,
                    model_declared_max=facts.max_position_embeddings if facts else None,
                    tokenizer=self.tokenizer_manifest,
                    backend_verified_max=None,
                    extra_tokens=budget,
                    _halts=False,
                )
                if per_budget is not None and per_budget.status == "exceeded":
                    exceeded.append(budget)
            self.context_exceeded_budgets = exceeded
            if exceeded:
                every = len(exceeded) == len(budgets)
                self.note(
                    make_issue(
                        ErrorCode.CONTEXT_EXCEEDED,
                        f"생성 예산 {', '.join(f'{b:,}' for b in exceeded)} tokens에서는 가장 긴 "
                        "prompt와 생성 길이의 합이 모델의 context 상한을 넘습니다. 해당 시나리오는 "
                        "적합 판정을 보류하며 자동으로 자르거나 제외하지 않습니다.",
                        severity=Severity.ERROR if every else Severity.WARNING,
                        stage=Stage.VALIDATING_DATA,
                        budgets=exceeded,
                    )
                )
        self.audit(None)
        self.publish()
        if halt_on_partial_scan and scan_result.coverage is not ScanCoverage.COMPLETE:
            audit = self.result.preservation_audit
            reported = [i for i in self.scan_issues if i.severity is Severity.ERROR] + [
                v for v in (audit.violations if audit else []) if v.code is ErrorCode.SCAN_PARTIAL
            ]
            self.halt(
                reported[0]
                if reported
                else make_issue(
                    ErrorCode.SCAN_PARTIAL,
                    "데이터셋 전체를 끝까지 확인하지 못해 batch 계획과 메모리 산정을 하지 "
                    "않았습니다. 일부만 본 최대 길이를 전체 최대 길이로 쓰지 않습니다.",
                    stage=Stage.VALIDATING_DATA,
                    coverage=scan_result.coverage.value,
                )
            )

    def grpo_budgets(self) -> list[int]:
        """Completion budgets of the GRPO scenarios, ascending ([] for other objectives)."""
        grpo = self.resolved.grpo if self.resolved is not None else None
        if grpo is not None:
            return sorted(set(grpo.completion_budgets))
        return _completion_budgets(self.request)

    def audit(self, plan: BatchPlan | None) -> None:
        scan_result = self.result.dataset_scan
        assert scan_result is not None and self.result.context_validation is not None
        audit = self.call(
            "scan.audit_preservation",
            scan.audit_preservation,
            scan_result,
            context=self.result.context_validation,
            batch_plan=plan,
            packing=self.request.training.packing,
            scope=self.request.scope,
            template_content_loss_rows=self.template_loss_rows,
        )
        self.result.preservation_audit = audit
        for violation in audit.violations:
            self.note(violation)

    def can_estimate(self) -> bool:
        return (
            self.resolved is not None
            and self.report is not None
            and self.report.support_grade is not None
            and self.inventory is not None
        )

    def plan(self) -> BatchPlan:
        self.enter(JobStatus.PLANNING_BATCHES)
        assert self.resolved is not None
        if self.lengths_path is None:
            self.halt(
                make_issue(
                    ErrorCode.INTERNAL_ERROR,
                    "row 길이 기록 파일이 없어 batch 계획을 세울 수 없습니다.",
                    stage=Stage.PLANNING_BATCHES,
                )
            )
        assert self.lengths_path is not None
        lengths = self.call("scan.load_lengths", scan.load_lengths, self.lengths_path)
        plan = self.call(
            "batching.plan_batches",
            batching.plan_batches,
            lengths,
            self.resolved,
            seed=self.request.training.seed,
        )
        self.result.batch_plan = plan
        for issue in plan.issues:
            self.note(issue)
        self.audit(plan)
        self.publish()
        return plan

    def estimate(self, plan: BatchPlan) -> None:
        self.enter(JobStatus.ESTIMATING)
        assert self.resolved is not None and self.inventory is not None
        request = self.request
        readiness = self.current_readiness()
        assert readiness is not None
        estimate = self.call(
            "memory.estimate_memory",
            memory.estimate_memory,
            self.inventory,
            self.resolved,
            plan,
            scope=request.scope,
            hardware=request.hardware,
            margin_policy=request.margin_policy,
            readiness=readiness,
        )
        estimate = self.guard_context_fits(estimate)
        estimate = self.guard_unverified_fits(estimate)
        self.result.memory = estimate
        self.result.hardware_fit = _summary_fit(estimate)
        self.collect_estimate_notes(estimate)
        self.secondary_estimates()

    def scan_incomplete(self) -> bool:
        """Not every row of the split was analyzed (partial/failed coverage or a sample scan)."""
        scan_result = self.result.dataset_scan
        return scan_result is not None and (
            scan_result.coverage is not ScanCoverage.COMPLETE
            or self.request.dataset.scan_mode is ScanMode.SAMPLE
        )

    def guard_unverified_fits(self, estimate: MemoryEstimate) -> MemoryEstimate:
        """No fit verdict that depends on unverified row lengths (plan §10.3: an expected fit
        needs the full analysis), for every scenario so neither the summary nor a card claims one.

        - Not every row was analyzed (partial coverage or a sample scan): each length-dependent
          verdict (expected fit, low margin, peak over capacity) becomes UNKNOWN with reason
          `scan_incomplete`. Verdicts that do not depend on lengths stay (the resident floor alone
          over capacity, the loading budget, unsupported, not evaluated).
        - Every row was read but some failed to tokenize: positive verdicts (expected fit, low
          margin) become UNKNOWN (`unknown_components`); a peak over capacity stays exceeded.
        """
        scan_result = self.result.dataset_scan
        reason: Literal["scan_incomplete", "unknown_components"]
        if self.scan_incomplete():
            reason = "scan_incomplete"
            message = SCAN_INCOMPLETE_FIT_MESSAGE
            detail = "scan_incomplete"
            statuses = {HardwareFit.EXPECTED_FIT, HardwareFit.LOW_MARGIN, HardwareFit.EXCEEDS}
        elif scan_result is not None and scan_result.rows_failed:
            reason = "unknown_components"
            message = (
                "처리하지 못한 row가 있어 데이터셋의 최대 길이를 확인하지 못했으므로 적합 판정을 "
                "보류합니다."
            )
            detail = "lengths_unverified"
            statuses = {HardwareFit.EXPECTED_FIT, HardwareFit.LOW_MARGIN}
        else:
            return estimate
        withheld = False
        scenarios = []
        for scenario in estimate.scenarios:
            fit = scenario.hardware_fit
            if fit.status in statuses and fit.reason not in _LENGTH_INDEPENDENT_FITS:
                withheld = True
                fit = fit.model_copy(
                    update={
                        "status": HardwareFit.UNKNOWN,
                        "reason": reason,
                        "message": message,
                        "utilization_ratio": None,  # computed from unverified lengths
                    }
                )
            scenarios.append(scenario.model_copy(update={"hardware_fit": fit}))
        if not withheld:
            return estimate
        self.note(
            make_issue(
                ErrorCode.SCAN_PARTIAL,
                message,
                severity=Severity.WARNING,
                stage=Stage.ESTIMATING,
                reason=detail,
            )
        )
        return estimate.model_copy(update={"scenarios": scenarios})

    def withhold_unestimated_fit(self) -> None:
        """A run that stopped on an incomplete scan has no scenario to judge: the summary still
        says why the fit is unknown (unless no hardware was selected)."""
        if (
            self.result.hardware_fit is None
            and self.request.hardware.mode is not HardwareMode.CAPACITY_ONLY
            and self.scan_incomplete()
            and self.can_estimate()
        ):
            self.result.hardware_fit = HardwareFitResult(
                status=HardwareFit.UNKNOWN,
                reason="scan_incomplete",
                message=SCAN_INCOMPLETE_FIT_MESSAGE,
            )

    def guard_context_fits(self, estimate: MemoryEstimate) -> MemoryEstimate:
        """A GRPO budget whose prompt + completion exceeds the context gets no positive fit."""
        if not self.context_exceeded_budgets:
            return estimate
        message = (
            "이 생성 예산에서는 가장 긴 prompt와 생성 길이의 합이 context 상한을 넘어 적합 판정을 "
            "보류합니다."
        )
        scenarios = []
        for scenario in estimate.scenarios:
            fit = scenario.hardware_fit
            budget = scenario.params.get("completion_budget")
            if budget in self.context_exceeded_budgets and fit.status in (
                HardwareFit.EXPECTED_FIT,
                HardwareFit.LOW_MARGIN,
            ):
                fit = fit.model_copy(
                    update={
                        "status": HardwareFit.UNKNOWN,
                        "reason": "unsupported",
                        "message": message,
                    }
                )
            scenarios.append(scenario.model_copy(update={"hardware_fit": fit}))
        return estimate.model_copy(update={"scenarios": scenarios})

    def collect_estimate_notes(self, estimate: MemoryEstimate) -> None:
        result = self.result
        result.assumptions = list(estimate.assumptions)
        excluded: dict[str, ExcludedComponent] = {}
        unknown: dict[str, UnknownComponent] = {}
        for scenario in estimate.scenarios:
            for item in scenario.excluded_components:
                excluded.setdefault(item.name, item)
            for device in scenario.devices:
                for u in device.unknown_components:
                    unknown.setdefault(u.name, u)
        result.excluded_components = list(excluded.values())
        result.unknown_components = list(unknown.values())
        for issue in estimate.issues:  # e.g. a loading-budget warning of the memory engine
            self.note(issue)
        primary = next(
            (s for s in estimate.scenarios if s.scenario_id == estimate.primary_scenario_id),
            estimate.scenarios[0] if estimate.scenarios else None,
        )
        if primary is not None and primary.devices:
            phases = primary.devices[0].phases
            result.measurement_scope = MeasurementScope(
                measured=False,
                phases_included=[p.phase for p in phases if p.included],
                phases_excluded=[p.phase for p in phases if not p.included],
                note="정적 추정입니다. GPU에서 측정한 값이 아닙니다.",
            )

    def secondary_estimates(self) -> None:
        """Host RAM, disk and analysis RAM: independent outputs that never stop the result."""
        assert self.resolved is not None and self.inventory is not None
        manifests = self.result.source_manifests
        host, issue = self.attempt(
            "memory.estimate_host_ram",
            memory.estimate_host_ram,
            self.inventory,
            self.resolved,
            manifests.model,
            _halts=False,
        )
        self.result.host_ram_estimate = host
        if issue is not None:
            self.note(issue.model_copy(update={"severity": Severity.WARNING}))
        disk, issue = self.attempt(
            "memory.estimate_disk",
            memory.estimate_disk,
            self.inventory,
            self.resolved,
            manifests.model,
            manifests.dataset,
            _artifact_bytes(self.lengths_path),
            _halts=False,
        )
        self.result.disk_estimate = disk
        if issue is not None:
            self.note(issue.model_copy(update={"severity": Severity.WARNING}))
        scan_result = self.result.dataset_scan
        analysis_ram, issue = self.attempt(
            "memory.estimate_analysis_ram",
            memory.estimate_analysis_ram,
            scan_result.rows_seen if scan_result else None,
            self.tokenizer_bytes(),
            _halts=False,
        )
        self.result.analysis_ram_estimate = analysis_ram
        if issue is not None:
            self.note(issue.model_copy(update={"severity": Severity.WARNING}))

    def tokenizer_bytes(self) -> int | None:
        tok = self.tokenizer_manifest
        model = self.result.source_manifests.model
        if tok is None or model is None or not tok.files_sha256:
            return None
        sizes = {f.path: f.size for f in model.files}
        values = [sizes.get(name) for name in tok.files_sha256]
        if any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    # -- status ------------------------------------------------------------------------
    def current_readiness(self) -> TrainingReadiness | None:
        if self.report is None:
            return None
        if self.resolved is None or self.report.support_grade is None:
            return TrainingReadiness.UNSUPPORTED
        data = None
        if self.result.dataset_scan is not None:
            data = _data_readiness(self.result.preservation_audit, self.result.context_validation)
            if self.result.dataset_scan.coverage is not ScanCoverage.COMPLETE:
                data = _combine_readiness(data, TrainingReadiness.CONDITIONAL)
        return _combine_readiness(self.readiness, data)

    def status_axes(self) -> StatusAxes:
        result = self.result
        scan_result = result.dataset_scan
        if scan_result is not None:
            coverage = scan_result.coverage
        elif self.stage is Stage.TOKENIZING and self.halted_status is JobStatus.CANCELLED:
            coverage = ScanCoverage.PARTIAL
        elif self.stage is Stage.TOKENIZING and halting_issue(result) is not None:
            coverage = ScanCoverage.FAILED
        else:
            coverage = ScanCoverage.NOT_STARTED
        audit = result.preservation_audit
        if audit is not None:
            preservation = audit.status
        elif scan_result is not None:
            preservation = DataPreservation.UNKNOWN
        else:
            preservation = DataPreservation.PENDING
        if result.memory is not None:
            evidence: EvidenceLevel | None = result.memory.evidence_level
        elif result.model_inventory_summary is not None:
            evidence = EvidenceLevel.METADATA_ONLY
        else:
            evidence = None
        if self.request.hardware.mode is HardwareMode.CAPACITY_ONLY:
            fit = HardwareFit.NOT_EVALUATED
        elif result.hardware_fit is not None:
            fit = result.hardware_fit.status
        else:
            fit = HardwareFit.UNKNOWN
        readiness = self.current_readiness()
        if readiness is TrainingReadiness.READY and self.stopped_early():
            # The whole setup was never verified; the trainer-config export is gated on `ready`.
            readiness = TrainingReadiness.CONDITIONAL
        return StatusAxes(
            scan_coverage=coverage,
            data_preservation=preservation,
            training_readiness=readiness,
            estimate_evidence=evidence,
            hardware_fit=fit,
        )

    def stopped_early(self) -> bool:
        """The run halted, was cancelled or ended with NEEDS_INPUT."""
        return self.halted_status is not None or halting_issue(self.result) is not None


def _new_result(request: AnalysisRequest, analysis_id: str) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=analysis_id,
        created_at=_now(),
        analysis_fingerprint=request_fingerprint(request),
        estimator_version=__version__,
        requested_config=request,
        planning_margin_policy=request.margin_policy,
    )


def analyze(request: AnalysisRequest, ctx: JobContext) -> AnalysisResult:
    run = _Run(request=request, ctx=ctx, result=_new_result(request, ctx.analysis_id))
    try:
        run.resolve_sources()
        run.inspect()
        run.tokenize()
        full = request.dataset.scan_mode is ScanMode.FULL
        run.validate(halt_on_partial_scan=full)
        if run.can_estimate():
            plan = run.plan()
            run.estimate(plan)
    except _Halt:
        stopped = halting_issue(run.result)
        if stopped is not None and stopped.code is ErrorCode.CANCELLED:
            run.halted_status = JobStatus.CANCELLED
    run.withhold_unestimated_fit()
    run.result.status = run.status_axes()
    return run.result


# ---------------------------------------------------------------- recompute


def _revision_changed(old: str | None, new: str | None, resolved: str | None) -> bool:
    if new is None:
        return old is not None
    return new not in {old, resolved}


def _mapping_changed(applied: ColumnMapping | None, new: ColumnMapping | None) -> str | None:
    """Why the request's mapping would tokenize differently from `applied`, the resolved mapping
    the stored scan used.

    `format=auto` holds role hints that detection completed: only the hinted roles are compared
    (a hint that agrees with the applied mapping resolves to the same mapping again). An explicit
    format is compared role by role. No mapping means auto-detection, which gives the applied
    mapping again for the same data.
    """
    if new is None or applied is None:
        return None
    hinted = new.format is DatasetFormat.AUTO
    for role in _ROLE_FIELDS:
        value = getattr(new, role)
        if hinted and value is None:
            continue
        if value != getattr(applied, role):
            return "컬럼 매핑이 바뀌어 전체 데이터를 다시 토큰화해야 합니다."
    if not hinted and new.format is not applied.format:
        return "데이터 형식 지정이 바뀌어 전체 데이터를 다시 토큰화해야 합니다."
    if new.empty_system_policy is not applied.empty_system_policy:
        return "빈 system 메시지 처리 방식이 바뀌어 전체 데이터를 다시 토큰화해야 합니다."
    return None


def reanalysis_reasons(base: AnalysisResult, request: AnalysisRequest) -> list[str]:
    """Korean reasons why `request` changes the preprocessing layer (docs/architecture.md §3.1).

    Empty when only batch/estimate-layer fields changed (strategy, LoRA, optimizer, precision,
    microbatch, accumulation, checkpointing, kernels, DPO/GRPO knobs, hardware, margin, scope).
    """
    old = base.requested_config
    manifests = base.source_manifests
    scan_result = base.dataset_scan
    reasons: list[str] = []
    if request.training.objective is not old.training.objective:
        reasons.append(
            "학습 방식(SFT/DPO/GRPO)이 바뀌어 해당 방식의 전처리를 다시 실행해야 합니다."
        )
    if (
        request.model.source_type is not old.model.source_type
        or request.model.reference.strip() != old.model.reference.strip()
    ):
        reasons.append("모델이 바뀌어 모델 구조와 tokenizer를 다시 확인해야 합니다.")
    elif _revision_changed(
        old.model.revision,
        request.model.revision,
        manifests.model.resolved_revision if manifests.model else None,
    ):
        reasons.append("모델 revision이 바뀌어 tokenizer와 모델 구조를 다시 확인해야 합니다.")
    new_ds, old_ds = request.dataset, old.dataset
    if (
        new_ds.source_type is not old_ds.source_type
        or new_ds.reference.strip() != old_ds.reference.strip()
    ):
        reasons.append("데이터셋이 바뀌어 전체 데이터를 다시 분석해야 합니다.")
    elif _revision_changed(
        old_ds.revision,
        new_ds.revision,
        manifests.dataset.resolved_revision if manifests.dataset else None,
    ):
        reasons.append("데이터셋 revision이 바뀌어 전체 데이터를 다시 분석해야 합니다.")
    applied_config = scan_result.config if scan_result else old_ds.config
    if new_ds.config is not None and new_ds.config != applied_config:
        reasons.append("데이터셋 config가 바뀌어 전체 데이터를 다시 분석해야 합니다.")
    applied_split = scan_result.split if scan_result else old_ds.split
    if new_ds.split is not None and new_ds.split != applied_split:
        reasons.append("학습 split이 바뀌어 전체 데이터를 다시 분석해야 합니다.")
    if new_ds.eval_split != old_ds.eval_split:
        reasons.append("평가 split이 바뀌어 데이터를 다시 분석해야 합니다.")
    if new_ds.scan_mode is not old_ds.scan_mode or new_ds.sample_rows != old_ds.sample_rows:
        reasons.append("스캔 범위(전체/샘플)가 바뀌어 데이터를 다시 분석해야 합니다.")
    mapping_reason = _mapping_changed(
        (scan_result.mapping_applied if scan_result else None) or old_ds.mapping, new_ds.mapping
    )
    if mapping_reason:
        reasons.append(mapping_reason)
    if request.training.template != old.training.template:
        reasons.append("chat template 옵션(thinking 등)이 바뀌어 토큰화를 다시 해야 합니다.")
    if request.training.packing != old.training.packing:
        reasons.append("packing 설정이 바뀌어 데이터 보존 검사와 토큰화를 다시 해야 합니다.")
    if request.training.data_policy is not old.training.data_policy:
        reasons.append("데이터 보존 정책이 바뀌어 데이터를 다시 분석해야 합니다.")
    return reasons


def _preprocessing_layer_changed(base: ResolvedConfig | None, new: ResolvedConfig) -> str | None:
    if base is None:
        # The stored scan was keyed without a backend profile (UNRESOLVED_LOCK, request template
        # options), so it is not the tokenization the newly resolved profile would produce.
        return (
            "이전 분석에는 적용된 backend profile이 없어 해당 profile 기준으로 "
            "데이터를 다시 토큰화해야 합니다."
        )
    if new.preprocessing_adapter != base.preprocessing_adapter:
        return "적용되는 전처리 구현이 바뀌어 전체 데이터를 다시 토큰화해야 합니다."
    if new.dependency_lock_digest != base.dependency_lock_digest:
        return "학습 환경 버전 고정(dependency lock)이 바뀌어 전처리를 다시 확인해야 합니다."
    if dict(new.template_kwargs) != dict(base.template_kwargs):
        return "chat template 인자가 바뀌어 토큰화를 다시 해야 합니다."
    return None


def _stored_issues(raw: object) -> list[Issue]:
    """Issues saved in artifacts.json (entries that no longer validate are skipped)."""
    issues: list[Issue] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            issues.append(Issue.model_validate(item))
        except ValueError:
            log.warning("ignoring a stored issue that does not match the current schema")
    return issues


def recompute(
    base: AnalysisResult, request: AnalysisRequest, artifact_dir: Path
) -> ScenarioResponse:
    """Re-plan batches and re-estimate memory from stored artifacts. Returns
    `requires_reanalysis=True` (with reasons) when `request` changes the preprocessing layer."""
    fingerprint = request_fingerprint(request)
    reasons = reanalysis_reasons(base, request)
    if reasons:
        return ScenarioResponse(
            fingerprint=fingerprint, requires_reanalysis=True, reanalysis_reasons=reasons
        )
    try:
        result = _recompute(base, request, artifact_dir)
    except _NeedsReanalysis as exc:
        return ScenarioResponse(
            fingerprint=fingerprint, requires_reanalysis=True, reanalysis_reasons=[exc.reason]
        )
    return ScenarioResponse(fingerprint=fingerprint, requires_reanalysis=False, result=result)


def _recompute(
    base: AnalysisResult, request: AnalysisRequest, artifact_dir: Path
) -> AnalysisResult:
    scan_result = base.dataset_scan
    if scan_result is None:
        raise _NeedsReanalysis("저장된 데이터 길이 분석 결과가 없어 데이터 재분석이 필요합니다.")
    if (
        scan_result.coverage is not ScanCoverage.COMPLETE
        and base.requested_config.dataset.scan_mode is ScanMode.FULL
    ):
        raise _NeedsReanalysis(
            "이전 분석이 데이터셋 전체를 끝까지 확인하지 못해 데이터 재분석이 필요합니다."
        )
    inventory = load_inventory(artifact_dir)
    manifest = read_artifacts_manifest(artifact_dir) or {}
    lengths_rel = manifest.get("lengths")
    if inventory is None or not lengths_rel:
        raise _NeedsReanalysis("저장된 모델·길이 분석 파일이 없어 데이터 재분석이 필요합니다.")
    lengths_path = (artifact_dir / str(lengths_rel)).resolve()
    if not lengths_path.is_relative_to(artifact_dir.resolve()) or not lengths_path.exists():
        raise _NeedsReanalysis("저장된 길이 분석 파일을 찾을 수 없어 데이터 재분석이 필요합니다.")

    ctx = _RecomputeContext(artifact_dir)
    result = _new_result(request, base.analysis_id)
    result.source_manifests = base.source_manifests
    result.tokenizer_manifest = base.tokenizer_manifest
    result.model_inventory_summary = base.model_inventory_summary
    result.dataset_scan = scan_result
    run = _Run(request=request, ctx=ctx, result=result)
    run.inventory = inventory
    run.tokenizer_manifest = base.tokenizer_manifest
    run.lengths_path = lengths_path
    run.template_loss_rows = int(manifest.get("template_content_loss_rows") or 0)
    for issue in _stored_issues(manifest.get(DATA_ISSUES_KEY)):
        run.note(issue, live=False)  # the data did not change: its findings still apply
    try:
        run.stage = Stage.INSPECTING
        run.resolve_compatibility()
        if run.resolved is not None:
            changed = _preprocessing_layer_changed(base.resolved_config, run.resolved)
            if changed:
                raise _NeedsReanalysis(changed)
        full = request.dataset.scan_mode is ScanMode.FULL
        run.validate(halt_on_partial_scan=full)
        if run.can_estimate():
            plan = run.plan()
            run.estimate(plan)
    except _Halt:
        pass
    run.withhold_unestimated_fit()
    result.status = run.status_axes()
    return result


__all__ = [
    "ARTIFACTS_FILE",
    "INVENTORY_FILE",
    "LENGTHS_DIR",
    "CachedScan",
    "JobContext",
    "PartialResultContext",
    "ScanCacheContext",
    "analyze",
    "halting_issue",
    "inventory_summary",
    "load_inventory",
    "read_artifacts_manifest",
    "reanalysis_reasons",
    "recompute",
    "terminal_status",
]
