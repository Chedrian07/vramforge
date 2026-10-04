"""The analysis result (plan.md §12.3).

Integer bytes everywhere. "Unknown", "not applicable" and "not measured" are distinct:
unknown => `None` + an entry in `unknown_components`; not applicable => field omitted/None with the
reason in `excluded_components` or `measurement_scope`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .batching import BatchPlan
from .common import (
    Assumption,
    DataPreservation,
    EvidenceLevel,
    ExcludedComponent,
    HardwareFit,
    Issue,
    ScanCoverage,
    TrainingReadiness,
    UnknownComponent,
    VFModel,
)
from .memory import (
    AnalysisRamEstimate,
    DiskEstimate,
    HardwareFitResult,
    HostRamEstimate,
    MemoryEstimate,
    Phase,
)
from .request import AnalysisRequest, ColumnMapping, MarginPolicy
from .resolved import CompatibilityReport, ResolvedConfig
from .scan import ContextValidation, DatasetScanResult, PreservationAudit
from .sources import ModelInventorySummary, SourceManifest, TokenizerManifest


class StatusAxes(VFModel):
    """Independent status axes (plan §12.1) — never collapsed into a single success badge."""

    scan_coverage: ScanCoverage = ScanCoverage.NOT_STARTED
    data_preservation: DataPreservation = DataPreservation.PENDING
    training_readiness: TrainingReadiness | None = None
    estimate_evidence: EvidenceLevel | None = None
    hardware_fit: HardwareFit = HardwareFit.NOT_EVALUATED


class SourceManifests(VFModel):
    model: SourceManifest | None = None
    dataset: SourceManifest | None = None


class NeedsInputChoice(VFModel):
    field: str  # "dataset.config" | "dataset.split" | "dataset.mapping"
    options: list[str] = Field(default_factory=list)
    suggested: str | None = None
    reason: str


class NeedsInput(VFModel):
    choices: list[NeedsInputChoice]
    columns: list[str] = Field(default_factory=list)
    mapping_candidates: list[ColumnMapping] = Field(default_factory=list)


class MeasurementScope(VFModel):
    measured: bool = False
    phases_included: list[Phase] = Field(default_factory=list)
    phases_excluded: list[Phase] = Field(default_factory=list)
    note: str = ""


class AnalysisResult(VFModel):
    analysis_id: str
    schema_version: str = "1.0"
    created_at: datetime
    analysis_fingerprint: str
    estimator_version: str
    status: StatusAxes = Field(default_factory=StatusAxes)

    source_manifests: SourceManifests = Field(default_factory=SourceManifests)
    tokenizer_manifest: TokenizerManifest | None = None
    model_inventory_summary: ModelInventorySummary | None = None

    dataset_scan: DatasetScanResult | None = None
    preservation_audit: PreservationAudit | None = None
    context_validation: ContextValidation | None = None

    requested_config: AnalysisRequest
    resolved_config: ResolvedConfig | None = None
    observed_config: dict[str, object] | None = None  # GPU validation only (M5); None = 미검증
    compatibility_report: CompatibilityReport | None = None

    batch_plan: BatchPlan | None = None
    memory: MemoryEstimate | None = None  # memory_by_device_and_phase + peak breakdowns
    host_ram_estimate: HostRamEstimate | None = None
    analysis_ram_estimate: AnalysisRamEstimate | None = None
    disk_estimate: DiskEstimate | None = None

    planning_margin_policy: MarginPolicy = Field(default_factory=MarginPolicy)
    hardware_fit: HardwareFitResult | None = None  # of the primary scenario

    assumptions: list[Assumption] = Field(default_factory=list)
    unknown_components: list[UnknownComponent] = Field(default_factory=list)
    excluded_components: list[ExcludedComponent] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)
    errors: list[Issue] = Field(default_factory=list)
    needs_input: NeedsInput | None = None

    profile_id: str | None = None
    dependency_lock_digest: str | None = None
    measurement_scope: MeasurementScope = Field(default_factory=MeasurementScope)
