"""HTTP API envelopes (docs/architecture.md §6).

Defined here (not in the API service) so the OpenAPI document — and the generated TypeScript
types used by the web app — come from one source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import EvidenceLevel, Issue, JobStatus, Objective, VFModel
from .events import JobProgress
from .request import AnalysisRequest, ColumnMapping, DatasetFormat, DatasetSourceRef, ModelSourceRef
from .resolved import SupportEntry
from .result import AnalysisResult
from .sources import ModelInventorySummary, SourceManifest, TokenizerManifest


class ErrorResponse(VFModel):
    error: Issue
    issues: list[Issue] = Field(default_factory=list)


class HealthResponse(VFModel):
    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, str] = Field(default_factory=dict)  # db, redis, worker, gpu_worker


# ---------------------------------------------------------------- inspection


class InspectRequest(VFModel):
    model: ModelSourceRef | None = None
    dataset: DatasetSourceRef | None = None
    objective: Objective | None = None  # lets the dataset inspector rank mapping candidates


class ModelInspection(VFModel):
    manifest: SourceManifest | None = None
    summary: ModelInventorySummary | None = None
    tokenizer: TokenizerManifest | None = None
    architecture_adapter: str | None = None
    support: list[SupportEntry] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class DatasetColumn(VFModel):
    name: str
    dtype: str
    kind: str  # "string" | "messages" | "number" | "other"


class DatasetSplitInfo(VFModel):
    name: str
    num_rows: int | None = None
    num_bytes: int | None = None


class DatasetInspection(VFModel):
    manifest: SourceManifest | None = None
    configs: list[str] = Field(default_factory=list)
    selected_config: str | None = None
    splits: list[DatasetSplitInfo] = Field(default_factory=list)
    selected_split: str | None = None
    split_auto_selected: bool = False
    columns: list[DatasetColumn] = Field(default_factory=list)
    detected_format: DatasetFormat | None = None
    mapping_candidates: list[ColumnMapping] = Field(default_factory=list)
    suggested_mapping: ColumnMapping | None = None
    mapping_ambiguous: bool = False
    issues: list[Issue] = Field(default_factory=list)


class InspectResponse(VFModel):
    model: ModelInspection | None = None
    dataset: DatasetInspection | None = None


# ---------------------------------------------------------------- uploads, profiles, roots


class UploadResponse(VFModel):
    upload_id: str
    reference: str  # use as DatasetSourceRef.reference with source_type=upload
    filename: str
    size_bytes: int
    sha256: str
    expires_at: datetime


class BackendProfileInfo(VFModel):
    profile_id: str
    profile_version: str
    architecture_adapter: str
    description: str
    model_types: list[str] = Field(default_factory=list)
    environment_id: str
    support: list[SupportEntry] = Field(default_factory=list)


class EnvironmentInfo(VFModel):
    environment_id: str
    description: str
    packages: dict[str, str] = Field(default_factory=dict)
    dependency_lock_digest: str


class GpuPreset(VFModel):
    id: str
    name: str
    total_bytes: int
    note: str = ""


class BackendProfilesResponse(VFModel):
    estimator_version: str
    profiles: list[BackendProfileInfo] = Field(default_factory=list)
    environments: list[EnvironmentInfo] = Field(default_factory=list)
    hardware_presets: list[GpuPreset] = Field(default_factory=list)
    gpu_worker_connected: bool = False
    evidence_levels: list[EvidenceLevel] = Field(default_factory=lambda: list(EvidenceLevel))


class LocalRoot(VFModel):
    name: str
    reference_prefix: str  # e.g. "local:models/"
    description: str = ""
    read_only: bool = True


class LocalRootsResponse(VFModel):
    roots: list[LocalRoot] = Field(default_factory=list)


# ---------------------------------------------------------------- analyses


class AnalysisCreated(VFModel):
    analysis_id: str
    status: JobStatus
    fingerprint: str
    created_at: datetime
    reused: bool = False  # True when the Idempotency-Key matched an existing analysis


class AnalysisStatus(VFModel):
    analysis_id: str
    status: JobStatus
    fingerprint: str
    progress: JobProgress | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    # When the result and its artifacts are deleted by retention (plan §16.4); None while running.
    expires_at: datetime | None = None
    last_event_id: int | None = None
    request: AnalysisRequest | None = None  # lets the UI restore the form after a reload
    result: AnalysisResult | None = None  # partial while running, final when terminal
    error: Issue | None = None


class ScenarioRequest(VFModel):
    """The full current form state; the server decides what can be recomputed."""

    request: AnalysisRequest
    client_fingerprint: str | None = None


class ScenarioResponse(VFModel):
    fingerprint: str
    client_fingerprint: str | None = None
    requires_reanalysis: bool
    reanalysis_reasons: list[str] = Field(default_factory=list)
    result: AnalysisResult | None = None
