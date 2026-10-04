"""Shared enums, base model and the issue (error/warning) model.

These names are part of the public contract: the API, the worker, exports and the web UI
(through generated TypeScript types) all depend on them. Change them only through the
orchestrator (docs/git-conventions.md §6.1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VFModel(BaseModel):
    """Base for every contract model: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------- training axes


class Objective(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class Strategy(StrEnum):
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class SourceType(StrEnum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    UPLOAD = "upload"


class ScanMode(StrEnum):
    FULL = "full"
    SAMPLE = "sample"


class DataPolicy(StrEnum):
    STRICT_NO_TRUNCATION = "strict_no_truncation"


class HardwareMode(StrEnum):
    CAPACITY_ONLY = "capacity_only"
    GPU_PRESET = "gpu_preset"
    CUSTOM = "custom"


# ---------------------------------------------------------------- job state (plan §16.1)


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RESOLVING = "RESOLVING"
    INSPECTING = "INSPECTING"
    NEEDS_INPUT = "NEEDS_INPUT"
    TOKENIZING = "TOKENIZING"
    VALIDATING_DATA = "VALIDATING_DATA"
    PLANNING_BATCHES = "PLANNING_BATCHES"
    ESTIMATING = "ESTIMATING"
    COMPLETED = "COMPLETED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.NEEDS_INPUT,
        JobStatus.COMPLETED,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.PARTIAL,
    }
)

# The ordered pipeline stages a running job moves through.
PIPELINE_STAGES: tuple[JobStatus, ...] = (
    JobStatus.RESOLVING,
    JobStatus.INSPECTING,
    JobStatus.TOKENIZING,
    JobStatus.VALIDATING_DATA,
    JobStatus.PLANNING_BATCHES,
    JobStatus.ESTIMATING,
)


# ---------------------------------------------------------------- result status axes (plan §12.1)


class ScanCoverage(StrEnum):
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


class DataPreservation(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class TrainingReadiness(StrEnum):
    READY = "ready"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


class EvidenceLevel(StrEnum):
    """Support grade of an estimate (plan §11.4)."""

    METADATA_ONLY = "metadata_only"
    ANALYTIC = "analytic"
    CALIBRATED = "calibrated"
    MEASURED = "measured"


class HardwareFit(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    EXPECTED_FIT = "expected_fit"
    LOW_MARGIN = "low_margin"
    EXCEEDS = "exceeds"
    UNKNOWN = "unknown"


class Evidence(StrEnum):
    """Evidence attached to a single number (allocation, assumption)."""

    ANALYTIC = "analytic"
    CALIBRATED = "calibrated"
    MEASURED = "measured"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------- issues (plan §15.4)


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Stage(StrEnum):
    REQUEST = "request"
    RESOLVING = "resolving"
    INSPECTING = "inspecting"
    TOKENIZING = "tokenizing"
    VALIDATING_DATA = "validating_data"
    PLANNING_BATCHES = "planning_batches"
    ESTIMATING = "estimating"
    EXPORT = "export"
    API = "api"


class ErrorCode(StrEnum):
    # plan.md §15.4
    SOURCE_ACCESS_DENIED = "SOURCE_ACCESS_DENIED"
    SOURCE_REVISION_CHANGED = "SOURCE_REVISION_CHANGED"
    LOCAL_PATH_NOT_ALLOWED = "LOCAL_PATH_NOT_ALLOWED"
    MODEL_METADATA_UNAVAILABLE = "MODEL_METADATA_UNAVAILABLE"
    TOKENIZER_REQUIRED = "TOKENIZER_REQUIRED"
    TEMPLATE_REQUIRED = "TEMPLATE_REQUIRED"
    REMOTE_CODE_REQUIRED = "REMOTE_CODE_REQUIRED"
    COLUMN_MAPPING_REQUIRED = "COLUMN_MAPPING_REQUIRED"
    SCAN_PARTIAL = "SCAN_PARTIAL"
    SCAN_FAILED_ROWS = "SCAN_FAILED_ROWS"
    DATA_PRESERVATION_VIOLATION = "DATA_PRESERVATION_VIOLATION"
    CONTEXT_EXCEEDED = "CONTEXT_EXCEEDED"
    GRPO_REWARD_UNSPECIFIED = "GRPO_REWARD_UNSPECIFIED"
    GRPO_BUDGET_UNSPECIFIED = "GRPO_BUDGET_UNSPECIFIED"
    TRAINER_BATCH_CONSTRAINT = "TRAINER_BATCH_CONSTRAINT"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    UNSUPPORTED_BACKEND_COMBINATION = "UNSUPPORTED_BACKEND_COMBINATION"
    REQUESTED_OPTION_NOT_EFFECTIVE = "REQUESTED_OPTION_NOT_EFFECTIVE"
    UNKNOWN_MEMORY_COMPONENT = "UNKNOWN_MEMORY_COMPONENT"
    CALIBRATION_OUT_OF_DOMAIN = "CALIBRATION_OUT_OF_DOMAIN"
    PROFILE_OOM = "PROFILE_OOM"
    PROFILE_SCOPE_INCOMPLETE = "PROFILE_SCOPE_INCOMPLETE"
    # additional codes used by this implementation
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_URL_NOT_ALLOWED = "SOURCE_URL_NOT_ALLOWED"
    UNSUPPORTED_MODEL_FORMAT = "UNSUPPORTED_MODEL_FORMAT"
    TEMPLATE_CONTENT_LOSS = "TEMPLATE_CONTENT_LOSS"
    DATASET_CONFIG_REQUIRED = "DATASET_CONFIG_REQUIRED"
    DATASET_SPLIT_REQUIRED = "DATASET_SPLIT_REQUIRED"
    DATASET_FORMAT_UNSUPPORTED = "DATASET_FORMAT_UNSUPPORTED"
    SCAN_QUOTA_EXCEEDED = "SCAN_QUOTA_EXCEEDED"
    SAMPLER_DROPS_ROWS = "SAMPLER_DROPS_ROWS"
    UNSUPPORTED_DISTRIBUTED_TOPOLOGY = "UNSUPPORTED_DISTRIBUTED_TOPOLOGY"
    CONFLICTING_OPTIONS = "CONFLICTING_OPTIONS"
    REANALYSIS_REQUIRED = "REANALYSIS_REQUIRED"
    GPU_WORKER_UNAVAILABLE = "GPU_WORKER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    UPLOAD_TYPE_NOT_ALLOWED = "UPLOAD_TYPE_NOT_ALLOWED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CANCELLED = "CANCELLED"
    JOB_TIMEOUT = "JOB_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    EMPTY_DATASET = "EMPTY_DATASET"
    SAMPLER_ORDER_NOT_REPRODUCED = "SAMPLER_ORDER_NOT_REPRODUCED"
    PROMPT_BOUNDARY_MISMATCH = "PROMPT_BOUNDARY_MISMATCH"
    SPECIAL_TOKEN_DUPLICATED = "SPECIAL_TOKEN_DUPLICATED"  # noqa: S105 - an error code
    LOAD_BUDGET_EXCEEDED = "LOAD_BUDGET_EXCEEDED"


class Issue(VFModel):
    """An error, warning or notice attached to a result or an API error response.

    `user_message` is Korean, safe to show (no stack traces, tokens, absolute paths).
    `details` holds structured, non-sensitive values (e.g. row ids, field names).
    """

    code: ErrorCode
    severity: Severity
    stage: Stage | None = None
    retryable: bool = False
    user_message: str
    technical_detail_ref: str | None = None
    affected_component: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class Assumption(VFModel):
    """A stated assumption that a number depends on (shown in the evidence tab)."""

    id: str
    text: str
    evidence: Evidence = Evidence.ASSUMPTION
    source: str | None = None


class ExcludedComponent(VFModel):
    """Something deliberately outside the computed scope (result becomes conditional)."""

    name: str
    reason: str
    code: ErrorCode | None = None


class UnknownComponent(VFModel):
    """Something inside the scope whose size cannot be estimated (never filled with 0)."""

    name: str
    reason: str
    phase: str | None = None
