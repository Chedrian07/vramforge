"""Full-scan results, preservation audit and context validation (plan.md §7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .common import DataPreservation, ErrorCode, Issue, Objective, ScanCoverage, VFModel
from .request import ColumnMapping


class Branch(StrEnum):
    """Which token-length series a statistic describes."""

    PROMPT = "prompt"  # prompt tokens (GRPO generation prompt, DPO/SFT prompt part)
    COMPLETION = "completion"  # SFT completion part
    SEQUENCE = "sequence"  # SFT full training sequence
    LOSS_TOKENS = "loss_tokens"  # tokens that receive loss (SFT)
    CHOSEN = "chosen"  # DPO chosen completion part
    REJECTED = "rejected"  # DPO rejected completion part
    CHOSEN_SEQUENCE = "chosen_sequence"  # DPO prompt + chosen
    REJECTED_SEQUENCE = "rejected_sequence"  # DPO prompt + rejected
    PAIR_MAX = "pair_max"  # DPO max(chosen_sequence, rejected_sequence) per row


class LengthRecord(VFModel):
    """One row of the row-length artifact (plan §7.6). Raw text and token ids are never stored."""

    row_id: str  # stable id: "<split>:<index>" in source order
    row_index: int
    shard_id: str
    split: str
    objective: Objective
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    sequence_tokens: int | None = None  # SFT total
    loss_token_count: int | None = None
    chosen_total_tokens: int | None = None
    rejected_total_tokens: int | None = None
    chosen_completion_tokens: int | None = None
    rejected_completion_tokens: int | None = None
    template_fingerprint: str | None = None
    tokenizer_fingerprint: str
    content_digest: str  # sha256 of the mapped record (not the raw row) for change detection
    # sha256 of the token ids the trainer would see; lets GPU validation confirm a re-read row
    # tokenizes identically without storing the ids (plan §7.6).
    token_digest: str | None = None
    processing_status: Literal["ok", "failed"]
    context_status: Literal["ok", "exceeded", "unknown"] = "unknown"
    error_code: ErrorCode | None = None


class HistogramBin(VFModel):
    lo: int  # inclusive
    hi: int  # exclusive
    count: int


class LengthStats(VFModel):
    count: int
    min: int | None = None
    max: int | None = None
    max_row_id: str | None = None
    mean: float | None = None
    p50: int | None = None
    p90: int | None = None
    p95: int | None = None
    p99: int | None = None
    total_tokens: int = 0
    quantiles_exact: bool = True  # False => display "근사 분위수"
    histogram: list[HistogramBin] = Field(default_factory=list)


class RowLength(VFModel):
    row_id: str
    length: int


class BranchStats(VFModel):
    branch: Branch
    stats: LengthStats
    top_rows: list[RowLength] = Field(default_factory=list)  # longest rows, descending


class FailedRow(VFModel):
    row_id: str
    shard_id: str | None = None
    error_code: ErrorCode
    message: str


class DatasetScanResult(VFModel):
    coverage: ScanCoverage
    objective: Objective
    config: str | None = None
    split: str | None = None
    split_auto_selected: bool = False
    mapping_applied: ColumnMapping | None = None
    # e.g. "SFT: chosen 응답만 학습, rejected는 사용하지 않음 (데이터 변환)"
    transformation_note: str
    rows_expected: int | None = None
    rows_seen: int = 0
    rows_ok: int = 0
    rows_failed: int = 0
    rows_unprocessed: int | None = None
    shards_total: int | None = None
    shards_completed: int = 0
    branches: list[BranchStats] = Field(default_factory=list)
    failed_rows_sample: list[FailedRow] = Field(default_factory=list)
    duplicate_rows: int | None = None
    # None when no context limit is known (unknown is never reported as 0).
    context_exceeded_rows: int | None = None
    # System messages skipped by the empty-system policy (OMIT), None when not applicable.
    omitted_system_messages: int | None = None
    preprocess_key: str
    artifact_id: str | None = None
    tokenizer_fingerprint: str
    template_fingerprint: str | None = None
    preprocessing_adapter: str
    preprocessing_adapter_version: str
    elapsed_seconds: float | None = None


class PreservationCheckName(StrEnum):
    """The eight conditions of plan §7.4."""

    FULL_READ = "full_read"
    NO_LENGTH_DROP = "no_length_drop"
    NO_SPLIT_OR_CONCAT = "no_split_or_concat"
    TEMPLATE_CONTENT_PRESERVED = "template_content_preserved"
    LAST_BATCH_INCLUDED = "last_batch_included"
    PACKING_SAFE = "packing_safe"
    CONTEXT_WITHIN_LIMIT = "context_within_limit"
    EVAL_SCOPE_CONSISTENT = "eval_scope_consistent"


class PreservationCheck(VFModel):
    name: PreservationCheckName
    passed: bool | None  # None = not applicable / not evaluated
    detail: str


class PreservationAudit(VFModel):
    status: DataPreservation
    checks: list[PreservationCheck] = Field(default_factory=list)
    violations: list[Issue] = Field(default_factory=list)


class ContextValidation(VFModel):
    model_declared_max: int | None = None  # config max_position_embeddings
    tokenizer_model_max_length: int | None = None
    tokenizer_limit_is_sentinel: bool = False
    backend_verified_max: int | None = None
    effective_limit: int | None = None
    limit_source: str | None = None
    max_observed_length: int | None = None
    exceeded_rows: int | None = None  # None when no limit is known
    exceeded_rows_exact: bool = True  # False => exceeded_rows is a lower bound
    status: Literal["ok", "exceeded", "unknown"] = "unknown"
