"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    BatchPlan,
    ContextValidation,
    DatasetScanResult,
    PreservationAudit,
    ScopeConfig,
    TokenizerManifest,
)


def validate_context(
    scan: DatasetScanResult,
    *,
    model_declared_max: int | None,
    tokenizer: TokenizerManifest | None,
    backend_verified_max: int | None,
    extra_tokens: int = 0,
) -> ContextValidation:
    """Compare the longest sequence (+ `extra_tokens`, e.g. a GRPO completion budget) with the
    limits, keeping model/tokenizer/backend limits separate (plan §7.5)."""
    raise NotImplementedError


def audit_preservation(
    scan: DatasetScanResult,
    *,
    context: ContextValidation,
    batch_plan: BatchPlan | None,
    packing: bool,
    scope: ScopeConfig,
    template_content_loss_rows: int,
) -> PreservationAudit:
    """Evaluate the eight no-truncation conditions of plan §7.4."""
    raise NotImplementedError
