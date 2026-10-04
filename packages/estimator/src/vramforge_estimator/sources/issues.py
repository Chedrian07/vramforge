"""Typed construction of the blocking issues raised by sources and inspection."""

from __future__ import annotations

from collections.abc import Mapping

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, Issue, Severity, Stage


def blocking_error(
    code: ErrorCode,
    message: str,
    *,
    stage: Stage,
    component: str | None = None,
    retryable: bool = False,
    details: Mapping[str, object] | None = None,
) -> EstimatorError:
    """`message` is Korean and safe to show; `details` hold non-sensitive structured values."""
    return EstimatorError(
        Issue(
            code=code,
            severity=Severity.ERROR,
            stage=stage,
            retryable=retryable,
            user_message=message,
            affected_component=component,
            details=dict(details or {}),
        )
    )
