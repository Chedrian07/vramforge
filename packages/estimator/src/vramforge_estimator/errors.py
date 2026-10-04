"""Exceptions that carry a contract `Issue`.

Library code raises `EstimatorError(issue)`; the pipeline converts it into a partial result and
the API into an `ErrorResponse`. Messages in issues must be safe to show to users.
"""

from __future__ import annotations

from .schemas.common import ErrorCode, Issue, Severity, Stage


class EstimatorError(Exception):
    def __init__(self, issue: Issue) -> None:
        super().__init__(f"{issue.code}: {issue.user_message}")
        self.issue = issue


class CancelledError(EstimatorError):
    def __init__(self, stage: Stage | None = None) -> None:
        super().__init__(
            Issue(
                code=ErrorCode.CANCELLED,
                severity=Severity.WARNING,
                stage=stage,
                user_message="사용자 요청으로 작업이 취소되었습니다.",
            )
        )


def make_issue(
    code: ErrorCode,
    message: str,
    *,
    severity: Severity = Severity.ERROR,
    stage: Stage | None = None,
    retryable: bool = False,
    component: str | None = None,
    **details: object,
) -> Issue:
    return Issue(
        code=code,
        severity=severity,
        stage=stage,
        retryable=retryable,
        user_message=message,
        affected_component=component,
        details=dict(details),
    )
