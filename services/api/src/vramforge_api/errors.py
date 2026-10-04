"""Error responses: every failure is an `ErrorResponse { error: Issue }` (docs/architecture.md §6).

That includes what FastAPI/Starlette would otherwise answer in their own formats: unknown routes
(404), wrong methods (405, `Allow` kept), request validation (422, without echoing the submitted
values), malformed JSON, `HTTPException`s (their `detail` is never echoed), responses that fail
their declared model (500) and `NotImplementedError` (501 NOT_IMPLEMENTED).

User messages are Korean and never contain stack traces, tokens or absolute paths; details are
logged server-side only (with redaction, see `logging_setup`).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, ErrorResponse, Issue, Severity, Stage

log = logging.getLogger(__name__)

NO_STORE = {"Cache-Control": "no-store"}

# HTTP status for an `EstimatorError` raised while handling a request. Unlisted codes are request
# or compatibility problems (422).
STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.SOURCE_NOT_FOUND: 404,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.CONCURRENCY_LIMIT: 429,
    ErrorCode.UPLOAD_TOO_LARGE: 413,
    ErrorCode.UPLOAD_TYPE_NOT_ALLOWED: 415,
    ErrorCode.GPU_WORKER_UNAVAILABLE: 503,
    ErrorCode.REANALYSIS_REQUIRED: 409,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.NOT_IMPLEMENTED: 501,
    ErrorCode.JOB_TIMEOUT: 504,
}


class ApiError(Exception):
    """An HTTP error carrying a contract `Issue`."""

    def __init__(
        self,
        status_code: int,
        issue: Issue,
        issues: list[Issue] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{status_code} {issue.code}")
        self.status_code = status_code
        self.issue = issue
        self.issues = issues or []
        self.headers = headers or {}


class BodyTooLarge(StarletteHTTPException):
    """A request body read past its cap (raised by the security middleware while streaming).

    An `HTTPException`, so FastAPI's body parsing re-raises it instead of answering 400, and
    route code that streams the body itself must let it propagate.
    """

    def __init__(self, issue: Issue) -> None:
        super().__init__(status_code=413)
        self.issue = issue


def api_error(
    status_code: int,
    code: ErrorCode,
    message: str,
    *,
    stage: Stage | None = Stage.API,
    retryable: bool = False,
    **details: object,
) -> ApiError:
    issue = Issue(
        code=code,
        severity=Severity.ERROR,
        stage=stage,
        retryable=retryable,
        user_message=message,
        details=dict(details),
    )
    return ApiError(status_code, issue)


def not_found() -> ApiError:
    return api_error(404, ErrorCode.NOT_FOUND, "요청한 항목을 찾을 수 없습니다.")


def error_response(
    status_code: int,
    issue: Issue,
    issues: list[Issue] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=issue, issues=issues or [])
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={**NO_STORE, **(headers or {})},
    )


def _validation_fields(exc: RequestValidationError) -> list[dict[str, Any]]:
    fields = []
    for err in exc.errors()[:50]:
        loc = [str(p) for p in err.get("loc", ())]
        # Never echo the submitted value (it may contain tokens or dataset text).
        fields.append(
            {
                "loc": ".".join(loc),
                "type": str(err.get("type", "")),
                "message": str(err.get("msg", ""))[:300],
            }
        )
    return fields


async def _api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.issue, exc.issues, exc.headers)


async def _body_too_large_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, BodyTooLarge)
    return error_response(413, exc.issue)


async def _estimator_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, EstimatorError)
    status = STATUS_BY_CODE.get(exc.issue.code, 422)
    return error_response(status, exc.issue)


async def _validation_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    fields = _validation_fields(exc)
    where = ", ".join(f["loc"] for f in fields[:5])
    message = "요청 형식이 올바르지 않습니다."
    if any(f["type"] == "json_invalid" for f in fields):
        message = "요청 본문이 올바른 JSON이 아닙니다."
    elif where:
        message = f"요청 형식이 올바르지 않습니다: {where}"
    issue = Issue(
        code=ErrorCode.INVALID_REQUEST,
        severity=Severity.ERROR,
        stage=Stage.REQUEST,
        user_message=message,
        details={"fields": fields},
    )
    return error_response(422, issue)


_HTTP_CODES: dict[int, tuple[ErrorCode, str]] = {
    400: (ErrorCode.INVALID_REQUEST, "요청 형식이 올바르지 않습니다."),
    401: (ErrorCode.UNAUTHORIZED, "접근 토큰이 필요합니다."),
    403: (ErrorCode.FORBIDDEN, "이 요청을 수행할 권한이 없습니다."),
    404: (ErrorCode.NOT_FOUND, "요청한 항목을 찾을 수 없습니다."),
    405: (ErrorCode.INVALID_REQUEST, "허용되지 않는 요청 방식입니다."),
    413: (ErrorCode.INVALID_REQUEST, "요청 본문이 너무 큽니다."),
}


async def _http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code, message = _HTTP_CODES.get(
        exc.status_code,
        (
            ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.INVALID_REQUEST,
            "요청을 처리할 수 없습니다.",
        ),
    )
    issue = make_issue(code, message, stage=Stage.API)
    return error_response(exc.status_code, issue, headers=dict(exc.headers or {}))


async def _not_implemented_handler(request: Request, exc: Exception) -> JSONResponse:
    log.warning("not implemented: %s %s", request.method, request.url.path)
    issue = make_issue(
        ErrorCode.NOT_IMPLEMENTED,
        "이 기능은 현재 빌드에서 아직 구현되지 않았습니다.",
        stage=Stage.API,
    )
    return error_response(501, issue)


async def _response_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """A response that does not match its declared model is a server bug: answer 500 like any
    other, and log where it failed without the offending values (they may hold user data)."""
    assert isinstance(exc, ResponseValidationError)
    where = sorted({".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()[:20]})
    log.error(
        "response of %s %s does not match its model at %s",
        request.method,
        request.url.path,
        ", ".join(where),
    )
    issue = make_issue(
        ErrorCode.INTERNAL_ERROR,
        "내부 오류가 발생했습니다. 잠시 후 다시 시도하세요.",
        stage=Stage.API,
        retryable=True,
    )
    return error_response(500, issue)


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    issue = make_issue(
        ErrorCode.INTERNAL_ERROR,
        "내부 오류가 발생했습니다. 잠시 후 다시 시도하세요.",
        stage=Stage.API,
        retryable=True,
    )
    return error_response(500, issue)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(BodyTooLarge, _body_too_large_handler)
    app.add_exception_handler(EstimatorError, _estimator_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(ResponseValidationError, _response_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(NotImplementedError, _not_implemented_handler)
    app.add_exception_handler(Exception, _unhandled_handler)


__all__ = [
    "NO_STORE",
    "STATUS_BY_CODE",
    "ApiError",
    "BodyTooLarge",
    "api_error",
    "error_response",
    "install_error_handlers",
    "not_found",
]
