"""Every error body is `ErrorResponse { error: Issue }`, including the responses FastAPI and
Starlette would otherwise produce in their own formats (docs/architecture.md §6)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vramforge_api.settings import Settings
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, ErrorResponse, HealthResponse

SECRET = "hf_" + "E" * 30
AID = "a" * 32


@pytest.fixture
def probe(settings: Settings, client_factory: Callable[..., TestClient]) -> TestClient:
    """A client whose app has a few extra routes that fail in framework-specific ways."""
    client = client_factory(settings, raise_server_exceptions=False)
    app = client.app

    @app.get("/api/v1/_probe/http-exception")  # type: ignore[attr-defined, untyped-decorator]
    def http_exception() -> None:
        raise HTTPException(status_code=400, detail=f"detail with {SECRET}")

    @app.get("/api/v1/_probe/teapot")  # type: ignore[attr-defined, untyped-decorator]
    def teapot() -> None:
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/api/v1/_probe/bad-response", response_model=HealthResponse)  # type: ignore[attr-defined, untyped-decorator]
    def bad_response() -> dict[str, str]:
        return {"status": SECRET}

    @app.get("/api/v1/_probe/not-implemented")  # type: ignore[attr-defined, untyped-decorator]
    def not_implemented() -> None:
        raise NotImplementedError

    @app.get("/api/v1/_probe/estimator-not-implemented")  # type: ignore[attr-defined, untyped-decorator]
    def estimator_not_implemented() -> None:
        raise EstimatorError(make_issue(ErrorCode.NOT_IMPLEMENTED, "아직 없습니다."))

    return client


def _error(resp, status: int, code: str) -> ErrorResponse:  # type: ignore[no-untyped-def]
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["cache-control"] == "no-store"
    body = ErrorResponse.model_validate(resp.json())
    assert body.error.code.value == code
    assert body.error.user_message  # Korean, display-ready
    assert "Traceback" not in resp.text and SECRET not in resp.text
    return body


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/no-such-page"),  # outside the API prefix
        ("GET", "/api/v1/no-such-route"),
        ("POST", "/api/v1/no-such-route"),
        ("DELETE", f"/api/v1/analyses/{AID}/no-such-child"),
    ],
)
def test_unknown_routes_are_error_responses(client: TestClient, method: str, path: str) -> None:
    _error(client.request(method, path), 404, "NOT_FOUND")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/api/v1/health"),
        ("PATCH", "/api/v1/analyses"),
        ("OPTIONS", "/api/v1/analyses"),
        ("POST", "/api/v1/local-roots"),
    ],
)
def test_wrong_methods_are_error_responses(client: TestClient, method: str, path: str) -> None:
    resp = client.request(method, path)
    _error(resp, 405, "INVALID_REQUEST")
    assert resp.headers["allow"]  # what the route does accept


def test_head_is_405_with_an_empty_body(client: TestClient) -> None:
    resp = client.head("/api/v1/health")
    assert resp.status_code == 405 and resp.content == b""


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "where"),
    [
        ("GET", f"/api/v1/analyses/{AID}/export?format=pdf", {}, "query.format"),
        ("GET", f"/api/v1/analyses/{AID}/events?after=-1", {}, "query.after"),
        (
            "POST",
            "/api/v1/analyses",
            {"json": {}, "headers": {"Idempotency-Key": "k" * 200}},
            "header.Idempotency-Key",
        ),
        ("POST", "/api/v1/analyses", {"json": {"model": {"reference": SECRET}}}, "body.dataset"),
        (
            "POST",
            "/api/v1/analyses",
            {"content": b"plain text", "headers": {"content-type": "text/plain"}},
            "body",
        ),
    ],
)
def test_request_validation_is_an_error_response(
    client: TestClient, method: str, path: str, kwargs: dict, where: str
) -> None:
    body = _error(client.request(method, path, **kwargs), 422, "INVALID_REQUEST")
    assert body.error.stage is not None and body.error.stage.value == "request"
    fields = body.error.details["fields"]
    assert where in {f["loc"] for f in fields} or where in body.error.user_message
    assert all(set(f) == {"loc", "type", "message"} for f in fields)  # no submitted values


def test_malformed_json_says_so(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/analyses",
        content=b'{"model": ' + SECRET.encode(),
        headers={"content-type": "application/json"},
    )
    body = _error(resp, 422, "INVALID_REQUEST")
    assert body.error.user_message == "요청 본문이 올바른 JSON이 아닙니다."


def test_http_exceptions_never_echo_their_detail(probe: TestClient) -> None:
    body = _error(probe.get("/api/v1/_probe/http-exception"), 400, "INVALID_REQUEST")
    assert body.error.user_message == "요청 형식이 올바르지 않습니다."
    _error(probe.get("/api/v1/_probe/teapot"), 418, "INVALID_REQUEST")


def test_invalid_responses_are_500_without_values_in_the_log(
    probe: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="vramforge_api.errors"):
        body = _error(probe.get("/api/v1/_probe/bad-response"), 500, "INTERNAL_ERROR")
    assert body.error.retryable is True
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "response.status" in logged and SECRET not in logged
    assert not any(r.exc_info for r in caplog.records)


def test_not_implemented_is_501(probe: TestClient) -> None:
    _error(probe.get("/api/v1/_probe/not-implemented"), 501, "NOT_IMPLEMENTED")
    _error(probe.get("/api/v1/_probe/estimator-not-implemented"), 501, "NOT_IMPLEMENTED")


def test_unhandled_exceptions_are_500(probe: TestClient, monkeypatch) -> None:
    from vramforge_estimator import compatibility

    def boom() -> None:
        raise RuntimeError(f"/srv/secret {SECRET}")

    monkeypatch.setattr(compatibility, "backend_profiles", boom)
    body = _error(probe.get("/api/v1/backend-profiles"), 500, "INTERNAL_ERROR")
    assert "/srv/secret" not in body.error.user_message
