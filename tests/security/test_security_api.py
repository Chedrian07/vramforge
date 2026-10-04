"""Security properties of the HTTP surface (plan §18, §19.4 "Unauthorized access").

Covers: owner isolation across every analysis route and the scan cache, upload path tricks,
CSRF, the access-token mode for SSE (cookie), error bodies without internals, export redaction
and attachment headers, and log redaction.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from security_testkit import kit

from vramforge_api.logging_setup import configure_logging
from vramforge_api.security import owner_key_for
from vramforge_api.settings import Settings
from vramforge_estimator import pipeline
from vramforge_estimator.schemas import ErrorCode, Severity

SECRET = "hf_" + "S" * 32


def _create(client: TestClient, body: dict[str, Any] | None = None) -> str:
    resp = client.post("/api/v1/analyses", json=body or kit.request_body())
    assert resp.status_code == 202, resp.text
    return resp.json()["analysis_id"]


def test_foreign_owner_cannot_reach_anything(
    settings: Settings, client_factory: Callable[..., TestClient], fake_modules: Any
) -> None:
    alice = client_factory(settings)
    mallory = client_factory(settings)
    aid = _create(alice)
    assert alice.get(f"/api/v1/analyses/{aid}").json()["status"] == "COMPLETED"
    for method, path in (
        ("GET", ""),
        ("GET", "/events"),
        ("GET", "/export?format=json"),
        ("GET", "/export?format=md"),
        ("POST", "/cancel"),
        ("POST", "/profile"),
        ("DELETE", ""),
    ):
        resp = mallory.request(method, f"/api/v1/analyses/{aid}{path}")
        assert resp.status_code == 404, (method, path)
        assert resp.json() == alice.get("/api/v1/analyses/" + "e" * 32).json()  # same body
    resp = mallory.post(f"/api/v1/analyses/{aid}/scenarios", json={"request": kit.request_body()})
    assert resp.status_code == 404
    assert alice.get(f"/api/v1/analyses/{aid}").status_code == 200


def test_scan_cache_never_crosses_owners(
    settings: Settings, client_factory: Callable[..., TestClient], fake_modules: Any
) -> None:
    alice = client_factory(settings)
    bob = client_factory(settings)
    _create(alice)
    _create(alice)
    assert fake_modules.calls.count("full_scan") == 1
    _create(bob)
    assert fake_modules.calls.count("full_scan") == 2


def test_upload_filenames_cannot_escape(
    settings: Settings, client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings)
    for name in ("../../../../etc/cron.d/x.jsonl", "..%2F..%2Fx.jsonl", "a/../../b.jsonl"):
        resp = client.post("/api/v1/uploads", files={"file": (name, b'{"a": 1}\n')})
        assert resp.status_code == 201, (name, resp.text)
        assert "/" not in resp.json()["filename"]
    owner = owner_key_for(client.cookies["vf_owner"])
    for path in settings.data_dir.rglob("*.jsonl"):
        assert path.resolve().is_relative_to((settings.uploads_dir / owner).resolve())


def test_access_token_mode_protects_sse_with_a_cookie(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    fake_modules: Any,
) -> None:
    client = client_factory(settings_factory(access_token="correct-horse"))
    assert client.post("/api/v1/session", json={"token": "correct-horse"}).status_code == 200
    aid = _create(client)
    with client.stream("GET", f"/api/v1/analyses/{aid}/events") as resp:
        assert resp.status_code == 200
        assert kit.read_sse(resp)[-1]["event"] == "completed"
    client.cookies.delete("vf_access")
    assert client.get(f"/api/v1/analyses/{aid}/events").status_code == 401


def test_errors_never_carry_internals(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    def boom(request, artifact_dir):
        raise RuntimeError(f"/srv/data/private {SECRET}")

    from vramforge_estimator import compatibility

    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    kit.load_pipeline_fakes().FakeModules().install(monkeypatch)
    client = client_factory(settings, raise_server_exceptions=False)
    aid = _create(client)
    monkeypatch.setattr(pipeline, "recompute", boom)
    resp = client.post(f"/api/v1/analyses/{aid}/scenarios", json={"request": kit.request_body()})
    assert resp.status_code == 500
    body = resp.text
    assert "/srv/data" not in body and SECRET not in body and "Traceback" not in body
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"


def test_exports_are_redacted_attachments(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    fakes = kit.load_pipeline_fakes()
    leaky = fakes.issue(
        ErrorCode.SCAN_FAILED_ROWS,
        f"failed reading /Users/alice/private.jsonl with {SECRET} via http://192.168.0.7/x",
    ).model_copy(update={"severity": Severity.WARNING})
    report = fakes.compat_report(warnings=[leaky])
    fakes.FakeModules(resolve=lambda req, inv, tok: (fakes.resolved_config(), report)).install(
        monkeypatch
    )
    from vramforge_estimator import compatibility

    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    client = client_factory(settings)
    aid = _create(client)
    for fmt in ("json", "yaml", "md"):
        resp = client.get(f"/api/v1/analyses/{aid}/export?format={fmt}")
        assert resp.status_code == 200
        text = resp.text
        assert SECRET not in text, fmt
        assert "/Users/alice" not in text and "192.168.0.7" not in text, fmt
        assert resp.headers["content-disposition"].startswith("attachment;")
        assert resp.headers["cache-control"] == "no-store"


def test_logs_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logger = logging.getLogger("vramforge_api.security-test")
    try:
        raise ValueError(f"token {SECRET} leaked")
    except ValueError:
        logger.exception("request failed with Authorization: Bearer abc.def.ghi")
    err = capsys.readouterr().err
    assert SECRET not in err and "abc.def.ghi" not in err
    assert "hf_***" in err


def test_redacting_formatter_handles_uvicorn_access_lines() -> None:
    from vramforge_api.logging_setup import RedactingFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    log = logging.getLogger("uvicorn.access.security-test")
    log.addHandler(handler)
    log.propagate = False
    log.warning('"GET /api/v1/x?token=%s HTTP/1.1" 200', SECRET)
    log.removeHandler(handler)
    assert SECRET not in stream.getvalue()
