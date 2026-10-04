"""Offline tests of the infra helper scripts (infra/scripts)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

from vramforge_estimator.schemas import AnalysisRequest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "infra" / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"infra_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


env_check = load("check_env_example")
smoke = load("analysis_smoke")


# ---------------------------------------------------------------- check_env_example.py


def write_pair(tmp_path: Path, compose: str, example: str) -> tuple[Path, Path]:
    compose_path, example_path = tmp_path / "compose.yaml", tmp_path / ".env.example"
    compose_path.write_text(compose, encoding="utf-8")
    example_path.write_text(example, encoding="utf-8")
    return compose_path, example_path


COMPOSE = """
services:
  app:
    image: ${APP_IMAGE:-busybox:1}
    ports: ["${BIND:-127.0.0.1}:${PORT:-8080}:80"]
    environment:
      TOKEN:
      SHELL_ONLY: $${NOT_INTERPOLATED}
"""


BASE = "BIND=127.0.0.1\nPORT=8080\n"


def test_env_example_matches_the_repository_compose_file() -> None:
    assert env_check.check(ROOT / "compose.yaml", ROOT / ".env.example") == []


def test_env_example_accepts_matching_defaults(tmp_path: Path) -> None:
    pair = write_pair(
        tmp_path, COMPOSE, "APP_IMAGE=busybox:1\nBIND=127.0.0.1\nPORT=8080\n# TOKEN=\n"
    )
    assert env_check.check(*pair) == []


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        (f"APP_IMAGE=busybox:2\n{BASE}# TOKEN=\n", "APP_IMAGE: expected"),
        ("APP_IMAGE=busybox:1\n# BIND=127.0.0.1\nPORT=8080\n# TOKEN=\n", "BIND: expected"),
        ("APP_IMAGE=busybox:1\nBIND=127.0.0.1\n# TOKEN=\n", "PORT: missing"),
        (f"APP_IMAGE=busybox:1\n{BASE}", "TOKEN: pass-through"),
        (f"APP_IMAGE=busybox:1\n{BASE}TOKEN=\n", "TOKEN: comment it out"),
        (f"APP_IMAGE=busybox:1\n{BASE}# TOKEN=\nEXTRA=1\n", "EXTRA: documented"),
    ],
)
def test_env_example_reports_drift(tmp_path: Path, example: str, expected: str) -> None:
    errors = env_check.check(*write_pair(tmp_path, COMPOSE, example))
    assert any(error.startswith(expected) for error in errors), errors


def test_escaped_dollar_is_not_a_compose_variable() -> None:
    interpolated, passthrough = env_check.compose_variables(COMPOSE)
    assert "NOT_INTERPOLATED" not in interpolated
    assert passthrough == {"TOKEN"}


# ---------------------------------------------------------------- analysis_smoke.py


def test_safetensors_file_has_the_stored_header_and_zero_data(tmp_path: Path) -> None:
    header_path = smoke.MODEL_FIXTURE / "model.safetensors.header.json"
    target = tmp_path / "model.safetensors"
    data_bytes = smoke.write_safetensors(header_path, target)
    raw = target.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    assert length % 8 == 0
    header = json.loads(raw[8 : 8 + length])
    assert header == json.loads(header_path.read_text(encoding="utf-8"))
    ends = [spec["data_offsets"][1] for key, spec in header.items() if key != "__metadata__"]
    assert data_bytes == max(ends) == 213_632  # tiny-dense-decoder PROVENANCE.md
    assert len(raw) == 8 + length + data_bytes
    assert raw[8 + length :] == bytes(data_bytes)


def test_prepare_builds_a_readable_tree_in_an_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "local"
    smoke.prepare(target)
    model, dataset = target / smoke.MODEL_PATH, target / smoke.DATASET_PATH
    assert {p.name for p in model.iterdir()} == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    }
    assert (dataset / "train.jsonl").read_bytes() == smoke.DATASET_FIXTURE.read_bytes()
    for path in (target, *target.rglob("*")):
        assert path.stat().st_mode & 0o004, path  # readable by the container's uid 10001
    with pytest.raises(smoke.SmokeError, match="new or empty"):
        smoke.prepare(target)


def test_request_is_a_valid_local_analysis_request() -> None:
    request = AnalysisRequest.model_validate(smoke.analysis_request())
    assert request.model.reference == f"local:local/{smoke.MODEL_PATH}"
    assert request.dataset.reference == f"local:local/{smoke.DATASET_PATH}"
    assert smoke.expected_rows() == 4


def completed_status(rows: int = 4, **axes: str) -> dict[str, Any]:
    return {
        "status": "COMPLETED",
        "result": {
            "status": {"scan_coverage": "complete", "data_preservation": "verified", **axes},
            "dataset_scan": {"rows_seen": rows, "rows_ok": rows, "rows_failed": 0},
            "errors": [],
            "memory": {"scenarios": [{"recommendation": {}}]},
        },
    }


def test_check_result_accepts_a_complete_analysis() -> None:
    assert smoke.check_result(completed_status(), 4).startswith("4/4 rows")


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ({"status": "FAILED", "error": {"code": "X"}}, "analysis ended FAILED"),
        (completed_status(scan_coverage="partial"), "scan coverage"),
        (completed_status(data_preservation="violated"), "data preservation"),
        (completed_status(rows=3), "expected all 4 rows"),
    ],
)
def test_check_result_rejects_incomplete_analyses(status: dict[str, Any], message: str) -> None:
    with pytest.raises(smoke.SmokeError, match=message):
        smoke.check_result(status, 4)


class FakeStream(io.BytesIO):
    def __init__(self, payload: str, headers: dict[str, str]) -> None:
        super().__init__(payload.encode())
        self.headers = headers


class FakeClient:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream

    def send(self, method: str, path: str, **_: Any) -> FakeStream:
        assert (method, path) == ("GET", "/api/v1/analyses/a1/events")
        return self.stream


SSE = (
    ": keepalive\n\n"
    'id: 1\nevent: progress\ndata: {"status": "QUEUED"}\n\n'
    'id: 2\nevent: completed\ndata: {"status": "COMPLETED"}\n\n'
)


def follow(payload: str, headers: dict[str, str]) -> list[str]:
    client = FakeClient(FakeStream(payload, headers))
    return smoke.follow_events(client, "a1", deadline=float("inf"))


def test_follow_events_reads_until_the_terminal_event() -> None:
    events = follow(SSE, {"Content-Type": "text/event-stream; charset=utf-8"})
    assert events == ["progress", "completed"]


@pytest.mark.parametrize(
    ("payload", "headers", "message"),
    [
        (SSE, {"Content-Type": "text/event-stream", "Content-Encoding": "gzip"}, "compressed"),
        (SSE, {"Content-Type": "application/json"}, "expected text/event-stream"),
        (SSE.split("id: 2")[0], {"Content-Type": "text/event-stream"}, "ended before"),
    ],
)
def test_follow_events_rejects_bad_streams(
    payload: str, headers: dict[str, str], message: str
) -> None:
    with pytest.raises(smoke.SmokeError, match=message):
        follow(payload, headers)


# ---------------------------------------------------------------- smoke-test.sh


class StackStub(BaseHTTPRequestHandler):
    health: ClassVar[str] = '{"status":"ok","components":{"db":"ok","redis":"ok","worker":"ok"}}'
    page_headers: ClassVar[dict[str, str]] = {}

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.reply("text/plain", "ok")
        elif self.path == "/api/v1/health":
            self.reply("application/json", self.health)
        elif self.path == "/":
            self.reply("text/html; charset=utf-8", "<html></html>", self.page_headers)
        else:
            self.send_error(404)

    def reply(self, content_type: str, body: str, extra: dict[str, str] | None = None) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "frame-ancestors 'none'",
}


@pytest.fixture
def stack() -> Iterator[tuple[str, type[StackStub]]]:
    handler = type("Stub", (StackStub,), {"page_headers": dict(SECURITY_HEADERS)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", handler
    finally:
        server.shutdown()
        server.server_close()


def run_smoke_test(base_url: str) -> subprocess.CompletedProcess[str]:
    sh, curl = shutil.which("sh"), shutil.which("curl")
    if sh is None or curl is None:
        pytest.skip("smoke-test.sh needs sh and curl")
    return subprocess.run(  # noqa: S603 - fixed script, local stub server
        [sh, str(SCRIPTS / "smoke-test.sh"), base_url],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "SMOKE_HEALTH_WAIT_S": "0"},
        check=False,
    )


def test_smoke_test_passes_on_a_healthy_stack(stack: tuple[str, type[StackStub]]) -> None:
    result = run_smoke_test(stack[0])
    assert result.returncode == 0, result.stderr
    assert "all checks passed" in result.stdout


def test_smoke_test_fails_when_a_component_is_degraded(
    stack: tuple[str, type[StackStub]],
) -> None:
    base_url, handler = stack
    handler.health = '{"status":"degraded","components":{"db":"ok","worker":"none"}}'
    result = run_smoke_test(base_url)
    assert result.returncode == 1
    assert "not ok" in result.stderr and '"worker":"none"' in result.stderr


def test_smoke_test_fails_without_security_headers(stack: tuple[str, type[StackStub]]) -> None:
    base_url, handler = stack
    del handler.page_headers["Content-Security-Policy"]
    result = run_smoke_test(base_url)
    assert result.returncode == 1
    assert "missing Content-Security-Policy" in result.stderr
