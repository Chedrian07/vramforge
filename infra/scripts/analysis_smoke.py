"""Offline end-to-end check: one full-dataset analysis through the proxy of a running stack.

    python3 infra/scripts/analysis_smoke.py prepare DIR
    VRAMFORGE_LOCAL_SOURCES_DIR=DIR docker compose up -d --build --wait
    python3 infra/scripts/analysis_smoke.py run [BASE_URL]   # default http://127.0.0.1:8080

`prepare` assembles a tiny local model (config, tokenizer, chat template and a zero-filled
safetensors file) and a 4-row preference dataset from the repository test fixtures, so the run
needs no Hugging Face access and no GPU (plan.md §19.4 "CPU-only 배포"). `run` submits a DPO + LoRA
analysis of `local:local/...`, follows its SSE stream through the proxy (which must not compress
it), checks that the job completed with full scan coverage and every row tokenized, then deletes
the analysis. Python standard library only; set VRAMFORGE_ACCESS_TOKEN when the stack requires it.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
MODEL_FIXTURE = FIXTURES / "models" / "tiny-dense-decoder"
TOKENIZER_FIXTURE = FIXTURES / "tokenizers" / "tiny-chat"
DATASET_FIXTURE = FIXTURES / "datasets" / "preference.jsonl"
MODEL_PATH = "models/tiny-dense-decoder"
DATASET_PATH = "datasets/preference"
TERMINAL_EVENTS = frozenset({"completed", "failed", "cancelled", "needs_input"})


class SmokeError(Exception):
    """A check failed; the message says which one."""


# ---------------------------------------------------------------- prepare


def write_safetensors(header_json: Path, target: Path) -> int:
    """Write a valid safetensors file for a stored header, with zero-filled tensor data.

    Layout: u64 little-endian header length, JSON header (space-padded to 8 bytes), data.
    Returns the data size in bytes.
    """
    raw = header_json.read_bytes().strip()
    header: dict[str, Any] = json.loads(raw)
    data_bytes = max(
        (spec["data_offsets"][1] for name, spec in header.items() if name != "__metadata__"),
        default=0,
    )
    blob = raw + b" " * (-len(raw) % 8)
    with target.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        out.write(bytes(data_bytes))
    return data_bytes


def prepare(target: Path) -> None:
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise SmokeError(f"{target} must be a new or empty directory")
    model = target / MODEL_PATH
    dataset = target / DATASET_PATH
    model.mkdir(parents=True)
    dataset.mkdir(parents=True)
    shutil.copyfile(MODEL_FIXTURE / "config.json", model / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(TOKENIZER_FIXTURE / name, model / name)
    write_safetensors(MODEL_FIXTURE / "model.safetensors.header.json", model / "model.safetensors")
    shutil.copyfile(DATASET_FIXTURE, dataset / "train.jsonl")
    # api and worker run as uid 10001 and mount the directory read-only.
    for path in (target, *target.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(f"analysis-smoke: local sources ready in {target}")


# ---------------------------------------------------------------- run


def analysis_request() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model": {"source_type": "local", "reference": f"local:local/{MODEL_PATH}"},
        "dataset": {
            "source_type": "local",
            "reference": f"local:local/{DATASET_PATH}",
            "split": "train",
            "scan_mode": "full",
            "mapping": {
                "format": "preference",
                "system": "system",
                "prompt": "question",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "training": {
            "objective": "dpo",
            "strategy": "lora",
            "microbatch_per_device": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
        },
        "hardware": {"mode": "capacity_only"},
        "scope": {"include_evaluation": False, "include_checkpoint_save": False},
        "profiling": {"enabled": False},
    }


def expected_rows() -> int:
    lines = DATASET_FIXTURE.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip())


class Client:
    """Same-origin browser-like client: keeps the owner cookie, sends the CSRF header."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise SmokeError(f"base URL must be http(s): {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
        token = os.environ.get("VRAMFORGE_ACCESS_TOKEN")
        self.auth = {"Authorization": f"Bearer {token}"} if token else {}

    def send(
        self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
    ) -> Any:
        data = None if body is None else json.dumps(body).encode()
        all_headers = {**self.auth, **(headers or {})}
        if data is not None:
            all_headers["Content-Type"] = "application/json"
        if method not in {"GET", "HEAD"}:
            all_headers["X-VramForge-Request"] = "1"
        # The scheme is checked in __init__; the URL always points at the stack under test.
        request = urllib.request.Request(  # noqa: S310
            self.base_url + path, data=data, headers=all_headers, method=method
        )
        try:
            return self.opener.open(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            raise SmokeError(f"{method} {path}: HTTP {exc.code} {detail}") from None
        except OSError as exc:
            raise SmokeError(f"{method} {path}: {exc}") from None

    def send_json(
        self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
    ) -> tuple[int, Any]:
        with self.send(method, path, body, headers) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if payload else None)


def follow_events(client: Client, analysis_id: str, deadline: float) -> list[str]:
    """Read the SSE stream until a terminal event; return the event types in order."""
    path = f"/api/v1/analyses/{analysis_id}/events"
    headers = {"Accept": "text/event-stream", "Accept-Encoding": "gzip, zstd"}
    started = time.monotonic()
    seen: list[str] = []
    with client.send("GET", path, headers=headers) as stream:
        content_type = stream.headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            raise SmokeError(f"SSE: expected text/event-stream, got {content_type!r}")
        encoding = stream.headers.get("Content-Encoding")
        if encoding:
            raise SmokeError(f"SSE: the proxy compressed the event stream ({encoding})")
        event: dict[str, str] = {}
        while time.monotonic() < deadline:
            line = stream.readline()
            if not line:
                raise SmokeError(f"SSE: stream ended before a terminal event (seen {seen})")
            text = line.decode("utf-8").rstrip("\r\n")
            if text.startswith(":"):
                continue  # keepalive comment
            if text:
                key, _, value = text.partition(":")
                event[key] = value.removeprefix(" ")
                continue
            kind = event.get("event")
            event, data = {}, event.get("data", "")
            if kind is None:
                continue
            seen.append(kind)
            stage = json.loads(data).get("status") if data.startswith("{") else None
            print(f"analysis-smoke: +{time.monotonic() - started:5.1f}s {kind} {stage or ''}")
            if kind in TERMINAL_EVENTS:
                return seen
    raise SmokeError(f"SSE: no terminal event before the deadline (seen {seen})")


def check_result(status: dict[str, Any], rows: int) -> str:
    """Assert a completed, fully covered analysis; return a one-line summary."""
    if status.get("status") != "COMPLETED":
        raise SmokeError(f"analysis ended {status.get('status')}: {status.get('error')}")
    result = status.get("result") or {}
    axes = result.get("status") or {}
    if axes.get("scan_coverage") != "complete":
        raise SmokeError(f"scan coverage is {axes.get('scan_coverage')!r}, expected 'complete'")
    if axes.get("data_preservation") != "verified":
        raise SmokeError(f"data preservation is {axes.get('data_preservation')!r}")
    scan = result.get("dataset_scan") or {}
    counts = {key: scan.get(key) for key in ("rows_seen", "rows_ok", "rows_failed")}
    if counts != {"rows_seen": rows, "rows_ok": rows, "rows_failed": 0}:
        raise SmokeError(f"expected all {rows} rows tokenized, got {counts}")
    if result.get("errors"):
        raise SmokeError(f"result errors: {[e.get('code') for e in result['errors']]}")
    scenarios = (result.get("memory") or {}).get("scenarios") or []
    if not scenarios:
        raise SmokeError("result has no memory scenario")
    required = (scenarios[0].get("recommendation") or {}).get(
        "required_total_device_capacity_bytes"
    )
    return f"{rows}/{rows} rows, axes {axes}, required capacity {required} bytes"


def run(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    client = Client(base_url, timeout_s=min(60.0, timeout_s))
    rows = expected_rows()
    code, created = client.send_json(
        "POST",
        "/api/v1/analyses",
        analysis_request(),
        headers={"Idempotency-Key": f"analysis-smoke-{uuid.uuid4().hex}"},
    )
    if code != 202 or not created or not created.get("analysis_id"):
        raise SmokeError(f"POST /api/v1/analyses: expected 202 with an id, got {code} {created}")
    analysis_id = created["analysis_id"]
    print(f"analysis-smoke: submitted {analysis_id}")
    try:
        events = follow_events(client, analysis_id, deadline)
        if events[-1] != "completed":
            raise SmokeError(f"terminal event {events[-1]!r}, expected 'completed'")
        _, status = client.send_json("GET", f"/api/v1/analyses/{analysis_id}")
        print(f"analysis-smoke: ok   {check_result(status or {}, rows)}")
    finally:
        try:
            with client.send("DELETE", f"/api/v1/analyses/{analysis_id}") as response:
                print(f"analysis-smoke: deleted {analysis_id} (HTTP {response.status})")
        except SmokeError as exc:  # cleanup only; never hide the real outcome
            print(f"analysis-smoke: could not delete {analysis_id}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analysis_smoke.py", description="Offline full analysis through the VRAMForge proxy."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="build a local-sources directory from test fixtures")
    prep.add_argument("directory", type=Path)
    go = sub.add_parser("run", help="run one analysis through the proxy and check the result")
    port = os.environ.get("VRAMFORGE_PORT", "8080")
    go.add_argument("base_url", nargs="?", default=f"http://127.0.0.1:{port}")
    go.add_argument("--timeout", type=float, default=300.0, help="overall deadline in seconds")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.directory)
        else:
            run(args.base_url, args.timeout)
    except SmokeError as exc:
        print(f"analysis-smoke: FAIL {exc}", file=sys.stderr)
        return 1
    if args.command == "run":
        print("analysis-smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
