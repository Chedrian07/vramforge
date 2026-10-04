"""End-to-end API flow (plan §19.4): upload → analyze → status/SSE → recompute → export →
cancel → delete, with the real API, worker task and pipeline orchestration."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import yaml
from fastapi.testclient import TestClient
from integration_testkit import load_pipeline_fakes, read_sse, ready_grpo_body, request_body

from vramforge_api.security import owner_key_for
from vramforge_api.settings import Settings
from vramforge_estimator import compatibility, inspection
from vramforge_estimator.errors import CancelledError
from vramforge_estimator.schemas import Stage, TrainingReadiness
from vramforge_worker import tasks

fakes = load_pipeline_fakes()

JSONL = b'{"system": "", "question": "q", "chosen": "a", "rejected": "b"}\n' * 3


def _analyze(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    created = client.post("/api/v1/analyses", json=body)
    assert created.status_code == 202, created.text
    aid = created.json()["analysis_id"]
    status = client.get(f"/api/v1/analyses/{aid}").json()
    return status


def test_full_flow_from_upload_to_delete(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings)
    upload = client.post("/api/v1/uploads", files={"file": ("train.jsonl", JSONL)}).json()
    body = ready_grpo_body(
        **{"dataset.source_type": "upload", "dataset.reference": upload["reference"]}
    )
    status = _analyze(client, body)
    aid = status["analysis_id"]
    assert status["status"] == "COMPLETED"
    result = status["result"]
    assert result["status"] == {
        "scan_coverage": "complete",
        "data_preservation": "verified",
        "training_readiness": "ready",
        "estimate_evidence": "analytic",
        "hardware_fit": "not_evaluated",
    }
    assert result["memory"]["scenarios"][0]["devices"][0]["scenario_high_bytes"] > 0
    assert status["progress"]["stage"] == "COMPLETED"

    # SSE replay from the beginning ends with the terminal event
    with client.stream("GET", f"/api/v1/analyses/{aid}/events") as resp:
        events = read_sse(resp)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "progress" and kinds[-1] == "completed"
    stages = [json.loads(e["data"])["status"] for e in events if e["event"] == "progress"]
    assert stages[:2] == ["QUEUED", "RESOLVING"] and "ESTIMATING" in stages
    assert "partial_result" in kinds
    # reconnect after the last event: nothing left, EventSource is told to stop
    last = events[-1]["id"]
    assert (
        client.get(f"/api/v1/analyses/{aid}/events", headers={"Last-Event-ID": last}).status_code
        == 204
    )

    # recompute: r changes reuse the scan; objective changes need a new analysis
    scans_before = modules.calls.count("full_scan")
    changed = ready_grpo_body(
        **{
            "dataset.source_type": "upload",
            "dataset.reference": upload["reference"],
            "training.lora.r": 64,
        }
    )
    scenario = client.post(
        f"/api/v1/analyses/{aid}/scenarios",
        json={"request": changed, "client_fingerprint": "form-7"},
    ).json()
    assert scenario["requires_reanalysis"] is False
    assert scenario["client_fingerprint"] == "form-7"
    assert scenario["result"]["resolved_config"]["lora"]["r"] == 64
    assert modules.calls.count("full_scan") == scans_before  # no re-tokenization
    dpo = dict(changed)
    dpo["training"] = {**changed["training"], "objective": "dpo"}
    redo = client.post(f"/api/v1/analyses/{aid}/scenarios", json={"request": dpo}).json()
    assert redo["requires_reanalysis"] is True and redo["reanalysis_reasons"]

    # exports
    for fmt, filename, media in (
        ("json", "analysis.json", "application/json"),
        ("yaml", "resolved-plan.yaml", "application/yaml"),
        ("md", "report.md", "text/markdown"),
        ("trainer-config", "trainer-config.yaml", "application/yaml"),
    ):
        resp = client.get(f"/api/v1/analyses/{aid}/export?format={fmt}")
        assert resp.status_code == 200, (fmt, resp.text)
        assert resp.headers["content-type"].startswith(media)
        assert resp.headers["content-disposition"] == f'attachment; filename="{filename}"'
        assert resp.headers["x-content-type-options"] == "nosniff"
    exported = json.loads(client.get(f"/api/v1/analyses/{aid}/export?format=json").content)
    assert exported["memory"] == result["memory"]  # export matches the stored numbers
    trainer = yaml.safe_load(
        client.get(f"/api/v1/analyses/{aid}/export?format=trainer-config").content
    )
    assert trainer["trl"]["args"]["max_completion_length"] == 1024

    # a second analysis of the same owner reuses the complete scan
    second = _analyze(client, body)
    assert second["status"] == "COMPLETED"
    assert modules.calls.count("full_scan") == scans_before

    # deleting the first keeps the upload (still referenced by the second)
    owner = owner_key_for(client.cookies["vf_owner"])
    first_dir = settings.data_dir / "artifacts" / owner / aid
    assert first_dir.exists()
    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 204
    assert not first_dir.exists()
    upload_dir = settings.uploads_dir / owner / upload["upload_id"]
    assert upload_dir.exists()
    assert client.delete(f"/api/v1/analyses/{second['analysis_id']}").status_code == 204
    assert not upload_dir.exists()


def test_conditional_grpo_has_plan_but_no_trainer_config(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings)
    status = _analyze(client, request_body())  # plan example: no budget, reward unspecified
    aid = status["analysis_id"]
    assert status["status"] == "COMPLETED"
    assert status["result"]["status"]["training_readiness"] == "conditional"
    assert client.get(f"/api/v1/analyses/{aid}/export?format=yaml").status_code == 200
    refused = client.get(f"/api/v1/analyses/{aid}/export?format=trainer-config")
    assert refused.status_code == 409
    assert "ready일 때만" in refused.json()["error"]["user_message"]


def test_needs_input_then_explicit_mapping(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any, monkeypatch
) -> None:
    monkeypatch.setattr(
        inspection,
        "inspect_dataset",
        lambda *a, **k: fakes.dataset_inspection(mapping_ambiguous=True),
    )
    client = client_factory(settings)
    status = _analyze(client, ready_grpo_body(**{"dataset.mapping": None}))
    assert status["status"] == "NEEDS_INPUT"
    assert status["result"]["needs_input"]["choices"][0]["field"] == "dataset.mapping"
    assert status["error"]["code"] == "COLUMN_MAPPING_REQUIRED"
    with client.stream("GET", f"/api/v1/analyses/{status['analysis_id']}/events") as resp:
        assert read_sse(resp)[-1]["event"] == "needs_input"
    again = _analyze(client, ready_grpo_body())  # the example request has an explicit mapping
    assert again["status"] == "COMPLETED"


def test_running_job_cancel_end_to_end(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    started = threading.Event()

    def waiting_scan(stream, adapter, ctx, **kwargs):
        started.set()
        deadline = time.monotonic() + 10
        while not ctx.cancelled():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raise CancelledError(Stage.TOKENIZING)

    fakes.FakeModules(full_scan=waiting_scan).install(monkeypatch)
    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    client = client_factory(settings, queue_is_async=True)  # the job waits in the queue
    aid = client.post("/api/v1/analyses", json=request_body()).json()["analysis_id"]
    runner = threading.Thread(target=tasks.run_analysis, args=(aid, 1))
    runner.start()
    assert started.wait(10)
    cancel = client.post(f"/api/v1/analyses/{aid}/cancel").json()
    assert cancel["status"] == "CANCEL_REQUESTED"
    runner.join(10)
    assert not runner.is_alive()
    final = client.get(f"/api/v1/analyses/{aid}").json()
    assert final["status"] == "CANCELLED"
    assert final["result"]["status"]["scan_coverage"] == "partial"
    with client.stream("GET", f"/api/v1/analyses/{aid}/events") as resp:
        kinds = [e["event"] for e in read_sse(resp)]
    assert "progress" in kinds and kinds[-1] == "cancelled"


def test_unsupported_architecture_is_metadata_only_end_to_end(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    report = fakes.compat_report(readiness=TrainingReadiness.UNSUPPORTED, grade=None, adapter=None)
    fakes.FakeModules(resolve=lambda req, inv, tok: (None, report)).install(monkeypatch)
    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    client = client_factory(settings)
    status = _analyze(client, request_body())
    assert status["status"] == "COMPLETED"
    assert status["result"]["status"]["estimate_evidence"] == "metadata_only"
    assert status["result"]["memory"] is None
    assert any(e["code"] == "UNSUPPORTED_ARCHITECTURE" for e in status["result"]["errors"])
    aid = status["analysis_id"]
    assert client.get(f"/api/v1/analyses/{aid}/export?format=trainer-config").status_code == 409
    assert client.post(f"/api/v1/analyses/{aid}/profile").status_code == 503


def test_stub_modules_give_an_honest_partial_result(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    """With an unimplemented module the job ends PARTIAL with the reason, never with numbers."""
    fakes.FakeModules(full_scan=fakes.raising(NotImplementedError())).install(monkeypatch)
    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])
    client = client_factory(settings)
    status = _analyze(client, request_body())
    assert status["status"] == "PARTIAL"
    assert status["error"]["details"]["reason"] == "not_implemented"
    assert status["result"]["memory"] is None and status["result"]["batch_plan"] is None
