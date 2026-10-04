"""POST /analyses/{id}/scenarios/export: recompute exactly like /scenarios, then export that
result (plan §4.2, §12.4)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from integration_testkit import ready_grpo_body, request_body

from vramforge_api.settings import Settings


def _completed(client: TestClient, body: dict[str, Any]) -> str:
    created = client.post("/api/v1/analyses", json=body)
    assert created.status_code == 202, created.text
    aid = created.json()["analysis_id"]
    assert client.get(f"/api/v1/analyses/{aid}").json()["status"] == "COMPLETED"
    return aid


def _changed(**overrides: Any) -> dict[str, Any]:
    return ready_grpo_body(**{"training.lora.r": 64, **overrides})


def test_export_matches_the_recomputed_scenario(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings)
    aid = _completed(client, ready_grpo_body())
    scans = modules.calls.count("full_scan")
    shown = client.post(
        f"/api/v1/analyses/{aid}/scenarios",
        json={"request": _changed(), "client_fingerprint": "form-3"},
    ).json()
    assert shown["requires_reanalysis"] is False

    resp = client.post(
        f"/api/v1/analyses/{aid}/scenarios/export",
        json={"request": _changed(), "format": "json", "client_fingerprint": "form-3"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["content-disposition"] == 'attachment; filename="analysis.json"'
    assert resp.headers["cache-control"] == "no-store"
    exported = json.loads(resp.content)
    for key in ("analysis_fingerprint", "requested_config", "resolved_config", "memory", "status"):
        assert exported[key] == shown["result"][key], key
    assert exported["resolved_config"]["lora"]["r"] == 64
    assert exported["analysis_fingerprint"] == shown["fingerprint"]
    assert modules.calls.count("full_scan") == scans  # recomputed, not re-tokenized

    stored = json.loads(client.get(f"/api/v1/analyses/{aid}/export?format=json").content)
    assert stored["resolved_config"]["lora"]["r"] == 16  # the analysis itself is unchanged


@pytest.mark.parametrize(
    ("fmt", "media", "filename"),
    [
        ("yaml", "application/yaml", "resolved-plan.yaml"),
        ("md", "text/markdown", "report.md"),
        ("trainer-config", "application/yaml", "trainer-config.yaml"),
    ],
)
def test_every_export_format_is_available_for_a_scenario(
    settings: Settings,
    client_factory: Callable[..., TestClient],
    modules: Any,
    fmt: str,
    media: str,
    filename: str,
) -> None:
    client = client_factory(settings)
    aid = _completed(client, ready_grpo_body())
    resp = client.post(
        f"/api/v1/analyses/{aid}/scenarios/export", json={"request": _changed(), "format": fmt}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith(media)
    assert resp.headers["content-disposition"] == f'attachment; filename="{filename}"'
    if fmt == "trainer-config":
        assert yaml.safe_load(resp.content)["peft"]["r"] == 64


def test_trainer_config_needs_a_ready_scenario(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings)
    aid = _completed(client, ready_grpo_body())
    unready = _changed(**{"grpo.completion_budget": None})  # several budgets: not executable
    resp = client.post(
        f"/api/v1/analyses/{aid}/scenarios/export",
        json={"request": unready, "format": "trainer-config"},
    )
    assert resp.status_code == 409
    assert "ready일 때만" in resp.json()["error"]["user_message"]
    plan = client.post(
        f"/api/v1/analyses/{aid}/scenarios/export", json={"request": unready, "format": "yaml"}
    )
    assert plan.status_code == 200  # the plan itself is always available


def test_scenarios_that_need_reanalysis_are_not_exported(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings)
    aid = _completed(client, ready_grpo_body())
    dpo = ready_grpo_body(**{"training.objective": "dpo"})
    resp = client.post(f"/api/v1/analyses/{aid}/scenarios/export", json={"request": dpo})
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "REANALYSIS_REQUIRED"
    assert error["details"]["reasons"] and error["details"]["reasons"][0] in error["user_message"]


def test_scenario_export_checks_owner_state_csrf_and_format(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    alice = client_factory(settings)
    mallory = client_factory(settings)
    aid = _completed(alice, ready_grpo_body())
    url = f"/api/v1/analyses/{aid}/scenarios/export"
    payload = {"request": _changed(), "format": "json"}
    assert mallory.post(url, json=payload).status_code == 404
    no_csrf = alice.post(url, json=payload, headers={"X-VramForge-Request": "0"})
    assert no_csrf.status_code == 403
    bad = alice.post(url, json={"request": _changed(), "format": "pdf"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_REQUEST"
    extra = alice.post(url, json={**payload, "unexpected": 1})
    assert extra.status_code == 422


def test_running_analyses_cannot_export_scenarios(
    settings: Settings, client_factory: Callable[..., TestClient], modules: Any
) -> None:
    client = client_factory(settings, queue_is_async=True)  # the job stays queued
    aid = client.post("/api/v1/analyses", json=request_body()).json()["analysis_id"]
    resp = client.post(f"/api/v1/analyses/{aid}/scenarios/export", json={"request": request_body()})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
