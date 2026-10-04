"""Plan §21 example end-to-end through the API with the real modules (Hugging Face Hub access).

Run with VRAMFORGE_NETWORK_TESTS=1. Downloads only config, tokenizer files, safetensors headers and
the dataset files into a temporary data directory.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import yaml
from fastapi.testclient import TestClient
from integration_testkit import request_body

from vramforge_api.settings import Settings

pytestmark = [pytest.mark.network, pytest.mark.slow]


def _run(client: TestClient, body: dict) -> dict:
    created = client.post("/api/v1/analyses", json=body)
    assert created.status_code == 202, created.text
    return client.get(f"/api/v1/analyses/{created.json()['analysis_id']}").json()


def test_plan_example_grpo_then_dpo(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    settings = settings_factory(job_timeout_s=1800)
    client = client_factory(settings)

    grpo = _run(client, request_body())
    assert grpo["status"] == "COMPLETED", grpo["error"]
    result = grpo["result"]
    assert result["source_manifests"]["model"]["resolved_revision"]  # pinned commit
    assert result["dataset_scan"]["coverage"] == "complete"
    assert result["dataset_scan"]["rows_seen"] == 4656
    prompt = next(b for b in result["dataset_scan"]["branches"] if b["branch"] == "prompt")
    assert prompt["stats"]["max"] == 268  # research golden: user-only prompt, empty system omitted
    assert result["status"]["training_readiness"] == "conditional"  # reward unspecified
    assert {s["scenario_id"] for s in result["memory"]["scenarios"]} == {
        "budget_1024",
        "budget_2048",
        "budget_4096",
        "budget_8192",
    }
    assert (
        client.get(
            f"/api/v1/analyses/{grpo['analysis_id']}/export?format=trainer-config"
        ).status_code
        == 409
    )

    body = request_body(**{"training.objective": "dpo", "training.gradient_accumulation_steps": 8})
    dpo = _run(client, body)
    assert dpo["status"] == "COMPLETED", dpo["error"]
    pair = next(b for b in dpo["result"]["dataset_scan"]["branches"] if b["branch"] == "pair_max")
    assert pair["stats"]["max"] == 2272  # research golden (row 2355 / 3169)
    assert dpo["result"]["status"]["training_readiness"] == "ready"
    trainer = client.get(f"/api/v1/analyses/{dpo['analysis_id']}/export?format=trainer-config")
    assert trainer.status_code == 200
    args = yaml.safe_load(trainer.content)["trl"]["args"]
    assert args["max_length"] is None and args["loss_type"] == ["sigmoid"]

    changed = dict(body)
    changed["training"] = {**body["training"], "lora": {**body["training"]["lora"], "r": 64}}
    scenario = client.post(
        f"/api/v1/analyses/{dpo['analysis_id']}/scenarios", json={"request": changed}
    ).json()
    assert scenario["requires_reanalysis"] is False
    old_high = dpo["result"]["memory"]["scenarios"][0]["devices"][0]["scenario_high_bytes"]
    new_high = scenario["result"]["memory"]["scenarios"][0]["devices"][0]["scenario_high_bytes"]
    assert new_high > old_high  # a larger adapter never needs less memory

    # the recomputed scenario can be exported as shown (trainer-config: ready DPO)
    exported = client.post(
        f"/api/v1/analyses/{dpo['analysis_id']}/scenarios/export",
        json={"request": changed, "format": "trainer-config"},
    )
    assert exported.status_code == 200, exported.text
    assert yaml.safe_load(exported.content)["peft"]["r"] == 64
    assert dpo["expires_at"] is not None and dpo["request"]["training"]["objective"] == "dpo"
