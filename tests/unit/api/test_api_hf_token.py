"""The server's Hugging Face token never stands in for a user's access unless the operator of a
single-user deployment shares it (plan §18, VRAMFORGE_SHARE_SERVER_HF_TOKEN)."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from api_testkit import example_request
from fastapi.testclient import TestClient

from vramforge_api.access import ACCESS_HINTS, source_access, with_access_hint
from vramforge_api.settings import Settings
from vramforge_estimator import sources
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Stage

TOKEN = "hf_" + "T" * 32


@pytest.fixture(autouse=True)
def _no_shell_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HF_TOKEN", "VRAMFORGE_HF_TOKEN", "VRAMFORGE_SHARE_SERVER_HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)


def _gated(reason: str = "gated", code: ErrorCode = ErrorCode.SOURCE_ACCESS_DENIED):
    return make_issue(
        code,
        "접근 승인이 필요한 저장소(gated)입니다.",
        stage=Stage.RESOLVING,
        component="model",
        reason=reason,
    )


@pytest.mark.parametrize(
    ("overrides", "sent"),
    [
        ({}, None),
        ({"hf_token": TOKEN}, None),
        ({"hf_token": TOKEN, "share_server_hf_token": True}, TOKEN),
        ({"share_server_hf_token": True}, None),
    ],
)
def test_source_access_sends_the_server_token_only_when_shared(
    settings_factory: Callable[..., Settings], overrides: dict[str, Any], sent: str | None
) -> None:
    access = source_access(settings_factory(**overrides), "o" * 64)
    assert access.hf_token == sent
    assert TOKEN not in repr(access)


@pytest.mark.parametrize(
    ("overrides", "mode"),
    [
        ({}, "not_configured"),
        ({"hf_token": TOKEN}, "configured_not_shared"),
        ({"hf_token": TOKEN, "share_server_hf_token": True}, "shared"),
    ],
)
def test_access_denied_issues_get_the_server_token_hint(
    settings_factory: Callable[..., Settings], overrides: dict[str, Any], mode: str
) -> None:
    settings = settings_factory(**overrides)
    hinted = with_access_hint(_gated(), settings)
    assert hinted.user_message.endswith(ACCESS_HINTS[mode])  # type: ignore[index]
    assert hinted.details["server_hf_token"] == mode and hinted.details["reason"] == "gated"
    assert with_access_hint(hinted, settings) == hinted  # applied once
    if mode != "shared":
        assert "VRAMFORGE_SHARE_SERVER_HF_TOKEN=true" in hinted.user_message
    assert TOKEN not in hinted.model_dump_json()
    # private repositories look "not found" to callers without access
    private = with_access_hint(_gated("repository_not_found", ErrorCode.SOURCE_NOT_FOUND), settings)
    assert private.details["server_hf_token"] == mode


def test_the_hints_name_only_variables_the_compose_stack_forwards() -> None:
    """An operator following the hint on the default stack must be able to act on it: every
    variable it names reaches the api and worker containers (compose.yaml x-worker-env)."""
    compose = yaml.safe_load((Path(__file__).resolve().parents[3] / "compose.yaml").read_text())
    forwarded = set(compose["x-worker-env"])
    named = {
        name for hint in ACCESS_HINTS.values() for name in re.findall(r"[A-Z][A-Z0-9_]{3,}", hint)
    }
    assert named == {"HF_TOKEN", "VRAMFORGE_SHARE_SERVER_HF_TOKEN"}
    assert named <= forwarded


@pytest.mark.parametrize(
    "issue",
    [
        _gated("permission_denied"),  # local file permissions
        _gated("download_failed"),  # network
        _gated("disabled"),
        _gated("revision_not_found", ErrorCode.SOURCE_NOT_FOUND),
        _gated(code=ErrorCode.MODEL_METADATA_UNAVAILABLE),
    ],
)
def test_issues_a_token_cannot_fix_are_left_alone(
    settings_factory: Callable[..., Settings], issue: Any
) -> None:
    assert with_access_hint(issue, settings_factory(hf_token=TOKEN)) == issue


@pytest.mark.parametrize("share", [False, True])
def test_inspect_uses_and_explains_the_token_policy(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    share: bool,
) -> None:
    seen: list[str | None] = []

    def denied(ref: Any, access: Any) -> Any:
        seen.append(access.hf_token)
        raise EstimatorError(_gated())

    monkeypatch.setattr(sources, "resolve_model", denied)
    client = client_factory(settings_factory(hf_token=TOKEN, share_server_hf_token=share))
    resp = client.post("/api/v1/sources/inspect", json={"model": example_request()["model"]})
    assert resp.status_code == 200, resp.text
    (issue,) = resp.json()["model"]["issues"]
    assert issue["code"] == "SOURCE_ACCESS_DENIED"
    mode = "shared" if share else "configured_not_shared"
    assert issue["details"]["server_hf_token"] == mode
    assert issue["user_message"].endswith(ACCESS_HINTS[mode])  # type: ignore[index]
    assert seen == [TOKEN if share else None]
    assert TOKEN not in resp.text


@pytest.mark.parametrize(
    ("overrides", "mode"),
    [
        ({}, "not_configured"),
        ({"hf_token": TOKEN}, "configured_not_shared"),
        ({"hf_token": TOKEN, "share_server_hf_token": True}, "shared"),
    ],
)
def test_health_reports_the_token_mode_never_the_value(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    overrides: dict[str, Any],
    mode: str,
) -> None:
    resp = client_factory(settings_factory(**overrides)).get("/api/v1/health")
    assert resp.json()["components"]["hf_token"] == mode
    assert TOKEN not in resp.text


def test_health_hides_the_token_mode_from_unauthenticated_callers(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings_factory(hf_token=TOKEN, access_token="open-sesame-123"))
    anonymous = client.get("/api/v1/health")
    assert anonymous.status_code == 200
    assert "hf_token" not in anonymous.json()["components"]
    authed = client.get("/api/v1/health", headers={"Authorization": "Bearer open-sesame-123"})
    assert authed.json()["components"]["hf_token"] == "configured_not_shared"
