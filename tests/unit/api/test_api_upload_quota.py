"""Per-owner upload quota (VRAMFORGE_MAX_UPLOAD_BYTES_PER_OWNER): the total of one owner's
unexpired uploads is capped; going over is 413 UPLOAD_TOO_LARGE (reason owner_quota)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Upload
from vramforge_api.routes import sources as sources_routes
from vramforge_api.security import owner_key_for
from vramforge_api.settings import Settings

ROW = b'{"prompt": "a", "chosen": "b", "rejected": "c"}\n'  # 48 bytes


def _upload(client: TestClient, rows: int, name: str = "train.jsonl"):  # type: ignore[no-untyped-def]
    return client.post("/api/v1/uploads", files={"file": (name, ROW * rows)})


def _owner_files(client: TestClient, settings: Settings) -> list[str]:
    owner_dir = settings.uploads_dir / owner_key_for(client.cookies["vf_owner"])
    return sorted(p.name for p in owner_dir.rglob("*") if p.is_file())


def _quota_issue(resp) -> dict:  # type: ignore[no-untyped-def]
    assert resp.status_code == 413, resp.text
    error = resp.json()["error"]
    assert error["code"] == "UPLOAD_TOO_LARGE"
    assert error["details"]["reason"] == "owner_quota"
    assert "업로드 보관 한도" in error["user_message"]
    return error


def test_default_quota_is_ten_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VRAMFORGE_MAX_UPLOAD_BYTES_PER_OWNER", raising=False)
    assert Settings().max_upload_bytes_per_owner == 10 * 1024**3
    monkeypatch.setenv("VRAMFORGE_MAX_UPLOAD_BYTES_PER_OWNER", "1048576")
    assert Settings().max_upload_bytes_per_owner == 1048576


def test_uploads_over_the_owner_quota_are_rejected_while_streaming(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    settings = settings_factory(max_upload_bytes_per_owner=48 * 15)
    alice, bob = client_factory(settings), client_factory(settings)
    assert _upload(alice, 10).status_code == 201
    error = _quota_issue(_upload(alice, 10, "second.jsonl"))
    assert (error["details"]["used_bytes"], error["details"]["quota_bytes"]) == (480, 720)
    assert _owner_files(alice, settings) == ["train.jsonl"]  # the partial file is gone
    assert _upload(alice, 5, "fits.jsonl").status_code == 201  # exactly up to the quota
    _quota_issue(_upload(alice, 1, "full.jsonl"))  # nothing left: refused before streaming
    assert _upload(bob, 10).status_code == 201  # quotas are per owner


def test_expired_uploads_do_not_count(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    settings = settings_factory(max_upload_bytes_per_owner=48 * 15)
    client = client_factory(settings)
    first = _upload(client, 10).json()["upload_id"]
    with session_scope(client.app.state.vf.sessions) as db:  # type: ignore[attr-defined]
        db.get(Upload, first).expires_at = utcnow() - timedelta(seconds=1)
    assert _upload(client, 10, "again.jsonl").status_code == 201


def test_quota_is_rechecked_when_the_upload_is_recorded(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent uploads may both pass the check made before streaming; the second one to be
    recorded is refused under the owner lock and its file removed."""
    settings = settings_factory(max_upload_bytes_per_owner=48 * 15)
    client = client_factory(settings)
    assert _upload(client, 10).status_code == 201
    monkeypatch.setattr(sources_routes, "_owner_usage", lambda db_factory, owner: 0)
    error = _quota_issue(_upload(client, 10, "racing.jsonl"))
    assert error["details"]["used_bytes"] == 480
    assert _owner_files(client, settings) == ["train.jsonl"]


def test_per_file_cap_message_is_kept_when_it_binds(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    settings = settings_factory(max_upload_bytes=100, max_upload_bytes_per_owner=10_000)
    resp = _upload(client_factory(settings), 5)
    assert resp.status_code == 413
    error = resp.json()["error"]
    assert error["code"] == "UPLOAD_TOO_LARGE" and "reason" not in error["details"]
    assert error["details"]["max_bytes"] == 100
