"""Settings defaults work inside compose without a .env file; secrets come from env aliases."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vramforge_api.settings import Settings, parse_local_roots


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("VRAMFORGE_") or key == "HF_TOKEN":
            monkeypatch.delenv(key, raising=False)


def test_defaults_target_the_compose_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    s = Settings()
    assert s.database_url == "postgresql+psycopg://vramforge:vramforge@postgres:5432/vramforge"
    assert s.redis_url == "redis://redis:6379/0"
    assert s.data_dir == Path("/data")
    assert s.artifacts_dir == Path("/data/artifacts")
    assert s.uploads_dir == Path("/data/uploads")
    assert s.hf_home == Path("/data/hf")
    assert s.local_root_map == {"local": Path("/sources/local")}
    assert s.max_concurrent_jobs_per_owner == 2
    assert s.retention_days == 7
    assert s.inspect_timeout_s == 60
    assert s.max_upload_bytes == 2 * 1024**3
    assert s.job_timeout_s > 180  # RQ default (180 s) is far too short for a full scan
    assert s.cookie_secure is False
    assert s.allow_private_network is False
    assert s.access_token_value() is None
    assert s.hf_token_value() is None


def test_hf_token_accepts_plain_and_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_plainplainplain")
    assert Settings().hf_token_value() == "hf_plainplainplain"
    monkeypatch.setenv("VRAMFORGE_HF_TOKEN", "hf_prefixedprefixed")
    assert Settings().hf_token_value() == "hf_prefixedprefixed"


def test_secrets_are_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    s = Settings(hf_token="hf_secretsecretsecret", access_token="tok-123456")
    assert "hf_secretsecretsecret" not in repr(s)
    assert "tok-123456" not in repr(s)
    assert s.access_token_value() == "tok-123456"


def test_empty_access_token_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("VRAMFORGE_ACCESS_TOKEN", "")
    assert Settings().access_token_value() is None


def test_local_roots_parsing() -> None:
    assert parse_local_roots("models=/sources/models; data=/sources/data") == {
        "models": Path("/sources/models"),
        "data": Path("/sources/data"),
    }
    assert parse_local_roots("") == {}
    for bad in ("models", "models=relative/path", "../x=/abs", "a=/x,a=/y"):
        with pytest.raises(ValueError):
            parse_local_roots(bad)
    with pytest.raises(ValidationError):
        Settings(local_roots="bad entry")


def test_empty_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("VRAMFORGE_MAX_UPLOAD_BYTES", "")
    monkeypatch.setenv("VRAMFORGE_RETENTION_DAYS", "")
    s = Settings()
    assert s.max_upload_bytes == 2 * 1024**3 and s.retention_days == 7


def test_blank_hf_token_is_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    for blank in ("", "   "):
        s = Settings(hf_token=blank, share_server_hf_token=True)
        assert s.hf_token_value() is None
        assert s.source_hf_token() is None
        assert s.hf_token_mode == "not_configured"


def test_server_hf_token_is_used_only_when_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_serverserverserver")
    s = Settings()
    assert s.share_server_hf_token is False  # plan §18: opt-in only
    assert s.hf_token_value() == "hf_serverserverserver"
    assert s.source_hf_token() is None
    assert s.hf_token_mode == "configured_not_shared"

    monkeypatch.setenv("VRAMFORGE_SHARE_SERVER_HF_TOKEN", "true")
    s = Settings()
    assert s.source_hf_token() == "hf_serverserverserver"
    assert s.hf_token_mode == "shared"
    assert "hf_serverserverserver" not in repr(s)

    monkeypatch.setenv("VRAMFORGE_SHARE_SERVER_HF_TOKEN", "")  # compose passes empty values
    assert Settings().share_server_hf_token is False
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("VRAMFORGE_SHARE_SERVER_HF_TOKEN", "true")
    assert Settings().source_hf_token() is None  # nothing to share
    assert Settings().hf_token_mode == "not_configured"
