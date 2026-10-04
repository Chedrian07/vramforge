"""Hub client: endpoint policy, token handling, metadata conversion and error mapping (offline)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub import errors as hf_errors
from huggingface_hub.hf_api import BlobLfsInfo, RepoSibling

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode
from vramforge_estimator.sources import SourceAccess, hub
from vramforge_estimator.sources import references as refs

SHA = "2367e865d009c13ac81713a2878291d33ab28177"
TOKEN = "hf_" + "x" * 30


def _response(status: int) -> httpx.Response:
    request = httpx.Request("GET", f"https://huggingface.co/api/models/o/n?token={TOKEN}")
    return httpx.Response(status, request=request)


def _http(cls: type[hf_errors.HfHubHTTPError], status: int) -> hf_errors.HfHubHTTPError:
    return cls(f"failed with {TOKEN}", response=_response(status))


MAPPING = [
    (
        _http(hf_errors.RepositoryNotFoundError, 401),
        ErrorCode.SOURCE_NOT_FOUND,
        False,
        "repository_not_found",
    ),
    (
        _http(hf_errors.RevisionNotFoundError, 404),
        ErrorCode.SOURCE_NOT_FOUND,
        False,
        "revision_not_found",
    ),
    (_http(hf_errors.GatedRepoError, 403), ErrorCode.SOURCE_ACCESS_DENIED, False, "gated"),
    (_http(hf_errors.DisabledRepoError, 403), ErrorCode.SOURCE_ACCESS_DENIED, False, "disabled"),
    (_http(hf_errors.HfHubHTTPError, 401), ErrorCode.SOURCE_ACCESS_DENIED, False, "forbidden"),
    (_http(hf_errors.HfHubHTTPError, 403), ErrorCode.SOURCE_ACCESS_DENIED, False, "forbidden"),
    (_http(hf_errors.HfHubHTTPError, 404), ErrorCode.SOURCE_NOT_FOUND, False, "not_found"),
    (
        _http(hf_errors.HfHubHTTPError, 429),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        True,
        "server_unavailable",
    ),
    (
        _http(hf_errors.HfHubHTTPError, 503),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        True,
        "server_unavailable",
    ),
    (
        _http(hf_errors.RemoteEntryNotFoundError, 404),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        False,
        "file_not_found",
    ),
    (
        hf_errors.LocalEntryNotFoundError(f"offline {TOKEN}"),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        True,
        "network",
    ),
    (
        httpx.ConnectTimeout(f"timeout {TOKEN}"),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        True,
        "network",
    ),
    (httpx.ConnectError("refused"), ErrorCode.MODEL_METADATA_UNAVAILABLE, True, "network"),
    (
        hf_errors.SafetensorsParsingError("bad"),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        False,
        "malformed_safetensors_header",
    ),
    (
        hf_errors.OfflineModeIsEnabled("offline"),
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        False,
        "offline_mode",
    ),
]


@pytest.mark.parametrize(("exc", "code", "retryable", "reason"), MAPPING)
def test_hub_error_mapping(
    exc: BaseException, code: ErrorCode, retryable: bool, reason: str
) -> None:
    assert hub.is_hub_exception(exc)
    issue = hub.hub_error(exc, kind="model").issue
    assert issue.code is code
    assert issue.retryable is retryable
    assert issue.details["reason"] == reason
    assert issue.details["error_type"] == type(exc).__name__
    # exception messages (which may embed URLs or tokens) are never copied into the issue
    dumped = issue.model_dump_json()
    assert TOKEN not in dumped
    assert "huggingface.co/api" not in dumped


def test_gated_error_wins_over_repository_not_found() -> None:
    # GatedRepoError subclasses RepositoryNotFoundError: the order of checks matters
    issue = hub.hub_error(_http(hf_errors.GatedRepoError, 401), kind="dataset").issue
    assert issue.code is ErrorCode.SOURCE_ACCESS_DENIED
    assert issue.affected_component == "dataset"


class _StubApi:
    def __init__(self, info: Any) -> None:
        self.info = info
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def model_info(self, repo_id: str, **kwargs: Any) -> Any:
        self.calls.append((repo_id, kwargs))
        return self.info

    dataset_info = model_info

    def parse_safetensors_file_metadata(self, repo_id: str, filename: str, **kwargs: Any) -> Any:
        self.calls.append((filename, kwargs))
        tensor = SimpleNamespace(dtype="BF16", shape=[2, 3], data_offsets=(0, 12))
        return SimpleNamespace(tensors={"w": tensor}, metadata={"format": "pt"})


def _info(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "id": "org/name",
        "sha": SHA,
        "private": False,
        "gated": "manual",
        "last_modified": datetime(2026, 9, 22, 3, 52, 45, tzinfo=UTC),
        "siblings": [
            RepoSibling(rfilename="z.json", size=10, blob_id="b" * 40),
            RepoSibling(
                rfilename="model.safetensors",
                size=100,
                blob_id="c" * 40,
                lfs=BlobLfsInfo(size=100, sha256="d" * 64, pointer_size=133),
            ),
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_client_converts_repo_info_and_never_uses_implicit_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_" + "s" * 30)  # a server-side token must not be used
    client = hub.HfHubClient(SourceAccess(hf_token=None, http_timeout_s=7.0))
    stub = _StubApi(_info())
    client._api = stub  # type: ignore[assignment]
    info = client.repo_info("model", "org/name", None)
    assert info.repo_id == "org/name" and info.sha == SHA
    assert [f.path for f in info.files] == ["model.safetensors", "z.json"]
    assert info.files[0].sha256 == "d" * 64 and info.files[0].blob_id == "c" * 40
    assert info.files[1].sha256 is None
    assert info.gated is True and info.private is False
    assert info.last_modified == "2026-09-22T03:52:45+00:00"
    _, kwargs = stub.calls[0]
    assert kwargs["token"] is False
    assert kwargs["files_metadata"] is True
    assert kwargs["timeout"] == 7.0

    header = client.read_safetensors_header("org/name", SHA, "model.safetensors")
    assert header == {
        "w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "__metadata__": {"format": "pt"},
    }
    assert stub.calls[1][1]["revision"] == SHA


def test_client_passes_user_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = hub.HfHubClient(SourceAccess(hf_token=TOKEN))
    stub = _StubApi(_info(gated=False))
    client._api = stub  # type: ignore[assignment]
    info = client.repo_info("dataset", "org/name", "main")
    assert info.gated is False
    assert stub.calls[0][1]["token"] == TOKEN
    assert stub.calls[0][1]["revision"] == "main"


def test_client_rejects_missing_commit_sha() -> None:
    client = hub.HfHubClient(SourceAccess())
    client._api = _StubApi(_info(sha=None))  # type: ignore[assignment]
    with pytest.raises(EstimatorError) as exc:
        client.repo_info("model", "org/name", None)
    assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://huggingface.co",
        "https://10.0.0.5",
        "https://127.0.0.1:8080",
        "https://169.254.169.254",
        "https://localhost",
        "https://[::1]",
    ],
)
def test_private_or_plain_http_endpoint_is_refused(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refs, "configured_hf_endpoint", lambda: endpoint)
    monkeypatch.setattr(hub, "configured_hf_endpoint", lambda: endpoint)
    with pytest.raises(EstimatorError) as exc:
        hub.HfHubClient(SourceAccess())
    assert exc.value.issue.code is ErrorCode.SOURCE_URL_NOT_ALLOWED
    # an operator can explicitly allow a private mirror
    hub.HfHubClient(SourceAccess(allow_private_network=True))


def test_public_mirror_endpoint_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "configured_hf_endpoint", lambda: "https://hf-mirror.example.com")
    client = hub.HfHubClient(SourceAccess())
    assert client._endpoint == "https://hf-mirror.example.com"


def test_hub_cache_dir_follows_hf_home(tmp_path: Any) -> None:
    assert hub.hub_cache_dir(SourceAccess(hf_home=tmp_path)) == tmp_path / "hub"
    assert hub.hub_cache_dir(SourceAccess()) is None
