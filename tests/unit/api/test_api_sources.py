"""Uploads (size/type/path tricks), local roots, backend profiles and metadata inspection."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from api_testkit import example_request
from fastapi.testclient import TestClient

from vramforge_api.security import owner_key_for
from vramforge_api.settings import Settings
from vramforge_estimator import compatibility, inspection, sources
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import (
    BackendProfilesResponse,
    DatasetColumn,
    DatasetInspection,
    ErrorCode,
    SourceManifest,
    SourceType,
)
from vramforge_estimator.sources import ResolvedSource

JSONL = b'{"prompt": "a", "chosen": "b", "rejected": "c"}\n' * 4


def _owner_key(client: TestClient) -> str:
    client.get("/api/v1/local-roots")
    return owner_key_for(client.cookies["vf_owner"])


def _upload(client: TestClient, name: str, data: bytes, field: str = "file"):
    return client.post("/api/v1/uploads", files={field: (name, data, "application/octet-stream")})


def test_upload_streams_to_owner_directory(client: TestClient, settings: Settings) -> None:
    resp = _upload(client, "train.jsonl", JSONL)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["reference"] == f"upload:{body['upload_id']}"
    assert body["filename"] == "train.jsonl"
    assert body["size_bytes"] == len(JSONL)
    assert body["sha256"] == hashlib.sha256(JSONL).hexdigest()
    stored = settings.uploads_dir / _owner_key(client) / body["upload_id"] / "train.jsonl"
    assert stored.read_bytes() == JSONL
    assert not any(p.name.startswith(".upload-") for p in stored.parent.iterdir())

    # the owner can analyze it; another owner cannot even see it
    ok = client.post(
        "/api/v1/analyses",
        json=example_request(
            **{"dataset.source_type": "upload", "dataset.reference": body["reference"]}
        ),
    )
    assert ok.status_code == 202


def test_upload_is_invisible_to_other_owners(
    settings: Settings, client_factory: Callable[..., TestClient]
) -> None:
    alice, bob = client_factory(settings), client_factory(settings)
    ref = _upload(alice, "train.jsonl", JSONL).json()["reference"]
    resp = bob.post(
        "/api/v1/analyses",
        json=example_request(**{"dataset.source_type": "upload", "dataset.reference": ref}),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "SOURCE_NOT_FOUND"


@pytest.mark.parametrize(
    "name", ["../../../evil.jsonl", "..\\..\\evil.jsonl", "/etc/evil.jsonl", "..evil.jsonl"]
)
def test_upload_path_tricks_stay_inside_the_upload_dir(
    client: TestClient, settings: Settings, name: str
) -> None:
    resp = _upload(client, name, JSONL)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "/" not in body["filename"] and "\\" not in body["filename"]
    assert not body["filename"].startswith(".")
    upload_dir = settings.uploads_dir / _owner_key(client) / body["upload_id"]
    assert [p.name for p in upload_dir.iterdir()] == [body["filename"]]
    assert not (settings.data_dir / "evil.jsonl").exists()


def test_upload_size_cap_is_enforced_while_streaming(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    settings = settings_factory(max_upload_bytes=1024)
    client = client_factory(settings)
    resp = _upload(client, "big.jsonl", b'{"a": 1}\n' * 400)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    owner_dir = settings.uploads_dir / _owner_key(client)
    assert not owner_dir.exists() or not any(owner_dir.rglob("*"))


def test_declared_content_length_over_cap_is_rejected_early(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings_factory(max_upload_bytes=1024))
    resp = client.post(
        "/api/v1/uploads",
        content=b"x" * 10,
        headers={"Content-Length": str(10**9), "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert resp.status_code == 413


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("tool.exe", b"MZ\x90\x00"),
        ("data.json.gz", b"\x1f\x8b\x08\x00"),
        ("data.parquet", JSONL),  # extension says parquet, content is JSON
        ("data.json", b"PAR1\x00\x00PAR1"),  # extension says JSON, content is binary
        ("data.csv", b"a,b\x00\x01"),
        ("empty.jsonl", b""),
    ],
)
def test_upload_type_allowlist_and_sniffing(client: TestClient, name: str, data: bytes) -> None:
    resp = _upload(client, name, data)
    assert resp.status_code == 415, resp.text
    assert resp.json()["error"]["code"] == "UPLOAD_TYPE_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("d.json", b'\xef\xbb\xbf  [{"a": 1}]'),
        ("d.csv", "prompt,chosen\n안녕,hi\n".encode()),
        ("d.parquet", b"PAR1" + b"\x00" * 16 + b"PAR1"),
        ("d.arrow", b"\xff\xff\xff\xff" + b"\x00" * 8),
    ],
)
def test_upload_accepts_supported_formats(client: TestClient, name: str, data: bytes) -> None:
    assert _upload(client, name, data).status_code == 201


def test_upload_requires_exactly_one_file_field(client: TestClient) -> None:
    assert _upload(client, "a.jsonl", JSONL, field="other").status_code == 422
    two = client.post(
        "/api/v1/uploads",
        files=[("file", ("a.jsonl", JSONL)), ("file", ("b.jsonl", JSONL))],
    )
    assert two.status_code == 422
    plain = client.post("/api/v1/uploads", json={"file": "x"})
    assert plain.status_code == 422


def test_local_roots_lists_mounted_roots_only(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    tmp_path: Path,
) -> None:
    (tmp_path / "mounted").mkdir()
    settings = settings_factory(
        local_roots=f"local={tmp_path / 'mounted'},missing={tmp_path / 'nope'}"
    )
    roots = client_factory(settings).get("/api/v1/local-roots").json()["roots"]
    assert [r["name"] for r in roots] == ["local"]
    assert roots[0]["reference_prefix"] == "local:local/"
    assert roots[0]["read_only"] is True
    assert str(tmp_path) not in str(roots)  # never expose server paths


def test_backend_profiles_never_claim_a_gpu_worker(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        compatibility,
        "backend_profiles",
        lambda: BackendProfilesResponse(estimator_version="0.1.0", gpu_worker_connected=True),
    )
    body = client.get("/api/v1/backend-profiles").json()
    assert body["gpu_worker_connected"] is False


def test_unimplemented_backend_profiles_is_reported_honestly(
    client: TestClient, monkeypatch
) -> None:
    def stub() -> BackendProfilesResponse:
        raise NotImplementedError

    monkeypatch.setattr(compatibility, "backend_profiles", stub)
    resp = client.get("/api/v1/backend-profiles")
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"


def _manifest(kind: str) -> SourceManifest:
    return SourceManifest(
        kind=kind,  # type: ignore[arg-type]
        source_type=SourceType.HUGGINGFACE,
        reference=f"hf:org/{kind}",
        repo_id=f"org/{kind}",
        resolved_revision="a" * 40,
        fingerprint=f"src_{kind}",
    )


def test_inspect_returns_partial_info_and_issues(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        sources,
        "resolve_model",
        lambda ref, access: ResolvedSource(
            kind="model", manifest=_manifest("model"), repo_id="org/model", revision="a" * 40
        ),
    )

    def no_metadata(source, access):
        raise EstimatorError(
            make_issue(
                ErrorCode.MODEL_METADATA_UNAVAILABLE, "safetensors header를 읽을 수 없습니다."
            )
        )

    def not_yet(source, access):
        raise NotImplementedError

    monkeypatch.setattr(inspection, "inspect_model", no_metadata)
    monkeypatch.setattr(inspection, "load_tokenizer", not_yet)
    monkeypatch.setattr(
        sources,
        "resolve_dataset",
        lambda ref, access: ResolvedSource(
            kind="dataset", manifest=_manifest("dataset"), repo_id="org/dataset"
        ),
    )
    monkeypatch.setattr(
        inspection,
        "inspect_dataset",
        lambda source, ref, access, objective=None: DatasetInspection(
            columns=[DatasetColumn(name="question", dtype="string", kind="string")],
            selected_split="train",
        ),
    )
    req = example_request()
    resp = client.post(
        "/api/v1/sources/inspect",
        json={"model": req["model"], "dataset": req["dataset"], "objective": "grpo"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"]["manifest"]["repo_id"] == "org/model"
    assert body["model"]["summary"] is None
    codes = {i["code"] for i in body["model"]["issues"]}
    assert codes == {"MODEL_METADATA_UNAVAILABLE", "INTERNAL_ERROR"}
    assert body["dataset"]["manifest"]["repo_id"] == "org/dataset"
    assert body["dataset"]["columns"][0]["name"] == "question"


def test_inspect_times_out_with_partial_info(
    settings_factory: Callable[..., Settings],
    client_factory: Callable[..., TestClient],
    monkeypatch,
) -> None:
    client = client_factory(settings_factory(inspect_timeout_s=0.3))
    monkeypatch.setattr(
        sources,
        "resolve_dataset",
        lambda ref, access: ResolvedSource(kind="dataset", manifest=_manifest("dataset")),
    )

    def slow(source, ref, access, objective=None):
        time.sleep(1.5)
        return DatasetInspection()

    monkeypatch.setattr(inspection, "inspect_dataset", slow)
    started = time.monotonic()
    resp = client.post("/api/v1/sources/inspect", json={"dataset": example_request()["dataset"]})
    assert time.monotonic() - started < 1.4
    body = resp.json()
    assert body["model"] is None
    assert body["dataset"]["manifest"]["reference"] == "hf:org/dataset"
    assert body["dataset"]["issues"][-1]["code"] == "JOB_TIMEOUT"


def test_inspect_rejects_foreign_upload_without_calling_resolvers(
    client: TestClient, monkeypatch
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("resolver called for a foreign upload")

    monkeypatch.setattr(sources, "resolve_dataset", must_not_run)
    dataset = example_request(
        **{"dataset.source_type": "upload", "dataset.reference": "upload:" + "b" * 32}
    )["dataset"]
    body = client.post("/api/v1/sources/inspect", json={"dataset": dataset}).json()
    assert body["dataset"]["issues"][0]["code"] == "SOURCE_NOT_FOUND"
