"""HF resolution through a fake Hub client: pinning, manifest fields, gating and errors."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from huggingface_hub import errors as hf_errors

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.keys import source_key
from vramforge_estimator.schemas import DatasetSourceRef, ErrorCode, ModelSourceRef, SourceType
from vramforge_estimator.sources import SourceAccess, hub, resolve_dataset, resolve_model

SHA = "2367e865d009c13ac81713a2878291d33ab28177"
DATA_SHA = "81aeacf06cf43b16d7278a3a01f019a496a53c51"
TOKEN = "hf_" + "t" * 34


@pytest.fixture
def fake(fake_hub: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    model = fake_hub.FakeRepo(
        repo_id="org/model",
        sha=SHA,
        files={"config.json": b'{"model_type": "llama"}', "tokenizer.json": b"{}"},
        shards={"model.safetensors": ({}, 1234, "e" * 64)},
        aliases=("model",),
    )
    data = fake_hub.FakeRepo(
        repo_id="org/data",
        sha=DATA_SHA,
        files={"README.md": b"# card", "train.jsonl": b'{"a": 1}\n'},
    )
    client = fake_hub.FakeHub(
        tmp_path, {("model", "org/model"): model, ("dataset", "org/data"): data}
    )
    monkeypatch.setattr(hub, "get_hub_client", lambda access: client)
    return client


def test_model_is_pinned_to_commit(fake: Any) -> None:
    source = resolve_model(ModelSourceRef(reference="org/model"), SourceAccess())
    manifest = source.manifest
    assert (source.kind, source.repo_id, source.revision, source.local_path) == (
        "model",
        "org/model",
        SHA,
        None,
    )
    assert manifest.source_type is SourceType.HUGGINGFACE
    assert manifest.reference == "hf:org/model"
    assert manifest.repo_id == "org/model"
    assert manifest.requested_revision is None
    assert manifest.resolved_revision == SHA
    assert manifest.private is False and manifest.gated is False
    assert manifest.last_modified == "2026-09-22T03:52:45+00:00"
    assert manifest.fingerprint == source_key("huggingface", f"model:org/model@{SHA}")
    assert [f.path for f in manifest.files] == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    shard = manifest.files[1]
    assert (shard.size, shard.sha256) == (1234, "e" * 64)
    assert any(SHA[:12] in note for note in manifest.notes)


def test_branch_revision_and_url_are_recorded(fake: Any) -> None:
    url = "https://huggingface.co/org/model/tree/main"
    manifest = resolve_model(ModelSourceRef(reference=url), SourceAccess()).manifest
    assert manifest.requested_revision == "main"
    assert manifest.resolved_revision == SHA


def test_canonical_repo_id_after_redirect(fake: Any) -> None:
    manifest = resolve_model(ModelSourceRef(reference="model"), SourceAccess()).manifest
    assert manifest.repo_id == "org/model"
    assert manifest.reference == "hf:org/model"
    assert any("org/model" in note for note in manifest.notes)


def test_dataset_reference_and_viewer_notes(fake: Any) -> None:
    ref = DatasetSourceRef(
        reference="https://huggingface.co/datasets/org/data/viewer/default/train?row=0",
        revision=DATA_SHA,
    )
    source = resolve_dataset(ref, SourceAccess())
    manifest = source.manifest
    assert manifest.kind == "dataset"
    assert manifest.reference == "hf-dataset:org/data"
    assert manifest.resolved_revision == DATA_SHA
    assert manifest.requested_revision == DATA_SHA
    assert any("row" in note for note in manifest.notes)
    assert any("'train'" in note for note in manifest.notes)
    assert source.revision == DATA_SHA


def test_gated_repo_without_token_is_denied_early(fake: Any) -> None:
    fake.repos[("model", "org/model")].gated = True
    with pytest.raises(EstimatorError) as exc:
        resolve_model(ModelSourceRef(reference="org/model"), SourceAccess())
    assert exc.value.issue.code is ErrorCode.SOURCE_ACCESS_DENIED
    assert exc.value.issue.details["reason"] == "gated_without_token"


def test_gated_repo_with_token_probes_a_non_readme_file(fake: Any) -> None:
    fake.repos[("model", "org/model")].gated = True
    manifest = resolve_model(
        ModelSourceRef(reference="org/model"), SourceAccess(hf_token=TOKEN)
    ).manifest
    assert manifest.gated is True
    assert ("check_file_access", "config.json") in fake.calls

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    fake.errors["check_file_access"] = hf_errors.GatedRepoError(
        f"no access {TOKEN}", response=response
    )
    with pytest.raises(EstimatorError) as exc:
        resolve_model(ModelSourceRef(reference="org/model"), SourceAccess(hf_token=TOKEN))
    assert exc.value.issue.code is ErrorCode.SOURCE_ACCESS_DENIED
    assert TOKEN not in exc.value.issue.model_dump_json()


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (
            hf_errors.RepositoryNotFoundError(
                "nope", response=httpx.Response(401, request=httpx.Request("GET", "https://x"))
            ),
            ErrorCode.SOURCE_NOT_FOUND,
            False,
        ),
        (
            hf_errors.RevisionNotFoundError(
                "nope", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))
            ),
            ErrorCode.SOURCE_NOT_FOUND,
            False,
        ),
        (httpx.ReadTimeout("slow"), ErrorCode.MODEL_METADATA_UNAVAILABLE, True),
    ],
)
def test_resolution_errors(
    fake: Any, error: BaseException, code: ErrorCode, retryable: bool
) -> None:
    fake.errors["repo_info"] = error
    with pytest.raises(EstimatorError) as exc:
        resolve_model(ModelSourceRef(reference="org/model", revision="main"), SourceAccess())
    assert exc.value.issue.code is code
    assert exc.value.issue.retryable is retryable
    assert exc.value.issue.stage is not None and exc.value.issue.stage.value == "resolving"


def test_normalization_errors_happen_before_any_request(fake: Any) -> None:
    with pytest.raises(EstimatorError) as exc:
        resolve_model(ModelSourceRef(reference="https://evil.example/org/model"), SourceAccess())
    assert exc.value.issue.code is ErrorCode.SOURCE_URL_NOT_ALLOWED
    assert fake.calls == []
