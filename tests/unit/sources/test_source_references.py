"""Reference normalization table (plan.md §6.1) and SSRF / path rejections (plan.md §18)."""

from __future__ import annotations

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    DatasetSourceRef,
    ErrorCode,
    ModelSourceRef,
    SourceType,
)
from vramforge_estimator.sources.references import (
    normalize_dataset_reference,
    normalize_model_reference,
)

SHA = "2367e865d009c13ac81713a2878291d33ab28177"

MODEL_CASES = [
    # (reference, revision, repo_id, effective revision, display)
    (
        "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        None,
        "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        None,
        "hf:XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    ),
    ("  org/name  ", None, "org/name", None, "hf:org/name"),
    ("gpt2", None, "gpt2", None, "hf:gpt2"),
    ("hf:org/name", "main", "org/name", "main", "hf:org/name"),
    ("https://huggingface.co/org/name", None, "org/name", None, "hf:org/name"),
    ("https://huggingface.co/org/name/", None, "org/name", None, "hf:org/name"),
    ("https://hf.co/org/name", None, "org/name", None, "hf:org/name"),
    ("https://www.huggingface.co/org/name", None, "org/name", None, "hf:org/name"),
    ("https://huggingface.co/org/name#model-card", None, "org/name", None, "hf:org/name"),
    (f"https://huggingface.co/org/name/tree/{SHA}", None, "org/name", SHA, "hf:org/name"),
    ("https://huggingface.co/org/name/tree/main", "main", "org/name", "main", "hf:org/name"),
    (
        "https://huggingface.co/org/name/tree/refs%2Fpr%2F3",
        None,
        "org/name",
        "refs/pr/3",
        "hf:org/name",
    ),
    ("https://huggingface.co/org/name/tree/v1.0/sub/dir", None, "org/name", "v1.0", "hf:org/name"),
    (
        "https://huggingface.co/org/name/blob/main/config.json",
        None,
        "org/name",
        "main",
        "hf:org/name",
    ),
    (
        "https://huggingface.co/org/name/resolve/dev/model.safetensors",
        None,
        "org/name",
        "dev",
        "hf:org/name",
    ),
    (f"https://huggingface.co/org/name/commit/{SHA}", None, "org/name", SHA, "hf:org/name"),
    ("https://huggingface.co/gpt2", None, "gpt2", None, "hf:gpt2"),
]


@pytest.mark.parametrize(("reference", "revision", "repo_id", "effective", "display"), MODEL_CASES)
def test_model_reference_normalization(
    reference: str, revision: str | None, repo_id: str, effective: str | None, display: str
) -> None:
    norm = normalize_model_reference(ModelSourceRef(reference=reference, revision=revision))
    assert norm.source_type is SourceType.HUGGINGFACE
    assert norm.kind == "model"
    assert norm.repo_id == repo_id
    assert norm.revision == effective
    assert norm.display == display


def test_model_file_url_keeps_path_as_information_only() -> None:
    norm = normalize_model_reference(
        ModelSourceRef(reference="https://huggingface.co/org/name/blob/main/sub/config.json")
    )
    assert norm.path_in_repo == "sub/config.json"
    assert norm.notes  # the user is told the path does not narrow the analysis


DATASET_CASES = [
    # (reference, repo_id, revision, config_hint, split_hint)
    (
        "CyberNative/Code_Vulnerability_Security_DPO",
        "CyberNative/Code_Vulnerability_Security_DPO",
        None,
        None,
        None,
    ),
    ("hf-dataset:org/data", "org/data", None, None, None),
    ("https://huggingface.co/datasets/org/data", "org/data", None, None, None),
    ("https://huggingface.co/datasets/squad", "squad", None, None, None),
    (
        "https://huggingface.co/datasets/org/data/tree/refs%2Fconvert%2Fparquet",
        "org/data",
        "refs/convert/parquet",
        None,
        None,
    ),
    (
        "https://huggingface.co/datasets/org/data/blob/main/train.jsonl",
        "org/data",
        "main",
        None,
        None,
    ),
    ("https://huggingface.co/datasets/org/data/viewer", "org/data", None, None, None),
    ("https://huggingface.co/datasets/org/data/viewer/default", "org/data", None, "default", None),
    (
        "https://huggingface.co/datasets/org/data/viewer/default/train?row=0",
        "org/data",
        None,
        "default",
        "train",
    ),
    (
        "https://huggingface.co/datasets/org/data/viewer/cs-en/validation?p=3&q=foo",
        "org/data",
        None,
        "cs-en",
        "validation",
    ),
]


@pytest.mark.parametrize(("reference", "repo_id", "revision", "config", "split"), DATASET_CASES)
def test_dataset_reference_normalization(
    reference: str, repo_id: str, revision: str | None, config: str | None, split: str | None
) -> None:
    norm = normalize_dataset_reference(DatasetSourceRef(reference=reference))
    assert norm.kind == "dataset"
    assert norm.source_type is SourceType.HUGGINGFACE
    assert norm.repo_id == repo_id
    assert norm.display == f"hf-dataset:{repo_id}"
    assert norm.revision == revision
    assert norm.config_hint == config
    assert norm.split_hint == split


def test_viewer_row_is_not_a_range_restriction() -> None:
    norm = normalize_dataset_reference(
        DatasetSourceRef(reference="https://huggingface.co/datasets/o/d/viewer/default/train?row=0")
    )
    assert norm.split_hint == "train"
    assert any("row" in note for note in norm.notes)
    # nothing in the normalized reference narrows the rows to analyse
    assert not hasattr(norm, "row")


def test_url_revision_and_field_revision_must_agree() -> None:
    url = f"https://huggingface.co/org/name/tree/{SHA}"
    assert normalize_model_reference(ModelSourceRef(reference=url, revision=SHA)).revision == SHA
    with pytest.raises(EstimatorError) as exc:
        normalize_model_reference(ModelSourceRef(reference=url, revision="main"))
    assert exc.value.issue.code is ErrorCode.CONFLICTING_OPTIONS
    assert exc.value.issue.details["reason"] == "revision_conflict"


def test_viewer_hints_conflicting_with_explicit_fields_are_reported() -> None:
    url = "https://huggingface.co/datasets/o/d/viewer/default/train"
    ok = normalize_dataset_reference(
        DatasetSourceRef(reference=url, config="default", split="train")
    )
    assert (ok.config_hint, ok.split_hint) == ("default", "train")
    for kwargs, reason in (
        ({"split": "test"}, "split_conflict"),
        ({"config": "x"}, "config_conflict"),
    ):
        with pytest.raises(EstimatorError) as exc:
            normalize_dataset_reference(DatasetSourceRef(reference=url, **kwargs))
        assert exc.value.issue.code is ErrorCode.CONFLICTING_OPTIONS
        assert exc.value.issue.details["reason"] == reason


LOCAL_CASES = [
    ("local:models/qwen", "models", "qwen", "local:models/qwen"),
    ("local:models/org/qwen/", "models", "org/qwen", "local:models/org/qwen"),
    ("local:models", "models", "", "local:models/"),
    ("local:data/train.jsonl", "data", "train.jsonl", "local:data/train.jsonl"),
]


@pytest.mark.parametrize(("reference", "root", "relative", "display"), LOCAL_CASES)
def test_local_reference_normalization(
    reference: str, root: str, relative: str, display: str
) -> None:
    norm = normalize_model_reference(ModelSourceRef(reference=reference))
    assert norm.source_type is SourceType.LOCAL
    assert (norm.local_root, norm.local_relative, norm.display) == (root, relative, display)


def test_local_and_upload_without_prefix_use_source_type() -> None:
    local = normalize_dataset_reference(
        DatasetSourceRef(source_type=SourceType.LOCAL, reference="data/x.parquet")
    )
    assert (local.local_root, local.local_relative) == ("data", "x.parquet")
    upload = normalize_dataset_reference(
        DatasetSourceRef(
            source_type=SourceType.UPLOAD, reference="0f8c2f4e-7d0b-4f8e-9a8b-1c2d3e4f5a6b"
        )
    )
    assert upload.upload_id == "0f8c2f4e-7d0b-4f8e-9a8b-1c2d3e4f5a6b"
    assert upload.display == "upload:0f8c2f4e-7d0b-4f8e-9a8b-1c2d3e4f5a6b"


def test_upload_prefix() -> None:
    norm = normalize_dataset_reference(DatasetSourceRef(reference="upload:Abc_123-xyz789"))
    assert norm.source_type is SourceType.UPLOAD
    assert norm.upload_id == "Abc_123-xyz789"


def test_local_revision_only_accepts_content_digest() -> None:
    digest = "sha256:" + "a" * 64
    norm = normalize_model_reference(ModelSourceRef(reference="local:m/x", revision=digest))
    assert norm.revision == digest
    with pytest.raises(EstimatorError) as exc:
        normalize_model_reference(ModelSourceRef(reference="local:m/x", revision="main"))
    assert exc.value.issue.code is ErrorCode.CONFLICTING_OPTIONS


REJECTED = [
    # SSRF: other schemes, hosts, ports, credentials, plain http (plan §18)
    ("http://huggingface.co/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://evil.example.com/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co.evil.com/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://evil.com/huggingface.co/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co@evil.com/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://user:hf_secret@huggingface.co/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co:8443/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://169.254.169.254/latest/meta-data", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://localhost/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://127.0.0.1/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://[::1]/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("file:///etc/passwd", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("ftp://huggingface.co/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("s3://bucket/model", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("javascript:alert(1)", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    # unsupported HF pages
    ("https://huggingface.co/", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/models", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/spaces/org/app", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/api/models/org/name", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/org/name/discussions/1", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/org/name/blob/main", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/org/name/tree", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/org/name/tree/main/%2e%2e/x", ErrorCode.SOURCE_URL_NOT_ALLOWED),
    ("https://huggingface.co/datasets/org/data", ErrorCode.SOURCE_URL_NOT_ALLOWED),  # kind
    ("hf-dataset:org/data", ErrorCode.SOURCE_URL_NOT_ALLOWED),  # kind mismatch
    # host paths and traversal
    ("/etc/passwd", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("~/models/x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("./models/x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("../models/x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("C:\\Users\\me\\model", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("C:/Users/me/model", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("\\\\server\\share\\model", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:models/../etc", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:models/./x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:models//x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:/etc/passwd", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:../models/x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("local:models/C:/x", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("upload:../../etc/passwd", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("upload:abc", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    ("upload:abc.def.ghi.jkl", ErrorCode.LOCAL_PATH_NOT_ALLOWED),
    # malformed ids
    ("org/name/extra", ErrorCode.INVALID_REQUEST),
    ("org//name", ErrorCode.INVALID_REQUEST),
    ("org/na--me", ErrorCode.INVALID_REQUEST),
    ("org/name.git", ErrorCode.INVALID_REQUEST),
    ("org name/x", ErrorCode.INVALID_REQUEST),
    ("org/na\nme", ErrorCode.INVALID_REQUEST),
    ("org/\x00name", ErrorCode.INVALID_REQUEST),
]


@pytest.mark.parametrize(("reference", "code"), REJECTED)
def test_model_reference_rejections(reference: str, code: ErrorCode) -> None:
    with pytest.raises(EstimatorError) as exc:
        normalize_model_reference(ModelSourceRef(reference=reference))
    issue = exc.value.issue
    assert issue.code is code
    assert issue.stage is not None and issue.stage.value == "resolving"
    # the raw reference (which may contain secrets) is never echoed back
    assert "hf_secret" not in issue.user_message
    assert "hf_secret" not in str(issue.details)


def test_dataset_rejects_model_url() -> None:
    with pytest.raises(EstimatorError) as exc:
        normalize_dataset_reference(DatasetSourceRef(reference="https://huggingface.co/org/name"))
    assert exc.value.issue.code is ErrorCode.SOURCE_URL_NOT_ALLOWED
    assert exc.value.issue.details["reason"] == "kind_mismatch"


def test_invalid_revision_rejected() -> None:
    for revision in ("../main", "main;rm", "a b", "/main", "main/", "x..y"):
        with pytest.raises(EstimatorError) as exc:
            normalize_model_reference(ModelSourceRef(reference="org/name", revision=revision))
        assert exc.value.issue.code is ErrorCode.INVALID_REQUEST


def test_explicit_source_type_conflicting_with_prefix() -> None:
    with pytest.raises(EstimatorError) as exc:
        normalize_model_reference(
            ModelSourceRef(source_type=SourceType.HUGGINGFACE, reference="local:models/x")
        )
    assert exc.value.issue.code is ErrorCode.CONFLICTING_OPTIONS
    with pytest.raises(EstimatorError):
        normalize_model_reference(
            ModelSourceRef(source_type=SourceType.LOCAL, reference="https://huggingface.co/o/n")
        )
    # the default (not explicitly set) source_type yields to the prefix
    assert normalize_model_reference(ModelSourceRef(reference="local:m/x")).source_type is (
        SourceType.LOCAL
    )


def test_configured_endpoint_host_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from vramforge_estimator.sources import references

    monkeypatch.setattr(references, "configured_hf_endpoint", lambda: "https://hf-mirror.example")
    norm = normalize_model_reference(ModelSourceRef(reference="https://hf-mirror.example/o/n"))
    assert norm.repo_id == "o/n"
