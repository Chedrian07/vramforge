"""Local roots and uploads: confinement (plan.md §18) and content identity (plan.md §16.2)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    DatasetSourceRef,
    ErrorCode,
    ModelSourceRef,
    SourceType,
)
from vramforge_estimator.sources import SourceAccess, resolve_dataset, resolve_model

UPLOAD_ID = "3f2b8c1e-5d4a-4b7c-9e1f-0a1b2c3d4e5f"


@pytest.fixture
def roots(tmp_path: Path, st_writer: ModuleType) -> dict[str, Path]:
    models = tmp_path / "roots" / "models"
    data = tmp_path / "roots" / "data"
    uploads = tmp_path / "uploads"
    outside = tmp_path / "outside"
    for path in (models / "tiny", data / "dpo", uploads / UPLOAD_ID, outside):
        path.mkdir(parents=True)

    tiny = models / "tiny"
    (tiny / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (tiny / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    st_writer.write_safetensors(
        tiny / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", [16, 8]), "lm_head.weight": ("BF16", [16, 8])},
    )
    (tiny / "training_args.bin").write_bytes(b"\x80\x04 pickle bytes never read")

    (data / "dpo" / "train.jsonl").write_text('{"q": "a"}\n{"q": "b"}\n', encoding="utf-8")
    (data / "dpo" / "sub").mkdir()
    (data / "dpo" / "sub" / "test.jsonl").write_text('{"q": "c"}\n', encoding="utf-8")
    (data / "single.parquet").write_bytes(b"PAR1....PAR1")
    (uploads / UPLOAD_ID / "upload.jsonl").write_text('{"q": "u"}\n', encoding="utf-8")
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    return {"models": models, "data": data, "uploads": uploads, "outside": outside, "tmp": tmp_path}


@pytest.fixture
def access(roots: dict[str, Path]) -> SourceAccess:
    return SourceAccess(
        local_roots={"models": roots["models"], "data": roots["data"]},
        uploads_dir=roots["uploads"],
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_host_paths(blob: str, roots: dict[str, Path]) -> None:
    assert str(roots["tmp"]) not in blob
    assert str(roots["tmp"].resolve()) not in blob


def test_local_model_directory_manifest(access: SourceAccess, roots: dict[str, Path]) -> None:
    source = resolve_model(ModelSourceRef(reference="local:models/tiny"), access)
    manifest = source.manifest
    assert source.kind == "model"
    assert source.local_path == (roots["models"] / "tiny").resolve()
    assert source.repo_id is None and source.revision is None
    assert manifest.source_type is SourceType.LOCAL
    assert manifest.reference == "local:models/tiny"
    assert manifest.resolved_revision.startswith("sha256:")
    assert manifest.fingerprint.startswith("src_")
    files = {f.path: f for f in manifest.files}
    assert list(files) == sorted(files)
    tiny = roots["models"] / "tiny"
    assert files["config.json"].sha256 == _sha(tiny / "config.json")
    assert files["tokenizer.json"].sha256 == _sha(tiny / "tokenizer.json")
    # weights are identified by header + size only, pickles by size only
    assert files["model.safetensors"].sha256 is None
    assert files["model.safetensors"].size == (tiny / "model.safetensors").stat().st_size
    assert files["training_args.bin"].sha256 is None
    assert any("header" in note for note in manifest.notes)
    _no_host_paths(manifest.model_dump_json(), roots)


def test_model_identity_is_deterministic_and_ignores_mtime(
    access: SourceAccess, roots: dict[str, Path], st_writer: ModuleType
) -> None:
    ref = ModelSourceRef(reference="local:models/tiny")
    first = resolve_model(ref, access).manifest
    tiny = roots["models"] / "tiny"
    for path in tiny.iterdir():
        os.utime(path, (1, 1))
    second = resolve_model(ref, access).manifest
    assert (first.resolved_revision, first.fingerprint) == (
        second.resolved_revision,
        second.fingerprint,
    )
    assert first.files == second.files

    # changing a tensor shape changes the header and therefore the identity
    st_writer.write_safetensors(
        tiny / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", [16, 8]), "lm_head.weight": ("BF16", [32, 8])},
    )
    third = resolve_model(ref, access).manifest
    assert third.resolved_revision != first.resolved_revision

    (tiny / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    fourth = resolve_model(ref, access).manifest
    assert fourth.resolved_revision != third.resolved_revision


def test_dataset_files_are_fully_hashed(access: SourceAccess, roots: dict[str, Path]) -> None:
    ref = DatasetSourceRef(reference="local:data/dpo")
    manifest = resolve_dataset(ref, access).manifest
    assert [f.path for f in manifest.files] == ["sub/test.jsonl", "train.jsonl"]
    train = roots["data"] / "dpo" / "train.jsonl"
    assert manifest.files[1].sha256 == _sha(train)
    before = manifest.resolved_revision

    # same size, different content: identity must change (no size/mtime shortcut)
    train.write_text('{"q": "A"}\n{"q": "b"}\n', encoding="utf-8")
    os.utime(train, (1, 1))
    after = resolve_dataset(ref, access).manifest
    assert after.resolved_revision != before

    # renaming a file changes identity too
    (roots["data"] / "dpo" / "sub" / "test.jsonl").rename(roots["data"] / "dpo" / "sub" / "t.jsonl")
    assert resolve_dataset(ref, access).manifest.resolved_revision != after.resolved_revision


def test_single_dataset_file(access: SourceAccess, roots: dict[str, Path]) -> None:
    source = resolve_dataset(DatasetSourceRef(reference="local:data/single.parquet"), access)
    assert source.local_path == (roots["data"] / "single.parquet").resolve()
    assert [f.path for f in source.manifest.files] == ["single.parquet"]


def test_identity_is_independent_of_root_location(
    access: SourceAccess, roots: dict[str, Path], tmp_path: Path
) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("train.jsonl",):
        (copy / name).write_bytes((roots["data"] / "dpo" / name).read_bytes())
    (copy / "sub").mkdir()
    (copy / "sub" / "test.jsonl").write_bytes(
        (roots["data"] / "dpo" / "sub" / "test.jsonl").read_bytes()
    )
    other = SourceAccess(local_roots={"other": tmp_path})
    a = resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access).manifest
    b = resolve_dataset(DatasetSourceRef(reference="local:other/copy"), other).manifest
    assert a.resolved_revision == b.resolved_revision
    assert a.reference != b.reference


def test_expected_content_digest(access: SourceAccess) -> None:
    pinned = resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access).manifest
    again = resolve_dataset(
        DatasetSourceRef(reference="local:data/dpo", revision=pinned.resolved_revision), access
    )
    assert again.manifest.requested_revision == pinned.resolved_revision
    with pytest.raises(EstimatorError) as exc:
        resolve_dataset(
            DatasetSourceRef(reference="local:data/dpo", revision="sha256:" + "0" * 64), access
        )
    assert exc.value.issue.code is ErrorCode.SOURCE_REVISION_CHANGED


def test_upload_reference(access: SourceAccess) -> None:
    source = resolve_dataset(DatasetSourceRef(reference=f"upload:{UPLOAD_ID}"), access)
    assert source.manifest.source_type is SourceType.UPLOAD
    assert source.manifest.reference == f"upload:{UPLOAD_ID}"
    assert [f.path for f in source.manifest.files] == ["upload.jsonl"]


def _expect(code: ErrorCode, fn, roots: dict[str, Path], reason: str | None = None) -> None:
    with pytest.raises(EstimatorError) as exc:
        fn()
    issue = exc.value.issue
    assert issue.code is code, issue
    if reason is not None:
        assert issue.details.get("reason") == reason
    _no_host_paths(issue.model_dump_json(), roots)


def test_unknown_root_and_missing_paths(access: SourceAccess, roots: dict[str, Path]) -> None:
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_model(ModelSourceRef(reference="local:nope/x"), access),
        roots,
        "unknown_root",
    )
    _expect(
        ErrorCode.SOURCE_NOT_FOUND,
        lambda: resolve_model(ModelSourceRef(reference="local:models/missing"), access),
        roots,
        "path_missing",
    )
    _expect(
        ErrorCode.SOURCE_NOT_FOUND,
        lambda: resolve_dataset(DatasetSourceRef(reference="upload:does-not-exist-123"), access),
        roots,
        "upload_missing",
    )
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(
            DatasetSourceRef(reference=f"upload:{UPLOAD_ID}"), SourceAccess(uploads_dir=None)
        ),
        roots,
        "uploads_disabled",
    )


def test_model_path_must_be_a_directory(access: SourceAccess, roots: dict[str, Path]) -> None:
    _expect(
        ErrorCode.INVALID_REQUEST,
        lambda: resolve_model(ModelSourceRef(reference="local:models/tiny/config.json"), access),
        roots,
        "model_path_not_directory",
    )


def test_empty_directory(access: SourceAccess, roots: dict[str, Path]) -> None:
    (roots["data"] / "empty").mkdir()
    _expect(
        ErrorCode.SOURCE_NOT_FOUND,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/empty"), access),
        roots,
        "empty_directory",
    )


# ---------------------------------------------------------------- symlinks (security)


def test_symlink_target_escaping_root_is_rejected(
    access: SourceAccess, roots: dict[str, Path]
) -> None:
    (roots["data"] / "leak.jsonl").symlink_to(roots["outside"] / "secret.txt")
    (roots["data"] / "leakdir").symlink_to(roots["outside"], target_is_directory=True)
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/leak.jsonl"), access),
        roots,
        "symlink_escape",
    )
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/leakdir"), access),
        roots,
        "symlink_escape",
    )
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(
            DatasetSourceRef(reference="local:data/leakdir/secret.txt"), access
        ),
        roots,
        "symlink_escape",
    )


def test_symlink_inside_walked_directory_escaping_root(
    access: SourceAccess, roots: dict[str, Path]
) -> None:
    (roots["data"] / "dpo" / "evil.jsonl").symlink_to(roots["outside"] / "secret.txt")
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access),
        roots,
        "symlink_escape",
    )


def test_symlink_relative_escape_via_parent(access: SourceAccess, roots: dict[str, Path]) -> None:
    (roots["models"] / "tiny" / "up.json").symlink_to(
        Path("..") / ".." / ".." / "outside" / "secret.txt"
    )
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_model(ModelSourceRef(reference="local:models/tiny"), access),
        roots,
        "symlink_escape",
    )


def test_symlinked_directory_inside_root_is_rejected(
    access: SourceAccess, roots: dict[str, Path]
) -> None:
    (roots["data"] / "dpo" / "loop").symlink_to(roots["data"] / "dpo", target_is_directory=True)
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access),
        roots,
        "symlinked_directory",
    )


def test_symlinked_file_inside_root_is_allowed(
    access: SourceAccess, roots: dict[str, Path]
) -> None:
    # HF-cache style: snapshot files are symlinks into a blobs/ directory of the same root
    blobs = roots["data"] / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_text('{"q": "z"}\n', encoding="utf-8")
    snap = roots["data"] / "snap"
    snap.mkdir()
    (snap / "train.jsonl").symlink_to(Path("..") / "blobs" / "abc")
    manifest = resolve_dataset(DatasetSourceRef(reference="local:data/snap"), access).manifest
    assert [(f.path, f.size) for f in manifest.files] == [("train.jsonl", 11)]
    assert manifest.files[0].sha256 == _sha(blobs / "abc")


def test_broken_symlink(access: SourceAccess, roots: dict[str, Path]) -> None:
    (roots["data"] / "dpo" / "dangling.jsonl").symlink_to(roots["data"] / "nowhere")
    _expect(
        ErrorCode.SOURCE_NOT_FOUND,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access),
        roots,
        "broken_symlink",
    )


def test_root_registered_through_a_symlink(roots: dict[str, Path], tmp_path: Path) -> None:
    link = tmp_path / "root-link"
    link.symlink_to(roots["data"], target_is_directory=True)
    access = SourceAccess(local_roots={"data": link})
    source = resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access)
    assert source.local_path == (roots["data"] / "dpo").resolve()


def test_upload_symlink_escaping_uploads_dir(access: SourceAccess, roots: dict[str, Path]) -> None:
    (roots["uploads"] / "escape-upload-1234").symlink_to(roots["outside"], target_is_directory=True)
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="upload:escape-upload-1234"), access),
        roots,
        "symlink_escape",
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_special_files_are_rejected(access: SourceAccess, roots: dict[str, Path]) -> None:
    os.mkfifo(roots["data"] / "dpo" / "pipe")
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access),
        roots,
        "special_file",
    )
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo/pipe"), access),
        roots,
        "special_file",
    )


def test_mount_boundary_is_checked(
    access: SourceAccess, roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vramforge_estimator.sources import local

    real_stat = os.stat
    other_dev = real_stat(roots["data"]).st_dev + 1
    nested = (roots["data"] / "dpo").resolve()

    class _Stat:
        def __init__(self, base: os.stat_result) -> None:
            self._base = base

        def __getattr__(self, name: str) -> object:
            if name == "st_dev":
                return other_dev
            return getattr(self._base, name)

    def fake_stat(path: object, *args: object, **kwargs: object) -> object:
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        return _Stat(result) if Path(str(path)) == nested else result

    monkeypatch.setattr(local.os, "stat", fake_stat)
    _expect(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        lambda: resolve_dataset(DatasetSourceRef(reference="local:data/dpo"), access),
        roots,
        "mount_boundary",
    )
