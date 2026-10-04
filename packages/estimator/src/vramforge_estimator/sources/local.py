"""Local roots and uploads: confinement checks and content identity (plan.md §6.1, §16.2, §18).

Paths are resolved below an admin-registered root (``SourceAccess.local_roots``) or the uploads
directory. ``..``/absolute parts are rejected during normalization; here every resolved path and
every file found while walking must stay inside the root (no symlink escape, no other mount, no
special files). Identity is a sha256 manifest over (relative path, size, content id) — never
mtime. Dataset files are hashed in full; model directories hash config/tokenizer/template/index
files in full and safetensors weights by header bytes + size (payloads are not read).
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.keys import source_key, stable_hash
from vramforge_estimator.schemas import ErrorCode, FileEntry, SourceManifest, SourceType, Stage

from .base import ResolvedSource, SourceAccess
from .issues import blocking_error
from .references import Kind, NormalizedReference
from .safetensors_frame import HeaderFrameError, read_header_frame

MAX_LOCAL_FILES = 100_000
MANIFEST_VERSION = "vramforge.local-manifest.v1"
# Never read (never unpickled); identity is path + size only.
WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".ggml",
        ".gguf",
        ".h5",
        ".msgpack",
        ".nemo",
        ".npy",
        ".npz",
        ".onnx",
        ".ot",
        ".pb",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".tflite",
    }
)


@dataclass(frozen=True)
class LocalFile:
    relative: str  # posix path relative to the target directory (or the file name)
    real: Path  # resolved path inside the root
    size: int


def _error(code: ErrorCode, message: str, kind: Kind, **details: object) -> EstimatorError:
    return blocking_error(code, message, stage=Stage.RESOLVING, component=kind, details=details)


def _escape(kind: Kind, reason: str) -> EstimatorError:
    return _error(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        "등록된 root 밖을 가리키거나(심볼릭 링크·다른 마운트 포함) 일반 파일이 아닌 항목이 있어 "
        "이 경로를 사용할 수 없습니다.",
        kind,
        reason=reason,
    )


def _not_found(kind: Kind, message: str, reason: str) -> EstimatorError:
    return _error(ErrorCode.SOURCE_NOT_FOUND, message, kind, reason=reason)


def resolve_local_target(norm: NormalizedReference, access: SourceAccess) -> tuple[Path, Path]:
    """(real root, logical target) for ``local:<root>/<relative>``.

    The logical target keeps the referenced names (a symlinked file keeps its own name and
    extension) and is verified to resolve inside the root.
    """
    kind = norm.kind
    assert norm.local_root is not None and norm.local_relative is not None
    root = access.local_roots.get(norm.local_root)
    if root is None:
        raise _error(
            ErrorCode.LOCAL_PATH_NOT_ALLOWED,
            "등록되지 않은 로컬 root입니다. 허용된 root 목록에서 선택하세요.",
            kind,
            reason="unknown_root",
            root=norm.local_root,
        )
    root_real = _real_root(root, kind)
    target = (
        root_real.joinpath(*norm.local_relative.split("/")) if norm.local_relative else root_real
    )
    _confined(target, root_real, kind)
    return root_real, target


def resolve_upload_target(norm: NormalizedReference, access: SourceAccess) -> tuple[Path, Path]:
    """(real uploads dir, logical target) for ``upload:<id>``.

    `access.uploads_dir` must already be scoped to the requesting owner (the API decides).
    """
    kind = norm.kind
    assert norm.upload_id is not None
    if access.uploads_dir is None:
        raise _error(
            ErrorCode.LOCAL_PATH_NOT_ALLOWED,
            "이 서버에는 업로드 저장소가 설정되어 있지 않습니다.",
            kind,
            reason="uploads_disabled",
        )
    root_real = _real_root(access.uploads_dir, kind)
    try:
        _confined(root_real / norm.upload_id, root_real, kind)
    except EstimatorError as exc:
        if exc.issue.code is ErrorCode.SOURCE_NOT_FOUND:
            raise _not_found(
                kind,
                "업로드를 찾을 수 없습니다. 만료되었거나 삭제되었을 수 있습니다.",
                "upload_missing",
            ) from None
        raise
    return root_real, root_real / norm.upload_id


def _real_root(root: Path, kind: Kind) -> Path:
    try:
        real = root.resolve(strict=True)
    except OSError:
        raise _not_found(
            kind,
            "등록된 로컬 root를 서버에서 찾을 수 없습니다. 관리자 설정을 확인하세요.",
            "root_missing",
        ) from None
    if not real.is_dir():
        raise _not_found(kind, "등록된 로컬 root가 디렉터리가 아닙니다.", "root_not_directory")
    return real


def _confined(target: Path, root_real: Path, kind: Kind) -> Path:
    try:
        real = target.resolve(strict=True)
    except OSError:
        raise _not_found(kind, "지정한 로컬 경로가 없습니다.", "path_missing") from None
    if not real.is_relative_to(root_real):
        raise _escape(kind, "symlink_escape")
    if os.stat(real).st_dev != os.stat(root_real).st_dev:
        raise _escape(kind, "mount_boundary")
    return real


def walk_local_files(root_real: Path, target: Path, kind: Kind) -> list[LocalFile]:
    """Every regular file under `target` (or the file itself), sorted by relative path."""
    root_dev = os.stat(root_real).st_dev
    target_real = target.resolve(strict=True)
    target_info = target_real.stat()
    if stat.S_ISREG(target_info.st_mode):
        return [LocalFile(target.name, target_real, target_info.st_size)]
    if not stat.S_ISDIR(target_info.st_mode):
        raise _escape(kind, "special_file")
    files: list[LocalFile] = []
    stack: list[tuple[Path, str]] = [(target_real, "")]
    while stack:
        directory, prefix = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}{entry.name}"
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    real = _follow_link(Path(entry.path), root_real, kind)
                    info = real.stat()
                    if stat.S_ISDIR(info.st_mode):
                        raise _escape(kind, "symlinked_directory")
                else:
                    real = Path(entry.path)
                if info.st_dev != root_dev:
                    raise _escape(kind, "mount_boundary")
                if stat.S_ISDIR(info.st_mode):
                    stack.append((real, f"{relative}/"))
                elif stat.S_ISREG(info.st_mode):
                    files.append(LocalFile(relative, real, info.st_size))
                    if len(files) > MAX_LOCAL_FILES:
                        raise _error(
                            ErrorCode.SCAN_QUOTA_EXCEEDED,
                            f"경로 안의 파일이 너무 많습니다(최대 {MAX_LOCAL_FILES:,}개).",
                            kind,
                            reason="too_many_files",
                            limit=MAX_LOCAL_FILES,
                        )
                else:
                    raise _escape(kind, "special_file")
    if not files:
        raise _not_found(kind, "지정한 로컬 경로에 파일이 없습니다.", "empty_directory")
    return sorted(files, key=lambda f: f.relative)


def _follow_link(link: Path, root_real: Path, kind: Kind) -> Path:
    try:
        real = link.resolve(strict=True)
    except OSError:
        raise _not_found(kind, "깨진 심볼릭 링크가 있습니다.", "broken_symlink") from None
    if not real.is_relative_to(root_real):
        raise _escape(kind, "symlink_escape")
    return real


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def content_entries(
    files: list[LocalFile], kind: Kind, *, max_metadata_bytes: int
) -> tuple[list[FileEntry], list[tuple[str, int, str]], list[str]]:
    """FileEntry list, identity tuples (path, size, content id) and manifest notes."""
    entries: list[FileEntry] = []
    identity: list[tuple[str, int, str]] = []
    header_only = False
    size_only: list[str] = []
    for item in files:
        suffix = Path(item.relative).suffix.lower()
        sha: str | None = None
        if kind == "dataset":
            sha = _sha256_file(item.real)
            content = f"sha256:{sha}"
        elif suffix == ".safetensors":
            header_only = True
            try:
                frame = read_header_frame(item.real, max_header_bytes=max_metadata_bytes)
                content = f"safetensors-header:{frame.digest}"
            except HeaderFrameError as exc:
                content = f"safetensors-header-unreadable:{exc.reason}"
        elif suffix in WEIGHT_SUFFIXES or item.size > max_metadata_bytes:
            size_only.append(item.relative)
            content = "size-only"
        else:
            sha = _sha256_file(item.real)
            content = f"sha256:{sha}"
        entries.append(FileEntry(path=item.relative, size=item.size, sha256=sha))
        identity.append((item.relative, item.size, content))

    notes: list[str] = []
    if header_only:
        notes.append(
            "가중치(*.safetensors)는 header와 크기만 identity에 넣었습니다. "
            "payload(가중치 값)는 hash하지 않습니다."
        )
    if size_only:
        notes.append(
            f"파일 {len(size_only)}개(가중치 형식이거나 큰 파일)는 "
            "내용 대신 크기만 identity에 넣었습니다."
        )
    return entries, identity, notes


def local_identity(kind: Kind, identity: list[tuple[str, int, str]]) -> str:
    return "sha256:" + stable_hash(MANIFEST_VERSION, kind, identity)


def build_local_source(norm: NormalizedReference, access: SourceAccess) -> ResolvedSource:
    kind = norm.kind
    if norm.source_type is SourceType.UPLOAD:
        root_real, target = resolve_upload_target(norm, access)
    else:
        root_real, target = resolve_local_target(norm, access)
    if kind == "model" and not target.is_dir():
        raise _error(
            ErrorCode.INVALID_REQUEST,
            "모델 경로는 config.json과 가중치가 들어 있는 디렉터리여야 합니다.",
            kind,
            reason="model_path_not_directory",
        )
    try:
        files = walk_local_files(root_real, target, kind)
        entries, identity, notes = content_entries(
            files, kind, max_metadata_bytes=access.max_metadata_bytes
        )
    except PermissionError:
        raise _error(
            ErrorCode.SOURCE_ACCESS_DENIED,
            "서버가 이 로컬 경로의 파일을 읽을 권한이 없습니다.",
            kind,
            reason="permission_denied",
        ) from None
    except FileNotFoundError:
        raise _not_found(
            kind,
            "확인 중에 파일이 사라졌습니다. 경로가 바뀌고 있는지 확인하세요.",
            "changed_during_walk",
        ) from None
    except OSError as exc:
        raise blocking_error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "로컬 파일을 읽는 중 입출력 오류가 났습니다. 잠시 후 다시 시도하세요.",
            stage=Stage.RESOLVING,
            retryable=True,
            component=kind,
            details={"reason": "io_error", "error_type": type(exc).__name__},
        ) from None
    resolved = local_identity(kind, identity)
    if norm.revision is not None and norm.revision != resolved:
        raise _error(
            ErrorCode.SOURCE_REVISION_CHANGED,
            "로컬 source의 내용이 지정한 digest와 다릅니다. 파일이 바뀌었는지 확인하세요.",
            kind,
            reason="content_digest_mismatch",
            expected=norm.revision,
            actual=resolved,
        )
    manifest = SourceManifest(
        kind=kind,
        source_type=norm.source_type,
        reference=norm.display,
        requested_revision=norm.revision,
        resolved_revision=resolved,
        files=entries,
        fingerprint=source_key(norm.source_type.value, f"{kind}:{resolved}"),
        notes=[*norm.notes, *notes],
    )
    return ResolvedSource(kind=kind, manifest=manifest, local_path=target)
