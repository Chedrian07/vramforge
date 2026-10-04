"""File access for dataset sources: local files/dirs and pinned Hugging Face dataset repos.

- Local: files are read in place. Every data file must resolve (symlinks included) inside the
  source directory (plan §18 path escape).
- Hugging Face: data files are downloaded on demand, one shard at a time, into the hub cache under
  `access.hf_home` (`hf_hub_download` at the pinned commit; data and card files only). Previews of
  files larger than the preview budget use ranged reads (`HfFileSystem`) instead of a download.
  Without a user token, no locally stored token is used (`token=False`).

Integrity (plan §7.7 "manifest 일치", §16.2): before a shard is read its size and, when the
manifest has one, its content digest (sha256, or the git blob id for non-LFS repo files) are
compared with the manifest; local files are re-checked (size, mtime, inode) after reading.

Failed Hub requests (a download, or a ranged read of a previewed file) never surface as raw
library errors: `remote_issue` turns them into an `EstimatorError` with SOURCE_ACCESS_DENIED
(401/403, repository no longer visible), SOURCE_REVISION_CHANGED (the pinned revision or file is
gone) or, for network errors, timeouts and server errors, a retryable MODEL_METADATA_UNAVAILABLE.
The issue carries `request` ("download" | "read"), the file and the exception type, never the
exception text (it can contain URLs and credentials).
"""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, cast

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, FileEntry, Issue, Severity, Stage
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .readers import FileSource, ReaderLimits

SourceKind = Literal["hf", "local_dir", "local_file"]
RemoteRequest = Literal["download", "read"]
_HASH_BLOCK = 4 << 20
# Hub files up to this size are downloaded for the preview (the scan reuses the cached copy);
# larger ones are previewed with ranged reads so inspection stays fast.
PREVIEW_DOWNLOAD_MAX = 16 << 20


@dataclass(frozen=True)
class DataFile:
    """One data file of a split. `shard_id` is display-safe; `location` is internal only."""

    shard_id: str  # repo-relative / directory-relative path, or the file name
    location: str  # repo path (HF) or absolute container path (local) — never shown to users
    size: int | None = None
    entry: FileEntry | None = None  # matching manifest entry, when found


@dataclass(frozen=True)
class FileSignature:
    size: int
    mtime_ns: int
    inode: int


class ManifestIndex:
    """Looks up manifest entries by relative path. Matches are exact: a file outside the pinned
    list must never be verified against another entry with the same name."""

    def __init__(self, entries: list[FileEntry]) -> None:
        self._by_path = {_norm(entry.path): entry for entry in entries}

    def __len__(self) -> int:
        return len(self._by_path)

    def find(self, rel_path: str) -> FileEntry | None:
        return self._by_path.get(_norm(rel_path))


def _norm(path: str) -> str:
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")


class SourceFiles:
    """Opens, downloads and verifies the files of one resolved dataset source."""

    def __init__(self, source: ResolvedSource, access: SourceAccess, limits: ReaderLimits) -> None:
        self.source = source
        self.access = access
        self.limits = limits
        self.index = ManifestIndex(list(source.manifest.files))
        if source.repo_id is not None:
            if not source.revision:
                raise EstimatorError(
                    _issue(
                        ErrorCode.SOURCE_REVISION_CHANGED,
                        "데이터셋 revision이 고정되지 않았습니다. 다시 확인해 주세요.",
                        reason="revision_missing",
                    )
                )
            self.kind: SourceKind = "hf"
            self.root: Path | None = None
        elif source.local_path is not None:
            path = Path(source.local_path)
            if path.is_dir():
                self.kind = "local_dir"
            elif path.is_file():
                self.kind = "local_file"
            else:
                raise EstimatorError(
                    _issue(
                        ErrorCode.SOURCE_NOT_FOUND,
                        "데이터셋 경로를 찾을 수 없습니다.",
                        reason="local_path_missing",
                    )
                )
            self.root = path.resolve()
        else:
            raise EstimatorError(
                _issue(ErrorCode.SOURCE_NOT_FOUND, "데이터셋 위치가 지정되지 않았습니다.")
            )

    # ------------------------------------------------------------------ metadata (card files)

    def metadata_path(self, rel_path: str) -> Path | None:
        """README.md / .huggingface.yaml / dataset_infos.json, when present (size-capped)."""
        cap = self.access.max_metadata_bytes
        if self.kind == "hf":
            if not self.has_repo_file(rel_path):
                return None
            entry = self.index.find(rel_path)
            if entry is not None and entry.size is not None and entry.size > cap:
                raise EstimatorError(
                    _issue(
                        ErrorCode.SCAN_QUOTA_EXCEEDED,
                        "데이터셋 메타데이터 파일이 허용 크기를 넘습니다.",
                        reason="metadata_too_large",
                        file=rel_path,
                        limit_bytes=cap,
                    )
                )
            return self._download(rel_path)
        if self.kind != "local_dir" or self.root is None:
            return None
        path = self.root / rel_path
        if not path.is_file():
            return None
        self.check_inside(path, rel_path)
        if path.stat().st_size > cap:
            raise EstimatorError(
                _issue(
                    ErrorCode.SCAN_QUOTA_EXCEEDED,
                    "데이터셋 메타데이터 파일이 허용 크기를 넘습니다.",
                    reason="metadata_too_large",
                    file=rel_path,
                    limit_bytes=cap,
                )
            )
        return path

    def has_repo_file(self, rel_path: str) -> bool:
        if self.kind == "hf":
            entry = self.index.find(rel_path)
            return entry is not None and _norm(entry.path) == _norm(rel_path)
        return (
            self.root is not None and self.kind == "local_dir" and (self.root / rel_path).is_file()
        )

    # ------------------------------------------------------------------ data files

    def check_inside(self, path: Path, shard_id: str) -> None:
        """Reject files that resolve outside the source directory (symlink escape)."""
        if self.root is None:
            return
        real = Path(os.path.realpath(path))
        if self.kind == "local_file":
            if real != self.root:
                raise EstimatorError(_escape_issue(shard_id))
            return
        try:
            real.relative_to(self.root)
        except ValueError:
            raise EstimatorError(_escape_issue(shard_id)) from None

    def scan_source(self, data_file: DataFile) -> FileSource:
        """A local, fully readable copy of the file (downloaded on demand for HF)."""
        if data_file.size is not None and data_file.size > self.limits.max_file_bytes:
            raise _quota_file(data_file)
        if self.kind == "hf":
            path = self._download(data_file.location, stage=Stage.TOKENIZING)
        else:
            path = Path(data_file.location)
            self.check_inside(path, data_file.shard_id)
        return FileSource(data_file.shard_id, path=path, size=_size(path))

    def preview_source(self, data_file: DataFile, budget: int) -> FileSource:
        """A byte-budgeted reader for the schema preview (no full download of large files)."""
        if self.kind != "hf":
            path = Path(data_file.location)
            self.check_inside(path, data_file.shard_id)
            return FileSource(data_file.shard_id, path=path, size=_size(path), byte_budget=budget)
        if data_file.size is not None and data_file.size <= min(budget, PREVIEW_DOWNLOAD_MAX):
            path = self._download(data_file.location)
            return FileSource(data_file.shard_id, path=path, size=_size(path), byte_budget=budget)
        return FileSource(
            data_file.shard_id,
            opener=self._remote_opener(data_file.location),
            size=data_file.size,
            byte_budget=budget,
        )

    # ------------------------------------------------------------------ integrity

    def signature(self, path: Path) -> FileSignature:
        stat = path.stat()
        return FileSignature(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)

    def verify(self, data_file: DataFile, path: Path) -> Issue | None:
        """Compare size and content digest with the manifest entry (None = consistent).

        The resolver lists every file of the snapshot, so a data file without an entry appeared
        after resolution. An empty manifest (nothing listed) cannot be checked.
        """
        entry = data_file.entry
        if entry is None:
            return (
                _changed_issue(data_file.shard_id, "not_in_manifest") if len(self.index) else None
            )
        size = _size(path)
        if entry.size is not None and size != entry.size:
            return _changed_issue(data_file.shard_id, "size_mismatch")
        if entry.sha256:
            if _file_digest(path, "sha256") != entry.sha256.lower().removeprefix("sha256:"):
                return _changed_issue(data_file.shard_id, "sha256_mismatch")
        elif (
            entry.blob_id
            and self.kind == "hf"
            and _git_blob_id(path, size) != entry.blob_id.lower()
        ):
            return _changed_issue(data_file.shard_id, "blob_id_mismatch")
        return None

    # ------------------------------------------------------------------ Hugging Face

    def _token(self) -> str | bool:
        return self.access.hf_token or False  # never fall back to a locally stored token

    def _cache_dir(self) -> Path | None:
        return self.access.hf_home / "hub" if self.access.hf_home is not None else None

    def _download(self, rel_path: str, *, stage: Stage = Stage.INSPECTING) -> Path:
        from huggingface_hub import hf_hub_download

        assert self.source.repo_id is not None
        try:
            path = hf_hub_download(
                repo_id=self.source.repo_id,
                filename=rel_path,
                repo_type="dataset",
                revision=self.source.revision,
                cache_dir=self._cache_dir(),
                token=self._token(),
                etag_timeout=self.access.http_timeout_s,
            )
        except Exception as exc:  # incl. Xet transfer errors that are not OSError subclasses
            raise EstimatorError(remote_issue(exc, rel_path, "download", stage)) from None
        return Path(str(path))

    def _remote_opener(self, rel_path: str) -> Callable[[], IO[bytes]]:
        def opener() -> IO[bytes]:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem(token=self._token(), skip_instance_cache=True)
            remote = f"datasets/{self.source.repo_id}@{self.source.revision}/{rel_path}"
            # Opening resolves the revision and stats the file (requests); reads are ranged GETs.
            handle = _request(rel_path, fs.open, remote, "rb", block_size=1 << 20)
            return cast(IO[bytes], RemoteFile(cast(IO[bytes], handle), rel_path))

        return opener


class RemoteFile(io.RawIOBase):
    """A remote Hub file read with ranged requests; a failed request raises the `remote_issue`.

    Errors that are not request failures (e.g. fsspec's ValueError for a seek before the start of
    a corrupt file) pass through unchanged, so the readers still report them as data problems.
    """

    def __init__(self, raw: IO[bytes], rel_path: str) -> None:
        super().__init__()
        self._raw = raw
        self._rel_path = rel_path

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return bool(self._raw.seekable())

    def read(self, size: int | None = -1) -> bytes:
        return _request(self._rel_path, self._raw.read, -1 if size is None else size)

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)  # positions only; the next read fetches

    def tell(self) -> int:
        return self._raw.tell()

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def _request[T](rel_path: str, call: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return call(*args, **kwargs)
    except Exception as exc:
        if not is_request_failure(exc):
            raise
        raise EstimatorError(remote_issue(exc, rel_path, "read")) from None


def is_request_failure(exc: BaseException) -> bool:
    """An HTTP status, transport, timeout or offline error of a Hub request (not a data error)."""
    import httpx

    return isinstance(exc, OSError | httpx.HTTPError)


def _size(path: Path) -> int:
    return path.stat().st_size


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_id(path: Path, size: int) -> str:
    """git's blob object id: sha1(b"blob <size>\\0" + content) (non-LFS repo files)."""
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _issue(
    code: ErrorCode, message: str, *, severity: Severity = Severity.ERROR, **details: Any
) -> Issue:
    return make_issue(
        code, message, severity=severity, stage=Stage.INSPECTING, component="dataset", **details
    )


def _escape_issue(shard_id: str) -> Issue:
    return _issue(
        ErrorCode.LOCAL_PATH_NOT_ALLOWED,
        "데이터 파일이 허용된 데이터셋 경로 밖을 가리킵니다(심볼릭 링크). 읽지 않았습니다.",
        reason="path_escape",
        shard_id=shard_id,
    )


def _changed_issue(shard_id: str, reason: str) -> Issue:
    return make_issue(
        ErrorCode.SOURCE_REVISION_CHANGED,
        "데이터 파일 내용이 확인한 snapshot과 다릅니다. 데이터를 다시 확인해 주세요.",
        stage=Stage.TOKENIZING,
        component="dataset",
        reason=reason,
        shard_id=shard_id,
    )


def _quota_file(data_file: DataFile) -> EstimatorError:
    return EstimatorError(
        _issue(
            ErrorCode.SCAN_QUOTA_EXCEEDED,
            "데이터 파일이 허용된 파일 크기 한도를 넘어 읽지 않았습니다.",
            reason="max_file_bytes",
            shard_id=data_file.shard_id,
        )
    )


def remote_issue(
    exc: BaseException,
    rel_path: str,
    request: RemoteRequest,
    stage: Stage = Stage.INSPECTING,
) -> Issue:
    """The contract issue of a failed Hub request for one dataset file (module docstring)."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    cause = exc
    if isinstance(exc, FileNotFoundError) and isinstance(exc.__cause__, OSError):
        cause = exc.__cause__  # HfFileSystem wraps repository/revision lookups this way
    response = getattr(cause, "response", None)
    status = getattr(response, "status_code", None)
    details: dict[str, Any] = {
        "request": request,
        "file": rel_path,
        "error_type": type(cause).__name__,
    }
    if isinstance(status, int):
        details["http_status"] = status

    def issue(code: ErrorCode, message: str, reason: str, retryable: bool = False) -> Issue:
        return make_issue(
            code,
            message,
            stage=stage,
            retryable=retryable,
            component="dataset",
            reason=reason,
            **details,
        )

    # A private, gated (GatedRepoError is a RepositoryNotFoundError) or deleted repository answers
    # 401/403/404 "repository not found": the pinned snapshot is no longer readable with this token.
    if isinstance(cause, RepositoryNotFoundError) or status in (401, 403):
        return issue(
            ErrorCode.SOURCE_ACCESS_DENIED,
            "데이터셋에 접근할 수 없습니다. 접근 권한이나 토큰을 확인해 주세요.",
            "access_denied",
        )
    if isinstance(cause, OfflineModeIsEnabled):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "서버가 오프라인 모드로 설정되어 Hugging Face에서 데이터셋 파일을 가져올 수 "
            "없습니다. 관리자 설정을 확인해 주세요.",
            "offline_mode",
        )
    # LocalEntryNotFoundError: the Hub was unreachable and the file is not cached (network).
    if isinstance(cause, RevisionNotFoundError) or (
        isinstance(cause, EntryNotFoundError | FileNotFoundError)
        and not isinstance(cause, LocalEntryNotFoundError)
    ):
        return issue(
            ErrorCode.SOURCE_REVISION_CHANGED,
            "고정한 revision에서 데이터 파일을 찾을 수 없습니다. 데이터셋을 다시 확인해 주세요.",
            "entry_not_found",
        )
    action = "받지" if request == "download" else "읽지"
    return issue(
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        f"Hugging Face에서 데이터셋 파일을 {action} 못했습니다(네트워크 오류, 시간 초과 또는 "
        "서버 오류). 잠시 후 다시 시도해 주세요.",
        "network",
        retryable=True,
    )
