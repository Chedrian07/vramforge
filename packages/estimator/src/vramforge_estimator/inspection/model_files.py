"""Read small metadata files and safetensors headers of a resolved model source.

Only files listed in the pinned manifest are read, each capped at
``SourceAccess.max_metadata_bytes``; weights are never downloaded. Downloaded or local bytes are
checked against the manifest (LFS sha256, git blob id, local sha256/size), so what is analysed is
exactly what was resolved (plan.md §16.2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, FileEntry, Stage
from vramforge_estimator.sources import ResolvedSource, SourceAccess, hub
from vramforge_estimator.sources.issues import blocking_error
from vramforge_estimator.sources.safetensors_frame import HeaderFrameError, read_header_frame

from .safetensors_header import HeaderError, parse_header_bytes


class SourceFiles(Protocol):
    """Pinned file set of one source."""

    def entries(self) -> dict[str, FileEntry]: ...

    def read(self, path: str) -> bytes: ...

    def header(self, path: str) -> tuple[dict[str, Any], int | None]:
        """Raw safetensors header mapping and the payload size when known."""
        ...


def _error(
    code: ErrorCode, message: str, *, retryable: bool = False, **details: object
) -> EstimatorError:
    return blocking_error(
        code,
        message,
        stage=Stage.INSPECTING,
        retryable=retryable,
        component="model",
        details=details,
    )


def _changed(path: str, reason: str) -> EstimatorError:
    return _error(
        ErrorCode.SOURCE_REVISION_CHANGED,
        "분석 중에 source 파일이 고정한 내용과 달라졌습니다. 다시 분석하세요.",
        reason=reason,
        file=path,
    )


def git_blob_id(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content, usedforsecurity=False).hexdigest()


def header_error(path: str, exc: HeaderError | HeaderFrameError) -> EstimatorError:
    details = dict(getattr(exc, "details", {}))
    if exc.reason == "unknown_dtype":
        return _error(
            ErrorCode.UNSUPPORTED_MODEL_FORMAT,
            "지원하지 않는 tensor dtype이 있어 가중치 크기를 계산할 수 없습니다.",
            reason="unknown_dtype",
            file=path,
            **details,
        )
    return _error(
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        "safetensors header를 읽을 수 없거나 형식이 올바르지 않습니다.",
        reason=f"malformed_safetensors_header:{exc.reason}",
        file=path,
        **details,
    )


class _BaseFiles:
    def __init__(self, source: ResolvedSource, access: SourceAccess) -> None:
        self._source = source
        self._limit = access.max_metadata_bytes
        self._entries = {entry.path: entry for entry in source.manifest.files}

    def entries(self) -> dict[str, FileEntry]:
        return dict(self._entries)

    def _entry(self, path: str) -> FileEntry:
        entry = self._entries.get(path)
        if entry is None:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "고정한 source에 필요한 파일이 없습니다.",
                reason="file_not_in_manifest",
                file=path,
            )
        if entry.size is not None and entry.size > self._limit:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "메타데이터 파일이 허용 크기보다 큽니다.",
                reason="file_too_large",
                file=path,
                size=entry.size,
                limit=self._limit,
            )
        return entry


class HubFiles(_BaseFiles):
    def __init__(self, source: ResolvedSource, access: SourceAccess) -> None:
        super().__init__(source, access)
        assert source.repo_id is not None and source.revision is not None
        self._repo_id = source.repo_id
        self._revision = source.revision
        self._client = hub.get_hub_client(access)

    def read(self, path: str) -> bytes:
        entry = self._entry(path)
        try:
            local = self._client.download_file("model", self._repo_id, self._revision, path)
            data = local.read_bytes() if local.stat().st_size <= self._limit else None
        except EstimatorError:
            raise
        except Exception as exc:
            if hub.is_hub_exception(exc):
                raise hub.hub_error(exc, kind="model", stage=Stage.INSPECTING) from None
            raise
        if data is None:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "메타데이터 파일이 허용 크기보다 큽니다.",
                reason="file_too_large",
                file=path,
            )
        if entry.sha256 is not None:
            ok = hashlib.sha256(data).hexdigest() == entry.sha256
        elif entry.blob_id is not None:
            ok = git_blob_id(data) == entry.blob_id
        else:
            ok = entry.size is None or entry.size == len(data)
        if not ok:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "내려받은 파일이 고정한 commit의 파일과 일치하지 않습니다.",
                reason="integrity_mismatch",
                file=path,
            )
        return data

    def header(self, path: str) -> tuple[dict[str, Any], int | None]:
        if path not in self._entries:
            self._entry(path)
        try:
            return self._client.read_safetensors_header(self._repo_id, self._revision, path), None
        except EstimatorError:
            raise
        except Exception as exc:
            if hub.is_hub_exception(exc):
                raise hub.hub_error(exc, kind="model", stage=Stage.INSPECTING) from None
            raise


class LocalFiles(_BaseFiles):
    def __init__(self, source: ResolvedSource, access: SourceAccess) -> None:
        super().__init__(source, access)
        assert source.local_path is not None
        self._base = source.local_path

    def _path(self, path: str) -> Path:
        try:
            return self._base.joinpath(*path.split("/")).resolve(strict=True)
        except OSError:
            raise _changed(path, "file_missing") from None

    def read(self, path: str) -> bytes:
        entry = self._entry(path)
        real = self._path(path)
        with real.open("rb") as handle:
            data = handle.read(self._limit + 1)
        if len(data) > self._limit:
            raise _changed(path, "size_changed")
        if entry.sha256 is not None:
            if hashlib.sha256(data).hexdigest() != entry.sha256:
                raise _changed(path, "content_changed")
        elif entry.size is not None and entry.size != len(data):
            raise _changed(path, "size_changed")
        return data

    def header(self, path: str) -> tuple[dict[str, Any], int | None]:
        if path not in self._entries:
            self._entry(path)
        entry = self._entries[path]
        try:
            frame = read_header_frame(self._path(path), max_header_bytes=self._limit)
            if entry.size is not None and frame.file_size != entry.size:
                raise _changed(path, "size_changed")
            return parse_header_bytes(frame.header), frame.payload_size
        except (HeaderError, HeaderFrameError) as exc:
            raise header_error(path, exc) from None
        except OSError:
            raise _changed(path, "file_missing") from None


def open_source_files(source: ResolvedSource, access: SourceAccess) -> SourceFiles:
    if source.repo_id is not None:
        return HubFiles(source, access)
    if source.local_path is not None:
        return LocalFiles(source, access)
    raise _error(
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        "source 위치 정보가 없습니다.",
        reason="unresolved_source",
    )
