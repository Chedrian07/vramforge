"""Streaming dataset uploads (plan.md §2.1, §18).

The multipart body is parsed incrementally and the file part is written straight to the owner's
upload directory with a byte cap, so an oversized upload never fills a temp directory first.
Only JSON/JSONL/Parquet/Arrow/CSV are accepted: the extension must match the sniffed content.
The client filename is reduced to a safe basename and never used to build a path outside the
upload's own directory.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import unicodedata
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from python_multipart.multipart import MultipartParser, parse_options_header

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import ErrorCode, Stage

from .errors import ApiError, BodyTooLarge

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
}
HEAD_BYTES = 8192
FLUSH_BYTES = 1024 * 1024
MAX_FILENAME = 120
MAX_FIELD_BYTES = 4096


def _reject_type(message: str) -> ApiError:
    return ApiError(
        415,
        make_issue(
            ErrorCode.UPLOAD_TYPE_NOT_ALLOWED,
            message,
            stage=Stage.API,
            allowed=sorted(ALLOWED_EXTENSIONS),
        ),
    )


def _too_large(max_bytes: int) -> ApiError:
    return ApiError(
        413,
        make_issue(
            ErrorCode.UPLOAD_TOO_LARGE,
            "업로드 파일이 허용 크기를 넘었습니다.",
            stage=Stage.API,
            max_bytes=max_bytes,
        ),
    )


def _bad_request(message: str) -> ApiError:
    return ApiError(422, make_issue(ErrorCode.INVALID_REQUEST, message, stage=Stage.REQUEST))


def sanitize_filename(raw: str) -> str:
    """Safe basename: no directories, no hidden files, ASCII `[A-Za-z0-9._-]` only."""
    name = unicodedata.normalize("NFKC", raw or "").replace("\\", "/").split("/")[-1]
    name = "".join(
        ch if (ch.isascii() and ch.isalnum()) or ch in "._-" else "_"
        for ch in name
        if ch.isprintable()
    )
    name = name.lstrip(".-")
    stem, ext = os.path.splitext(name)
    stem = stem.strip("._-")[: MAX_FILENAME - len(ext)] or "dataset"
    return f"{stem}{ext.lower()}"


def detect_format(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    fmt = ALLOWED_EXTENSIONS.get(ext)
    if fmt is None:
        raise _reject_type(
            "허용되지 않는 파일 형식입니다. "
            "JSON, JSONL, Parquet, Arrow, CSV만 업로드할 수 있습니다."
        )
    return fmt


def _text_head(head: bytes) -> str | None:
    if b"\x00" in head:
        return None
    for cut in range(4):  # a multi-byte character may be split at the sniff boundary
        try:
            return head[: len(head) - cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return None


def content_matches(fmt: str, head: bytes, tail: bytes) -> bool:
    if fmt == "parquet":
        return head.startswith(b"PAR1") and tail.endswith(b"PAR1")
    if fmt == "arrow":
        # Arrow IPC file magic, or the stream format's continuation marker (datasets' .arrow).
        return head.startswith(b"ARROW1") or head.startswith(b"\xff\xff\xff\xff")
    text = _text_head(head)
    if text is None:
        return False
    stripped = text.lstrip("\ufeff \t\r\n")
    if fmt == "json":
        return stripped[:1] in ("[", "{")
    if fmt == "jsonl":
        return stripped[:1] == "{"
    if fmt == "csv":
        return bool(stripped)
    return False


@dataclass
class SavedUpload:
    filename: str
    format: str
    size_bytes: int
    sha256: str
    path: Path


@dataclass
class _PartState:
    headers: dict[bytes, bytes] = field(default_factory=dict)
    header_field: bytearray = field(default_factory=bytearray)
    header_value: bytearray = field(default_factory=bytearray)
    is_file: bool = False
    field_bytes: int = 0


class _UploadSink:
    """Collects exactly one file part from a multipart stream into `target_dir`."""

    def __init__(
        self,
        target_dir: Path,
        max_bytes: int,
        too_large: Callable[[], ApiError] | None = None,
    ) -> None:
        self.target_dir = target_dir
        self.max_bytes = max_bytes
        self.too_large = too_large or (lambda: _too_large(max_bytes))
        self.part = _PartState()
        self.filename: str | None = None
        self.format: str | None = None
        self.temp_path: Path | None = None
        self._fh: object | None = None
        self.size = 0
        self.digest = hashlib.sha256()
        self.head = bytearray()
        self.tail = b""
        self.files = 0
        self.error: ApiError | None = None

    # -- python-multipart callbacks (synchronous) --------------------------------
    def on_part_begin(self) -> None:
        self.part = _PartState()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.part.header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.part.header_value += data[start:end]

    def on_header_end(self) -> None:
        self.part.headers[bytes(self.part.header_field).lower()] = bytes(self.part.header_value)
        self.part.header_field.clear()
        self.part.header_value.clear()

    def on_headers_finished(self) -> None:
        disposition, params = parse_options_header(
            self.part.headers.get(b"content-disposition", b"")
        )
        if disposition != b"form-data":
            self._fail(_bad_request("multipart 형식이 올바르지 않습니다."))
            return
        if params.get(b"name") != b"file" or b"filename" not in params:
            return  # other fields are ignored (size-capped)
        self.files += 1
        if self.files > 1:
            self._fail(_bad_request("파일은 한 번에 하나만 업로드할 수 있습니다."))
            return
        self.part.is_file = True
        raw_name = params[b"filename"].decode("utf-8", errors="replace")
        self.filename = sanitize_filename(raw_name)
        try:
            self.format = detect_format(self.filename)
        except ApiError as exc:
            self._fail(exc)
            return
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.target_dir / f".upload-{secrets.token_hex(8)}.part"
        self._fh = open(self.temp_path, "xb")  # noqa: SIM115 - closed in close()/finish()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self.error is not None:
            return
        chunk = data[start:end]
        if not self.part.is_file:
            self.part.field_bytes += len(chunk)
            if self.part.field_bytes > MAX_FIELD_BYTES:
                self._fail(_bad_request("요청 필드가 너무 큽니다."))
            return
        self.size += len(chunk)
        if self.size > self.max_bytes:
            self._fail(self.too_large())
            return
        if len(self.head) < HEAD_BYTES:
            self.head += chunk[: HEAD_BYTES - len(self.head)]
        self.tail = (self.tail + chunk)[-8:]
        self.digest.update(chunk)
        assert self._fh is not None
        self._fh.write(chunk)  # type: ignore[attr-defined]

    def on_part_end(self) -> None:
        self.part = _PartState()

    # -- control ------------------------------------------------------------------
    def _fail(self, error: ApiError) -> None:
        if self.error is None:
            self.error = error

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[attr-defined]
            self._fh = None

    def discard(self) -> None:
        self.close()
        if self.temp_path is not None:
            self.temp_path.unlink(missing_ok=True)

    def finish(self) -> SavedUpload:
        self.close()
        if self.error is not None:
            raise self.error
        if self.files == 0 or self.temp_path is None or self.filename is None:
            raise _bad_request("업로드할 파일(file 필드)이 없습니다.")
        assert self.format is not None
        if self.size == 0:
            raise _reject_type("빈 파일은 업로드할 수 없습니다.")
        if not content_matches(self.format, bytes(self.head), self.tail):
            raise _reject_type("파일 내용이 확장자와 맞지 않습니다.")
        final = self.target_dir / self.filename
        os.replace(self.temp_path, final)
        self.temp_path = None
        return SavedUpload(
            filename=self.filename,
            format=self.format,
            size_bytes=self.size,
            sha256=self.digest.hexdigest(),
            path=final,
        )


async def receive_upload(
    content_type: str | None,
    body: AsyncIterator[bytes],
    target_dir: Path,
    max_bytes: int,
    *,
    too_large: Callable[[], ApiError] | None = None,
) -> SavedUpload:
    """Stream a `multipart/form-data` body with one `file` part into `target_dir`.

    `too_large` builds the error for a file over `max_bytes` (default: the per-file cap message;
    the caller passes its own when the binding limit is the owner's remaining quota)."""
    ctype, params = parse_options_header(content_type or "")
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise _bad_request("multipart/form-data 형식으로 파일을 보내야 합니다.")
    sink = _UploadSink(target_dir, max_bytes, too_large)
    callbacks = {
        "on_part_begin": sink.on_part_begin,
        "on_part_data": sink.on_part_data,
        "on_part_end": sink.on_part_end,
        "on_header_field": sink.on_header_field,
        "on_header_value": sink.on_header_value,
        "on_header_end": sink.on_header_end,
        "on_headers_finished": sink.on_headers_finished,
    }
    try:
        parser = MultipartParser(boundary, callbacks)  # type: ignore[arg-type]
        pending = bytearray()
        stopped = False
        # Parsing and file writes run in a worker thread (batched) so a large upload never
        # blocks the event loop; at most FLUSH_BYTES are buffered in memory.
        async for chunk in body:
            if not chunk:
                continue
            pending += chunk
            if len(pending) >= FLUSH_BYTES:
                await anyio.to_thread.run_sync(parser.write, bytes(pending))
                pending.clear()
                if sink.error is not None:
                    stopped = True
                    break
        if not stopped:
            if pending:
                await anyio.to_thread.run_sync(parser.write, bytes(pending))
            if sink.error is None:
                parser.finalize()
        return await anyio.to_thread.run_sync(sink.finish)
    except (ApiError, BodyTooLarge):  # BodyTooLarge: the whole request passed its cap
        await anyio.to_thread.run_sync(sink.discard)
        raise
    except Exception as exc:
        await anyio.to_thread.run_sync(sink.discard)
        raise _bad_request("multipart 본문을 해석할 수 없습니다.") from exc


__all__ = [
    "ALLOWED_EXTENSIONS",
    "SavedUpload",
    "content_matches",
    "detect_format",
    "receive_upload",
    "sanitize_filename",
]
