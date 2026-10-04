"""Bounded readers for dataset files: shared limits, errors, byte streams and schema state.

The format readers (`readers_json`, `readers_tabular`) mirror the packaged builders of
datasets==5.0.1 (`packaged_modules/{json,csv,parquet,arrow}`) so that the rows, their order and
their Python values equal `datasets.load_dataset(...)[split]` for the same files. On top of the
builders they add what a full no-truncation scan needs (plan.md §7.6, §18):

- every byte read is counted against a quota (`ReaderLimits`); exceeding one raises
  `QuotaExceeded` instead of reading further;
- a JSON Lines record that cannot be parsed becomes a `RowError` (the file keeps being read),
  where datasets would abort the whole split;
- memory stays bounded: JSON is parsed in chunks, CSV in row chunks, Parquet per row group and
  Arrow per record batch, and rows are materialized in small slices.

Structural problems that make the rest of a file unreadable raise `ShardBroken`.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, cast

from vramforge_estimator.units import GiB, MiB

if TYPE_CHECKING:
    import pyarrow as pa
    from datasets import Features

Module = Literal["json", "csv", "parquet", "arrow"]
FileFormat = Literal[
    "json_lines", "json_array", "json_field", "csv", "parquet", "arrow_stream", "arrow_file"
]

SUPPORTED_MODULES: frozenset[str] = frozenset({"json", "csv", "parquet", "arrow"})
# Single-file compressions we can decode as a stream (datasets: COMPRESSION_FILESYSTEMS).
STREAM_COMPRESSIONS: frozenset[str] = frozenset({"gzip", "bz2", "xz", "zstd", "lz4"})
ROW_SLICE = 1024  # rows materialized as Python objects at a time


@dataclass(frozen=True)
class ReaderLimits:
    """Resource quotas of the dataset readers. Exceeding one stops the stream (plan §18)."""

    max_row_bytes: int = 64 * MiB  # one JSON Lines record / JSON array element / CSV row
    max_decompressed_bytes: int = 32 * GiB  # per compressed data file
    max_file_bytes: int = 512 * GiB  # per data file (download and read)
    json_array_max_bytes: int = 256 * MiB  # larger JSON arrays are parsed incrementally
    max_failed_rows: int = 100_000  # unparseable records before the stream gives up
    # one Parquet row group / Arrow record batch (uncompressed) / CSV 10,000-row chunk (read)
    max_batch_bytes: int = 2 * GiB
    preview_rows: int = 100


class ReaderError(Exception):
    """Base class; `reason` is a stable machine code, never raw data."""

    def __init__(self, reason: str, **detail: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail)


class QuotaExceeded(ReaderError):
    """A `ReaderLimits` (or preview/metadata) quota was hit; nothing past it was read."""

    def __init__(self, limit: str, value: int, **detail: object) -> None:
        super().__init__("quota_exceeded", limit=limit, value=value, **detail)
        self.limit = limit
        self.value = value


class ShardBroken(ReaderError):
    """The file is structurally unreadable from this point on (the shard stops)."""


class UnsupportedFormat(ReaderError):
    """The file (or its builder options) is outside what the readers support."""


# Korean explanations of `ReaderError.reason` codes (user-facing, no raw data).
REASON_MESSAGES: dict[str, str] = {
    "binary_content": "텍스트(JSON)가 아닌 바이너리 내용입니다.",
    "encoding": "지정한 문자 인코딩으로 읽을 수 없습니다.",
    "json_array_expected": "JSON 배열 형식이 아닙니다.",
    "json_array_unterminated": "JSON 배열이 닫히지 않았습니다(파일이 잘렸을 수 있습니다).",
    "json_trailing_data": "JSON 배열 뒤에 다른 내용이 있습니다.",
    "json_array_leading_whitespace": "JSON 배열 앞에 공백이나 줄바꿈이 있어 datasets가 이 파일을 "
    "읽지 못합니다. 파일이 '['로 시작하도록 고쳐 주세요.",
    "json_field_unreadable": "README의 field 설정으로 JSON을 읽을 수 없습니다.",
    "agent_traces": "datasets가 이 컬럼 구성을 agent trace로 해석해 row를 그대로 읽지 않습니다.",
    "csv_unreadable": "CSV로 해석할 수 없습니다.",
    "parquet_unreadable": "Parquet 파일 구조가 손상되었습니다.",
    "arrow_unreadable": "Arrow 파일 구조가 손상되었습니다.",
    "schema_mismatch": "다른 레코드나 파일과 컬럼 구성 또는 값의 형식이 달라 함께 읽을 수 "
    "없습니다.",
    "compressed_columnar": "압축된 Parquet/Arrow 파일은 지원하지 않습니다.",
    "compression_not_supported": "지원하지 않는 압축 형식입니다.",
    "parquet_filters": "row filter가 지정된 Parquet 설정은 지원하지 않습니다.",
    "parquet_skips_bad_files": "손상 파일을 건너뛰는 설정은 지원하지 않습니다.",
    "module_not_supported": "지원하지 않는 데이터 형식입니다.",
}


def reason_message(reason: str) -> str:
    return REASON_MESSAGES.get(reason, "파일을 읽을 수 없습니다.")


@dataclass(frozen=True)
class RowError:
    """A record that could not be decoded. `line` is 1-based within the (decompressed) file."""

    reason: str  # "json_parse" | "not_an_object" | "schema_mismatch" | "invalid_utf8"
    line: int | None = None
    element: int | None = None  # 0-based element index inside a JSON array


ReadItem = dict[str, Any] | RowError


@dataclass(frozen=True)
class FileSpec:
    """How one data file is read: the datasets builder module, its options and compression."""

    module: Module
    options: Mapping[str, Any] = field(default_factory=dict)  # BuilderConfig params
    compression: str | None = None


@dataclass
class SchemaState:
    """The Arrow schema locked by the first table of a split, like datasets' `info.features`.

    Later tables are cast to it (`datasets.table.table_cast`), and Json-typed fields are decoded
    back to Python objects exactly as the datasets Python formatter does.
    """

    features: Features | None = None
    json_field_paths: list[list[Any]] = field(default_factory=list)
    started: bool = False  # the first batch of the split was processed

    @classmethod
    def declared(cls, features: Features | None) -> SchemaState:
        """A state locked before any row is read (README `dataset_info` features), with the
        Json field paths datasets derives from them; None leaves the schema to be inferred."""
        state = cls(features=features)
        state.json_field_paths = state.decode_paths()
        return state

    def lock(self, table: pa.Table) -> pa.Table:
        from datasets import Features
        from datasets.table import table_cast

        if self.features is None:
            self.features = Features.from_arrow_schema(table.schema)
        return table_cast(table, self.features.arrow_schema)

    def decode_paths(self) -> list[list[Any]]:
        from datasets.utils.json import get_json_field_paths_from_feature

        return get_json_field_paths_from_feature(self.features) if self.features else []


@dataclass
class ReadInfo:
    """What a reader learned about the file it read (no row content)."""

    file_format: FileFormat | None = None
    rows: int = 0
    errors: int = 0
    # Columns datasets JSON-encodes from some record on (a later JSON batch mixed value types in
    # a column already locked as a non-Json type): (dotted path, records of the file before it).
    json_text_columns: list[tuple[str, int]] = field(default_factory=list)


class FileSource:
    """A data file to read: a local path or an opener for a remote stream, plus a byte budget."""

    def __init__(
        self,
        name: str,
        *,
        path: Path | None = None,
        opener: Callable[[], IO[bytes]] | None = None,
        size: int | None = None,
        byte_budget: int | None = None,
        budget_limit: str = "max_metadata_bytes",
    ) -> None:
        if path is None and opener is None:
            raise ValueError("FileSource needs a path or an opener")
        self.name = name  # display-safe shard id (relative path)
        self.path = path
        self._opener = opener
        self.size = size
        self.byte_budget = byte_budget  # shared by every stream opened from this source
        self.budget_limit = budget_limit
        self._counter = _Counter()

    @property
    def bytes_read(self) -> int:
        """Raw bytes read through budgeted streams so far (all opens together)."""
        return self._counter.value

    def open_raw(self) -> IO[bytes]:
        """A fresh binary stream positioned at 0, counted against the byte budget."""
        raw: IO[bytes] = self.path.open("rb") if self.path is not None else self._opener()  # type: ignore[misc]
        if self.byte_budget is None:
            return raw
        limited = _LimitedStream(raw, self.byte_budget, self.budget_limit, self._counter)
        return io.BufferedReader(limited)


@dataclass
class _Counter:
    value: int = 0


class _LimitedStream(io.RawIOBase):
    """Passes at most `cap` bytes through; asking for data beyond that raises `QuotaExceeded`.

    Reads are clipped at the cap (no read-ahead past it), so a file of exactly `cap` bytes is
    read completely and only a longer one fails.
    """

    def __init__(
        self, raw: IO[bytes], cap: int, limit: str, counter: _Counter | None = None
    ) -> None:
        super().__init__()
        self._raw = raw
        self._cap = cap
        self._limit = limit
        self._counter = counter if counter is not None else _Counter()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return bool(getattr(self._raw, "seekable", lambda: False)())

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        remaining = self._cap - self._counter.value
        if remaining <= 0:
            if self._raw.read(1):
                raise QuotaExceeded(self._limit, self._cap)
            return 0
        data = self._raw.read(min(len(view), remaining))
        n = len(data)
        self._counter.value += n
        view[:n] = data
        return n

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


_UNDECIDED = "?"


def compression_by_extension(name: str) -> str | None:
    """datasets' `_get_extraction_protocol` decision from the file name alone.

    Returns None when the extension says "not compressed" (json, jsonl, csv, parquet, arrow, ...),
    the protocol for a compression extension (gz, bz2, xz, zst, lz4, zip), or `_UNDECIDED` when
    the magic number has to decide.
    """
    from datasets.utils.file_utils import (
        BASE_KNOWN_EXTENSIONS,
        COMPRESSION_EXTENSION_TO_PROTOCOL,
        _get_path_extension,
    )

    extension = _get_path_extension(name)
    if (
        extension in BASE_KNOWN_EXTENSIONS
        or extension in ("tgz", "tar")
        or name.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    ):
        return None
    if extension in COMPRESSION_EXTENSION_TO_PROTOCOL:
        return COMPRESSION_EXTENSION_TO_PROTOCOL[extension]
    return _UNDECIDED


def detect_compression(name: str, head: bytes) -> str | None:
    """Compression protocol of a data file, decided like datasets' `_get_extraction_protocol`."""
    from datasets.utils.file_utils import (
        MAGIC_NUMBER_MAX_LENGTH,
        MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL,
        MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL,
    )

    by_name = compression_by_extension(name)
    if by_name != _UNDECIDED:
        return by_name
    magic = head[:MAGIC_NUMBER_MAX_LENGTH]
    for i in range(MAGIC_NUMBER_MAX_LENGTH):
        prefix = magic[: MAGIC_NUMBER_MAX_LENGTH - i]
        if prefix in MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL:
            return MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL[prefix]
        if prefix in MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL:
            return MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL[prefix]
    return None


def compression_of(source: FileSource) -> str | None:
    """`detect_compression` that reads the magic number only when the name is not conclusive."""
    by_name = compression_by_extension(source.name)
    if by_name != _UNDECIDED:
        return by_name
    raw = source.open_raw()
    try:
        head = raw.read(16)
    finally:
        raw.close()
    return detect_compression(source.name, head)


def open_decoded(source: FileSource, compression: str | None, limits: ReaderLimits) -> IO[bytes]:
    """Open a text-format file as a buffered byte stream, decompressing under a byte quota."""
    raw = source.open_raw()
    if compression is None:
        return raw if isinstance(raw, io.BufferedIOBase) else io.BufferedReader(cast(Any, raw))
    if compression not in STREAM_COMPRESSIONS:
        raw.close()
        raise UnsupportedFormat("compression_not_supported", compression=compression)
    decoded: Any
    if compression == "gzip":
        decoded = gzip.GzipFile(fileobj=raw, mode="rb")
    elif compression == "bz2":
        decoded = bz2.BZ2File(raw, "rb")
    elif compression == "xz":
        decoded = lzma.LZMAFile(raw, "rb")  # noqa: SIM115 - returned, closed by the caller
    else:
        decoded = _ArrowCodecStream(raw, compression)
    cap = limits.max_decompressed_bytes
    limit = "max_decompressed_bytes"
    if source.byte_budget is not None and source.byte_budget < cap:
        cap, limit = source.byte_budget, source.budget_limit
    return io.BufferedReader(_LimitedStream(decoded, cap, limit), buffer_size=1 * MiB)


class _ArrowCodecStream(io.RawIOBase):
    """zstd / lz4-frame decoding through pyarrow's codecs (no extra dependency)."""

    def __init__(self, raw: IO[bytes], codec: str) -> None:
        import pyarrow as pa

        super().__init__()
        self._raw = raw
        self._stream = pa.CompressedInputStream(pa.PythonFile(raw, mode="r"), codec)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        data = self._stream.read(len(view))
        n = len(data)
        view[:n] = data
        return n

    def close(self) -> None:
        try:
            self._stream.close()
            self._raw.close()
        finally:
            super().close()


def table_rows(table: pa.Table, decode_paths: list[list[Any]]) -> Iterator[dict[str, Any]]:
    """Python rows of a table in order, Json fields decoded like the datasets formatter."""
    from datasets.utils.json import json_decode_field

    for offset in range(0, table.num_rows, ROW_SLICE):
        for row in table.slice(offset, ROW_SLICE).to_pylist():
            for path in decode_paths:
                row = json_decode_field(row, path)
            yield row


def iter_file(
    source: FileSource,
    spec: FileSpec,
    limits: ReaderLimits,
    schema: SchemaState,
    info: ReadInfo,
) -> Iterator[ReadItem]:
    """Rows (dicts) and `RowError`s of one data file, in file order."""
    if spec.module not in SUPPORTED_MODULES:
        raise UnsupportedFormat("module_not_supported", module=spec.module)
    if source.size is not None and source.size > limits.max_file_bytes:
        raise QuotaExceeded("max_file_bytes", limits.max_file_bytes)
    if spec.module == "json":
        from .readers_json import iter_json_file

        yield from iter_json_file(source, spec, limits, schema, info)
    elif spec.module == "csv":
        from .readers_tabular import iter_csv_file

        yield from iter_csv_file(source, spec, limits, schema, info)
    elif spec.module == "parquet":
        from .readers_tabular import iter_parquet_file

        yield from iter_parquet_file(source, spec, limits, schema, info)
    else:
        from .readers_tabular import iter_arrow_file

        yield from iter_arrow_file(source, spec, limits, schema, info)
