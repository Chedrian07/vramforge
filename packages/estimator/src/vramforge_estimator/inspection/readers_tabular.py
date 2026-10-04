"""CSV, Parquet and Arrow readers mirroring datasets==5.0.1 `packaged_modules/{csv,parquet,arrow}`.

- CSV: `pandas.read_csv` with the builder's exact keyword arguments (`CsvConfig.pd_read_csv_kwargs`)
  in 10,000-row chunks, each chunk `pa.Table.from_pandas`; the first chunk locks the schema like
  datasets' ArrowWriter. What pandas may buffer is bounded before it builds a chunk: no physical
  line beyond `max_row_bytes` (a record is at least as long as each of its lines) and no chunk
  beyond `max_batch_bytes` read.
- Parquet: one row group at a time, cast to the schema of the split's first file.
- Arrow: IPC stream, falling back to the IPC file format, one record batch at a time.

Columnar formats inside a compression wrapper and builder options that drop rows (Parquet
`filters`, `on_bad_files="skip"`) are rejected as unsupported rather than approximated.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from dataclasses import fields as dataclass_fields
from typing import IO, TYPE_CHECKING, Any

from .readers import (
    FileSource,
    FileSpec,
    QuotaExceeded,
    ReaderLimits,
    ReadInfo,
    ReadItem,
    SchemaState,
    ShardBroken,
    UnsupportedFormat,
    open_decoded,
    table_rows,
)

if TYPE_CHECKING:
    import pyarrow as pa


def iter_csv_file(
    source: FileSource,
    spec: FileSpec,
    limits: ReaderLimits,
    schema: SchemaState,
    info: ReadInfo,
) -> Iterator[ReadItem]:
    import pandas as pd
    import pyarrow as pa
    from datasets.features.features import require_storage_cast
    from datasets.packaged_modules.csv.csv import CsvConfig
    from datasets.table import table_cast

    config = CsvConfig(**_known_options(CsvConfig, spec.options))
    if schema.features is None and config.features is not None:
        schema.features = config.features
    features = config.features
    dtype = None
    if features is not None:  # datasets Csv._generate_tables: read typed columns as such
        arrow_schema = features.arrow_schema
        dtype = {
            name: arrow_type.to_pandas_dtype() if not require_storage_cast(feature) else object
            for name, arrow_type, feature in zip(
                arrow_schema.names, arrow_schema.types, features.values(), strict=True
            )
        }
    info.file_format = "csv"
    stream = open_decoded(source, spec.compression, limits)
    # A custom record terminator makes physical lines meaningless as a bound.
    guard = _CsvGuard(
        stream,
        max_line=None if config.lineterminator else limits.max_row_bytes,
        max_chunk=limits.max_batch_bytes,
    )
    try:
        try:
            reader = pd.read_csv(
                io.BufferedReader(guard), iterator=True, dtype=dtype, **config.pd_read_csv_kwargs
            )
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ShardBroken("csv_unreadable") from exc
        while True:
            guard.start_chunk()
            try:
                df = next(reader)
            except StopIteration:
                break
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                raise ShardBroken("csv_unreadable", rows_read=info.rows) from exc
            try:
                table = pa.Table.from_pandas(df)
                if features is not None:
                    table = table_cast(table, features.arrow_schema)
                table = schema.lock(table)
            except (
                pa.ArrowInvalid,
                pa.ArrowTypeError,
                pa.ArrowNotImplementedError,
                ValueError,
            ) as exc:
                raise ShardBroken("schema_mismatch", rows_read=info.rows) from exc
            yield from _emit(table, limits, schema, info)
    finally:
        stream.close()


class _CsvGuard(io.RawIOBase):
    """Byte stream for pandas that raises `QuotaExceeded` before an oversized record or chunk is
    buffered (module docstring). Lines inside one read block are bounded by the block size, so
    only the line still open across blocks is tracked."""

    def __init__(self, raw: IO[bytes], *, max_line: int | None, max_chunk: int) -> None:
        super().__init__()
        self._raw = raw
        self._max_line = max_line
        self._max_chunk = max_chunk
        self._line = 0  # bytes of the line still open after the last block
        self._chunk = 0  # bytes read since `start_chunk`

    def readable(self) -> bool:
        return True

    def start_chunk(self) -> None:
        self._chunk = 0

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        data = self._raw.read(len(view))
        n = len(data)
        view[:n] = data
        self._chunk += n
        if self._chunk > self._max_chunk:
            raise QuotaExceeded("max_batch_bytes", self._max_chunk)
        if self._max_line is not None and n:
            breaks = [i for i in (data.find(b"\n"), data.find(b"\r")) if i >= 0]
            if breaks:
                if self._line + min(breaks) > self._max_line:
                    raise QuotaExceeded("max_row_bytes", self._max_line)
                self._line = n - max(data.rfind(b"\n"), data.rfind(b"\r")) - 1
            else:
                self._line += n
            if self._line > self._max_line:
                raise QuotaExceeded("max_row_bytes", self._max_line)
        return n

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def iter_parquet_file(
    source: FileSource,
    spec: FileSpec,
    limits: ReaderLimits,
    schema: SchemaState,
    info: ReadInfo,
) -> Iterator[ReadItem]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features

    _reject_compression(spec)
    options = spec.options
    if options.get("filters") is not None:
        raise UnsupportedFormat("parquet_filters")
    if options.get("on_bad_files", "error") != "error":
        raise UnsupportedFormat("parquet_skips_bad_files")
    columns = options.get("columns")
    info.file_format = "parquet"
    handle = _columnar_handle(source)
    try:
        try:
            parquet = pq.ParquetFile(handle)
        except (pa.ArrowInvalid, OSError, ValueError) as exc:
            raise ShardBroken("parquet_unreadable") from exc
        if schema.features is None:
            # datasets Parquet._split_generators: features from the split's first file schema.
            schema.features = options.get("features") or Features.from_arrow_schema(
                parquet.schema_arrow
            )
        if columns is not None and set(columns) != set(schema.features):
            # also applied to README features, like datasets
            schema.features = Features(
                {name: ftype for name, ftype in schema.features.items() if name in columns}
            )
        target = schema.features.arrow_schema
        metadata = parquet.metadata
        for group in range(parquet.num_row_groups):
            group_meta = metadata.row_group(group)
            if group_meta.total_byte_size > limits.max_batch_bytes:
                raise QuotaExceeded("max_batch_bytes", limits.max_batch_bytes, row_group=group)
            batch_size = max(1, min(group_meta.num_rows, 8192))
            batches = parquet.iter_batches(
                batch_size=batch_size, row_groups=[group], columns=columns, use_threads=True
            )
            while True:
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                except (pa.ArrowInvalid, OSError, ValueError) as exc:
                    raise ShardBroken("parquet_unreadable", row_group=group) from exc
                table = _cast_or_broken(pa.Table.from_batches([batch]), target, info)
                yield from _emit(table, limits, schema, info)
    finally:
        _close(handle)


def iter_arrow_file(
    source: FileSource,
    spec: FileSpec,
    limits: ReaderLimits,
    schema: SchemaState,
    info: ReadInfo,
) -> Iterator[ReadItem]:
    import pyarrow as pa
    from datasets import Features

    _reject_compression(spec)
    handle = _columnar_handle(source)
    try:
        batches: Iterator[pa.RecordBatch]
        try:
            stream_reader = pa.ipc.open_stream(handle)
            arrow_schema = stream_reader.schema
            batches = iter(stream_reader)
            info.file_format = "arrow_stream"
        except (OSError, pa.ArrowInvalid):
            try:
                file_reader = pa.ipc.open_file(handle)
            except (OSError, pa.ArrowInvalid) as exc:
                raise ShardBroken("arrow_unreadable") from exc
            arrow_schema = file_reader.schema
            batches = (file_reader.get_batch(i) for i in range(file_reader.num_record_batches))
            info.file_format = "arrow_file"
        if schema.features is None:
            schema.features = spec.options.get("features") or Features.from_arrow_schema(
                arrow_schema
            )
        target = schema.features.arrow_schema
        while True:
            try:
                batch = next(batches)
                if batch.nbytes > limits.max_batch_bytes:
                    raise QuotaExceeded("max_batch_bytes", limits.max_batch_bytes)
                batch.validate(full=True)
            except StopIteration:
                break
            except (pa.ArrowInvalid, OSError, ValueError) as exc:
                raise ShardBroken("arrow_unreadable", rows_read=info.rows) from exc
            table = _cast_or_broken(pa.Table.from_batches([batch]), target, info)
            yield from _emit(table, limits, schema, info)
    finally:
        _close(handle)


def _cast_or_broken(table: pa.Table, target: pa.Schema, info: ReadInfo) -> pa.Table:
    """Cast to the split schema; a file whose columns differ cannot be read like datasets."""
    import pyarrow as pa
    from datasets.table import CastError, table_cast

    try:
        return table_cast(table, target)
    except (
        CastError,
        pa.ArrowInvalid,
        pa.ArrowTypeError,
        pa.ArrowNotImplementedError,
        ValueError,
    ) as exc:
        raise ShardBroken("schema_mismatch", rows_read=info.rows) from exc


def _emit(
    table: pa.Table, limits: ReaderLimits, schema: SchemaState, info: ReadInfo
) -> Iterator[ReadItem]:
    _check_row_bytes(table, limits.max_row_bytes)
    schema.started = True
    info.rows += table.num_rows
    yield from table_rows(table, schema.decode_paths())


def _check_row_bytes(table: pa.Table, max_row: int) -> None:
    """Per-row quota over the top-level string/binary columns (computed in Arrow)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if table.num_rows == 0:
        return
    total = None
    for column in table.columns:
        column_type = column.type
        if not (
            pa.types.is_string(column_type)
            or pa.types.is_large_string(column_type)
            or pa.types.is_binary(column_type)
            or pa.types.is_large_binary(column_type)
        ):
            continue
        lengths = pc.fill_null(pc.cast(pc.binary_length(column), pa.int64()), 0)
        total = lengths if total is None else pc.add(total, lengths)
    if total is None:
        return
    largest = pc.max(total).as_py()
    if largest is not None and largest > max_row:
        raise QuotaExceeded("max_row_bytes", max_row)


def _reject_compression(spec: FileSpec) -> None:
    if spec.compression is not None:
        raise UnsupportedFormat("compressed_columnar", compression=spec.compression)


def _columnar_handle(source: FileSource) -> Any:
    """A local path is memory-mapped (fast, no budget); otherwise the counted raw stream."""
    import pyarrow as pa

    if source.path is not None and source.byte_budget is None:
        return pa.memory_map(str(source.path), "r")
    return source.open_raw()


def _close(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if close is not None:
        close()


def _known_options(config_cls: type, options: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclass_fields(config_cls)}
    return {key: value for key, value in options.items() if key in names}
