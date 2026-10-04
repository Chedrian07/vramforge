"""JSON Lines / JSON array / JSON `field` reader mirroring datasets==5.0.1 `packaged_modules/json`.

The batch pipeline follows `Json._generate_tables` (JSON Lines branch): 10 MiB chunks completed to
a line end, mixed-type fields JSON-encoded (`on_mixed_types="use_json"`), `pyarrow.json.read_json`
with the same block-size retries, and the same `_cast_table`. A file whose first chunk starts with
`[` is a JSON array: small arrays are converted to JSON Lines in one load exactly like datasets,
larger ones are split into elements incrementally (bounded memory).

Differences, all limited to inputs datasets cannot load at all: in a chunk that fails to parse,
each record that is not a JSON object pyarrow can read (syntax error, non-object, invalid UTF-8,
duplicate key, number overflow) becomes a `RowError` (datasets falls back to pandas and then
raises) and the other records are parsed together as one batch; if they still fail, they conflict
with each other or with the split schema and the shard stops (`schema_mismatch`). Records are never
cast one by one, since a one-row cast can coerce a value the batch parse rejected (2.5 -> True).
Quotas from `ReaderLimits` stop reading with `QuotaExceeded`.

Known approximations (documented, not silent): the schema is locked by the first batch of the
selected split (datasets infers it from the first batch of the config's first split, which may be
another split), and an incrementally read JSON array infers it from its first batch of elements
instead of the whole array. Both only matter when value types change across those boundaries.
"""

from __future__ import annotations

import codecs
import io
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

from .readers import (
    FileSource,
    FileSpec,
    QuotaExceeded,
    ReaderLimits,
    ReadInfo,
    ReadItem,
    RowError,
    SchemaState,
    ShardBroken,
    UnsupportedFormat,
    open_decoded,
    table_rows,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from datasets import Features

DEFAULT_CHUNKSIZE = 10 << 20  # datasets JsonConfig.chunksize
_ARRAY_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|"|[\[\]{},]', re.S)
_WHITESPACE = b" \t\r\n"


@dataclass(frozen=True)
class JsonOptions:
    """The JsonConfig fields that change what is read (README `configs` may set them)."""

    field: str | None = None
    encoding: str = "utf-8"
    encoding_errors: str | None = None
    chunksize: int = DEFAULT_CHUNKSIZE
    on_mixed_types: str | None = "use_json"
    features: Features | None = None
    parse_agent_traces: bool = True

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> JsonOptions:
        # datasets Json._info: a deprecated `block_size` overrides `chunksize`.
        chunksize = options.get("block_size") or options.get("chunksize") or DEFAULT_CHUNKSIZE
        return cls(
            field=options.get("field"),
            encoding=options.get("encoding") or "utf-8",
            encoding_errors=options.get("encoding_errors"),
            chunksize=int(chunksize),
            on_mixed_types=options.get("on_mixed_types", "use_json"),
            features=options.get("features"),
            parse_agent_traces=bool(options.get("parse_agent_traces", True)),
        )


class _BatchFailed(Exception):
    """The batch cannot be parsed as a whole; it is re-read record by record."""


def iter_json_file(
    source: FileSource,
    spec: FileSpec,
    limits: ReaderLimits,
    schema: SchemaState,
    info: ReadInfo,
) -> Iterator[ReadItem]:
    opts = JsonOptions.from_options(spec.options)
    if schema.features is None and opts.features is not None:
        schema.features = opts.features
        schema.json_field_paths = schema.decode_paths()
    chunksize = opts.chunksize
    if source.byte_budget is not None:
        chunksize = min(chunksize, max(source.byte_budget // 4, 1 << 10))
    stream = open_decoded(source, spec.compression, limits)
    try:
        head = stream.read(chunksize)
        if head.startswith(codecs.BOM_UTF8):
            head = head[len(codecs.BOM_UTF8) :]
        if opts.encoding.replace("_", "-").lower() in ("utf-8", "utf8") and b"\x00" in head[:8192]:
            raise ShardBroken("binary_content")  # JSON text never contains a raw NUL byte
        if not head.startswith(b"[") and head.lstrip(_WHITESPACE).startswith(b"["):
            # datasets only detects arrays at byte 0; otherwise pyarrow and pandas both fail.
            raise UnsupportedFormat("json_array_leading_whitespace")
        reader = _JsonReader(stream, opts, limits, schema, info, chunksize, source.byte_budget)
        if opts.field is not None:
            info.file_format = "json_field"
            yield from reader.field_object(head)
        elif head.startswith(b"["):
            info.file_format = "json_array"
            yield from reader.array(head)
        else:
            info.file_format = "json_lines"
            yield from reader.lines(head)
    finally:
        stream.close()


class _JsonReader:
    def __init__(
        self,
        stream: IO[bytes],
        opts: JsonOptions,
        limits: ReaderLimits,
        schema: SchemaState,
        info: ReadInfo,
        chunksize: int,
        budget: int | None,
    ) -> None:
        self.stream = stream
        self.budget = budget  # preview byte budget (None for full reads)
        self.opts = opts
        self.limits = limits
        self.schema = schema
        self.info = info
        self.chunksize = chunksize
        # datasets: block_size = max(chunksize // 32, 16 << 10), grown per file on retries.
        self.block_size = max(opts.chunksize // 32, 16 << 10)

    # ------------------------------------------------------------------ JSON Lines

    def lines(self, head: bytes) -> Iterator[ReadItem]:
        batch = head
        line_no = 1
        max_row = self.limits.max_row_bytes
        while batch:
            partial = len(batch) - (batch.rfind(b"\n") + 1)
            allowance = max_row + 1 - partial
            if allowance <= 0:
                raise QuotaExceeded("max_row_bytes", max_row, line=line_no + batch.count(b"\n"))
            tail = self.stream.readline(allowance)
            if tail and not tail.endswith(b"\n") and len(tail) >= allowance:
                raise QuotaExceeded("max_row_bytes", max_row, line=line_no + batch.count(b"\n"))
            batch += tail
            _check_line_lengths(batch, max_row, line_no)
            if self.opts.encoding != "utf-8":
                batch = self._to_utf8(batch)
            yield from self._batch_with_fallback(batch, line_no)
            line_no += batch.count(b"\n")
            batch = self.stream.read(self.chunksize)

    def _to_utf8(self, batch: bytes) -> bytes:
        try:
            return batch.decode(
                self.opts.encoding, errors=self.opts.encoding_errors or "strict"
            ).encode("utf-8")
        except (UnicodeDecodeError, LookupError) as exc:
            raise ShardBroken("encoding", encoding=self.opts.encoding) from exc

    def _batch_with_fallback(self, batch: bytes, line_no: int) -> Iterator[ReadItem]:
        try:
            table = self._parse(batch)
        except _BatchFailed:
            records = [
                (line_no + i, line) for i, line in enumerate(batch.split(b"\n")) if line.strip()
            ]
            yield from self._records(records, kind="line")
            return
        yield from self._emit(table)

    # ------------------------------------------------------------------ JSON array

    def array(self, head: bytes) -> Iterator[ReadItem]:
        limit = self.limits.json_array_max_bytes
        if self.budget is not None:
            limit = min(limit, self.budget // 2)
        rest = self.stream.read(max(limit + 1 - len(head), 0))
        full = head + rest
        # datasets: list of objects iff a "{" comes before the first '"' in the first 100 bytes
        objects = b"{" in head[:100].split(b'"', 1)[0]
        if len(full) <= limit:
            try:
                data = _ujson_loads(full)
            except ValueError:
                data = None
            if isinstance(data, list):
                yield from self._array_in_one_load(data, objects)
                return
        yield from self._array_incremental(full, objects)

    def _array_in_one_load(self, data: list[Any], objects: bool) -> Iterator[ReadItem]:
        from datasets.utils.json import ujson_dumps

        records = []
        for index, item in enumerate(data):
            line = (ujson_dumps(item) if objects else ujson_dumps({"text": item})).encode()
            if len(line) > self.limits.max_row_bytes:
                raise QuotaExceeded("max_row_bytes", self.limits.max_row_bytes, element=index)
            records.append((index, line))
        # datasets parses the whole converted array as one batch.
        batch = b"\n".join(line for _, line in records)
        try:
            table = self._parse(batch)
        except _BatchFailed:
            yield from self._records(records, kind="element")
            return
        yield from self._emit(table)

    def _array_incremental(self, initial: bytes, objects: bool) -> Iterator[ReadItem]:
        from datasets.utils.json import ujson_dumps

        pending: list[tuple[int, bytes]] = []
        pending_bytes = 0
        for index, element in enumerate(
            _iter_array_elements(self.stream, initial, self.limits.max_row_bytes)
        ):
            try:
                item = _ujson_loads(element)
            except ValueError:
                pending.append((index, b""))  # empty line -> decoded as a parse failure
                continue
            line = (ujson_dumps(item) if objects else ujson_dumps({"text": item})).encode()
            pending.append((index, line))
            pending_bytes += len(line) + 1
            if pending_bytes >= self.chunksize:
                yield from self._array_batch(pending)
                pending, pending_bytes = [], 0
        if pending:
            yield from self._array_batch(pending)

    def _array_batch(self, records: list[tuple[int, bytes]]) -> Iterator[ReadItem]:
        if all(line for _, line in records):
            try:
                table = self._parse(b"\n".join(line for _, line in records))
            except _BatchFailed:
                pass
            else:
                yield from self._emit(table)
                return
        yield from self._records(records, kind="element")

    # ------------------------------------------------------------------ `field` object

    def field_object(self, head: bytes) -> Iterator[ReadItem]:
        import pyarrow as pa
        from datasets.packaged_modules.json.json import pandas_read_json
        from datasets.utils.json import ujson_dumps

        limit = self.limits.json_array_max_bytes
        full = head + self.stream.read(max(limit + 1 - len(head), 0))
        if len(full) > limit:
            raise QuotaExceeded("json_array_max_bytes", limit)
        try:
            text = full.decode(self.opts.encoding, errors=self.opts.encoding_errors or "strict")
            dataset = _ujson_loads(text)[self.opts.field]
            df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, LookupError) as exc:
            raise ShardBroken("json_field_unreadable") from exc
        if df.columns.tolist() == [0]:
            df.columns = list(self.opts.features) if self.opts.features else ["text"]
        try:
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = self._cast(table, [])
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError) as exc:
            raise ShardBroken("schema_mismatch") from exc
        yield from self._emit(table)

    # ------------------------------------------------------------------ shared

    def _parse(self, batch: bytes) -> pa.Table:
        """`Json._generate_tables` batch handling: mixed types, pyarrow parse, `_cast_table`."""
        import pyarrow as pa
        import pyarrow.json as paj
        from datasets.utils.json import (
            find_mixed_struct_types_field_paths,
            get_json_field_path_from_pyarrow_json_error,
            insert_json_field_path,
            json_encode_field,
            json_encode_fields_in_json_lines,
            ujson_dumps,
        )

        schema = self.schema
        # Work on a copy: a batch that fails must not leave field paths behind for later batches.
        paths = [list(path) for path in schema.json_field_paths]
        use_json = self.opts.on_mixed_types == "use_json"
        if not schema.started and schema.features is None and use_json:
            try:
                examples = [_ujson_loads(line) for line in batch.splitlines()]
            except ValueError:
                pass  # datasets: "likely not JSON Lines"; the parse below decides
            else:
                paths += find_mixed_struct_types_field_paths(examples)
        original = batch
        if paths:
            try:
                examples = [_ujson_loads(line) for line in batch.splitlines()]
            except ValueError as exc:
                raise _BatchFailed from exc
            for path in paths:
                examples = [json_encode_field(example, path) for example in examples]
            batch = "\n".join(ujson_dumps(example) for example in examples).encode()
        if len(batch) // 8 > self.block_size:
            self.block_size = len(batch)
        while True:
            try:
                table = paj.read_json(
                    io.BytesIO(batch), read_options=paj.ReadOptions(block_size=self.block_size)
                )
                break
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                message = str(exc)
                if (
                    use_json
                    and isinstance(exc, pa.ArrowInvalid)
                    and "JSON parse error: Column(" in message
                    and ") changed from" in message
                ):
                    path = get_json_field_path_from_pyarrow_json_error(message)
                    if path in paths or path == [""]:  # root: a record that is not an object
                        raise _BatchFailed from exc
                    insert_json_field_path(paths, path)
                    try:
                        batch = json_encode_fields_in_json_lines(original, paths)
                    except ValueError as inner:
                        raise _BatchFailed from inner
                elif (
                    "straddling" in message or "JSON conversion to" in message
                ) and self.block_size < len(batch):
                    self.block_size *= 2
                else:
                    raise _BatchFailed from exc
        try:
            table = self._cast(table, paths)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError) as exc:
            raise _BatchFailed from exc
        schema.json_field_paths[:] = paths
        return table

    def _cast(self, table: pa.Table, paths: list[list[Any]]) -> pa.Table:
        """`Json._cast_table`, then the schema lock (datasets: `info.features` of the 1st table)."""
        import pandas as pd
        import pyarrow as pa
        from datasets import Features, Value
        from datasets.table import table_cast
        from datasets.utils.json import set_json_types_in_feature

        features = self.schema.features
        if features is not None:
            for name in set(features) - set(table.column_names):
                arrow_type = features.arrow_schema.field(name).type
                table = table.append_column(name, pa.array([None] * len(table), type=arrow_type))
            for i, name in enumerate(table.column_names):
                if pa.types.is_struct(table[name].type) and features.get(name) == Value("string"):
                    jsonl = (
                        table[name]
                        .to_pandas(types_mapper=pd.ArrowDtype)
                        .to_json(orient="records", lines=True)
                    )
                    strings = pa.array(
                        (
                            None if x.strip() == "null" else x.strip()
                            for x in jsonl.split("\n")
                            if x.strip()
                        ),
                        type=pa.string(),
                    )
                    table = table.set_column(i, name, strings)
            return table_cast(table, features.arrow_schema)
        if paths:
            json_features = set_json_types_in_feature(
                Features.from_arrow_schema(table.schema), paths
            )
            table = table_cast(table, json_features.arrow_schema)
        features = Features.from_arrow_schema(table.schema)
        if self.info.file_format == "json_lines" and self.opts.parse_agent_traces:
            from datasets.packaged_modules.json.json import has_agent_traces_markers

            if has_agent_traces_markers(features):
                # datasets would convert these rows with `teich` instead of reading them as is.
                raise UnsupportedFormat("agent_traces")
        self.schema.features = features
        return table

    def _emit(self, table: pa.Table) -> Iterator[ReadItem]:
        self.schema.started = True
        self.info.rows += table.num_rows
        yield from table_rows(table, self.schema.decode_paths())

    def _records(self, records: list[tuple[int, bytes]], *, kind: str) -> Iterator[ReadItem]:
        """Fallback for a batch that failed as a whole (module docstring): unreadable records
        become `RowError`s, the rest is parsed together or the shard stops."""
        errors = {
            position: _row_error(reason, position, kind)
            for position, line in records
            if (reason := _record_problem(line)) is not None
        }
        good = [(position, line) for position, line in records if position not in errors]
        rows: Iterator[dict[str, Any]] = iter(())
        if good:
            try:
                table = self._parse(b"\n".join(line for _, line in good))
            except _BatchFailed as exc:
                raise ShardBroken("schema_mismatch", **{kind: good[0][0]}) from exc
            if table.num_rows != len(good):
                raise ShardBroken("schema_mismatch", **{kind: good[0][0]})
            self.schema.started = True
            rows = table_rows(table, self.schema.decode_paths())
        for position, _ in records:
            if position in errors:
                self.info.errors += 1
                yield errors[position]
            else:
                self.info.rows += 1
                yield next(rows)


def _record_problem(line: bytes) -> str | None:
    """Why pyarrow cannot read this record as one JSON object (None = it can)."""
    if not line:
        return "json_parse"
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return "invalid_utf8"
    try:
        value = json.loads(text, object_pairs_hook=_unique_keys, parse_float=_finite_float)
    except ValueError:  # bad syntax, a duplicate key or a float pyarrow cannot store
        return "json_parse"
    return None if isinstance(value, dict) else "not_an_object"


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """pyarrow rejects an object that repeats a key ("Column(/a) was specified twice")."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result


def _finite_float(literal: str) -> float:
    """pyarrow rejects float literals beyond double range ("Number too big to be stored")."""
    value = float(literal)
    if math.isinf(value):
        raise ValueError("float overflow")
    return value


def _row_error(reason: str, position: int, kind: str) -> RowError:
    if kind == "line":
        return RowError(reason=reason, line=position)
    return RowError(reason=reason, element=position)


def _ujson_loads(data: bytes | str) -> Any:
    from datasets.utils.json import ujson_loads

    return ujson_loads(data)


def _check_line_lengths(batch: bytes, max_row: int, line_no: int) -> None:
    if len(batch) <= max_row:
        return
    for i, line in enumerate(batch.split(b"\n")):
        if len(line) > max_row:
            raise QuotaExceeded("max_row_bytes", max_row, line=line_no + i)


def _iter_array_elements(stream: IO[bytes], initial: bytes, max_element: int) -> Iterator[bytes]:
    """Split a top-level JSON array into raw element bytes without loading it whole.

    A tokenizer that only understands strings and brackets finds the commas at depth 0; element
    contents are validated by the caller. Memory: one element plus one read block.
    """
    buf = bytearray(initial)
    eof = False
    pos = 0
    while True:  # skip leading whitespace, expect "["
        while pos < len(buf) and buf[pos] in _WHITESPACE:
            pos += 1
        if pos < len(buf) or eof:
            break
        chunk = stream.read(1 << 20)
        eof = not chunk
        buf += chunk
    if pos >= len(buf) or buf[pos] != ord("["):
        raise ShardBroken("json_array_expected")
    pos += 1
    start = pos  # start of the current element
    depth = 0
    index = 0
    while True:
        match = _ARRAY_TOKEN.search(buf, pos)
        token = match.group() if match is not None else b""
        if match is None or token == b'"':
            if eof:
                raise ShardBroken("json_array_unterminated", element=index)
            if len(buf) - start > max_element:
                raise QuotaExceeded("max_row_bytes", max_element, element=index)
            pos = match.start() if match is not None else len(buf)
            del buf[:start]
            pos -= start
            start = 0
            chunk = stream.read(max(1 << 20, len(buf)))  # geometric growth: linear rescans
            eof = not chunk
            buf += chunk
            continue
        char = token[0]
        if char == ord('"'):
            pos = match.end()
            continue
        if char in b"[{":
            depth += 1
        elif char in b"]}":
            if depth == 0:
                element = bytes(buf[start : match.start()]).strip(_WHITESPACE)
                if element or index > 0:
                    yield element
                _expect_only_whitespace(stream, bytes(buf[match.end() :]))
                return
            depth -= 1
        elif char == ord(",") and depth == 0:
            yield bytes(buf[start : match.start()]).strip(_WHITESPACE)
            index += 1
            start = match.end()
        pos = match.end()


def _expect_only_whitespace(stream: IO[bytes], rest: bytes) -> None:
    while True:
        if rest.strip(_WHITESPACE):
            raise ShardBroken("json_trailing_data")
        rest = stream.read(1 << 20)
        if not rest:
            return
