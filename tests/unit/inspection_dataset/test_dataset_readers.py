"""Format readers: parity with datasets.load_dataset, failure records, quotas and broken files."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dataset_testkit import FIXTURES, write_jsonl

from vramforge_estimator.inspection.readers import (
    FileSource,
    FileSpec,
    QuotaExceeded,
    ReaderLimits,
    ReadInfo,
    RowError,
    SchemaState,
    ShardBroken,
    UnsupportedFormat,
    compression_of,
    iter_file,
)

Oracle = Callable[..., dict[str, list[dict[str, Any]]]]


def read_all(
    path: Path,
    module: str,
    *,
    options: dict[str, Any] | None = None,
    limits: ReaderLimits | None = None,
    budget: int | None = None,
) -> tuple[list[Any], ReadInfo]:
    source = FileSource(path.name, path=path, size=path.stat().st_size, byte_budget=budget)
    spec = FileSpec(module, options or {}, compression_of(source))  # type: ignore[arg-type]
    info = ReadInfo()
    items = list(iter_file(source, spec, limits or ReaderLimits(), SchemaState(), info))
    return items, info


@pytest.mark.parametrize(
    ("name", "module", "file_format"),
    [
        ("preference.jsonl", "json", "json_lines"),
        ("messages.jsonl", "json", "json_lines"),
        ("json_lines.json", "json", "json_lines"),
        ("array.json", "json", "json_array"),
        ("text.csv", "csv", "csv"),
        ("prompt_completion.parquet", "parquet", "parquet"),
    ],
)
def test_rows_equal_datasets(
    name: str, module: str, file_format: str, datasets_rows: Oracle
) -> None:
    path = FIXTURES / name
    rows, info = read_all(path, module)
    assert info.file_format == file_format
    assert rows == datasets_rows(path, module)["train"]


def test_unicode_and_code_round_trip() -> None:
    path = FIXTURES / "preference.jsonl"
    rows, _ = read_all(path, "json")
    expected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == expected
    assert any("문자열" in row["question"] for row in rows)
    assert any("́" in row["question"] for row in rows)  # combining mark kept, not normalized
    assert any("💬" in row["question"] for row in rows)
    assert all(row["chosen"].startswith("```") for row in rows)


def test_heterogeneous_messages_keep_their_keys(datasets_rows: Oracle) -> None:
    path = FIXTURES / "messages.jsonl"
    rows, _ = read_all(path, "json")
    tool_turn = rows[2]["messages"][1]
    assert tool_turn["tool_calls"] == [{"name": "weather", "arguments": {"city": "Seoul"}}]
    assert "tool_calls" not in rows[0]["messages"][0]  # no None-filled keys (datasets Json type)
    assert rows == datasets_rows(path, "json")["train"]


def test_malformed_jsonl_lines_are_row_failures() -> None:
    items, info = read_all(FIXTURES / "malformed.jsonl", "json")
    assert [type(item).__name__ for item in items] == [
        "dict",
        "dict",
        "RowError",
        "dict",
        "dict",
    ]
    error = items[2]
    assert isinstance(error, RowError)
    assert (error.reason, error.line) == ("json_parse", 3)
    assert items[3] == {"prompt": "fourth", "completion": "ok 4"}
    assert items[4]["prompt"] == "다섯째"
    assert (info.rows, info.errors) == (4, 1)


def test_non_object_records_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(b'{"a": 1}\n\n   \n[1, 2]\n"text"\n{"a": 2}\n')
    items, info = read_all(path, "json")
    assert items[0] == {"a": 1}
    assert [(e.reason, e.line) for e in items if isinstance(e, RowError)] == [
        ("not_an_object", 4),
        ("not_an_object", 5),
    ]
    assert items[-1] == {"a": 2}
    assert info.rows == 2


def test_bom_crlf_and_missing_final_newline_match_datasets(
    tmp_path: Path, datasets_rows: Oracle
) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"a": "x", "n": 1}\r\n\r\n{"a": "y", "n": 2.5}\r\n{"a": "z"}')
    rows, _ = read_all(path, "json")
    assert rows == datasets_rows(path, "json")["train"]
    assert rows[2] == {"a": "z", "n": None}  # missing key filled like datasets


def test_json_array_of_strings_becomes_text_column(tmp_path: Path, datasets_rows: Oracle) -> None:
    path = tmp_path / "strings.json"
    path.write_text(json.dumps(["첫 문장", "second"]), encoding="utf-8")
    rows, info = read_all(path, "json")
    assert info.file_format == "json_array"
    assert rows == [{"text": "첫 문장"}, {"text": "second"}]
    assert rows == datasets_rows(path, "json")["train"]


def test_large_json_array_is_read_incrementally_with_same_rows(tmp_path: Path) -> None:
    items = [{"q": f"질문 {i}", "a": "x" * (i % 7), "n": i} for i in range(300)]
    path = tmp_path / "big.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    one_load, _ = read_all(path, "json")
    incremental, info = read_all(path, "json", limits=ReaderLimits(json_array_max_bytes=512))
    assert info.file_format == "json_array"
    assert incremental == one_load == items


def test_json_array_with_a_bad_element_reports_it_by_position(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"a": 1}, {"a": tru}, {"a": "x,]{"}, 5]', encoding="utf-8")
    for limits in (ReaderLimits(), ReaderLimits(json_array_max_bytes=8)):
        items, _ = read_all(path, "json", limits=limits)
        assert items[0] == {"a": 1}
        assert items[2] == {"a": "x,]{"}
        assert [(e.reason, e.element) for e in items if isinstance(e, RowError)] == [
            ("json_parse", 1),
            ("not_an_object", 3),
        ]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('[{"a": 1}, {"a": 2}', "json_array_unterminated"),
        ('[{"a": 1}] trailing', "json_trailing_data"),
    ],
)
def test_structurally_broken_json_array_stops_the_file(
    tmp_path: Path, content: str, reason: str
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ShardBroken) as excinfo:
        read_all(path, "json", limits=ReaderLimits(json_array_max_bytes=4))
    assert excinfo.value.reason == reason


def test_binary_content_is_not_parsed_line_by_line(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_bytes(gzip.compress(b'{"a": 1}\n' * 100))  # gzip bytes behind a .jsonl name
    with pytest.raises(ShardBroken) as excinfo:
        read_all(path, "json")
    assert excinfo.value.reason == "binary_content"


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
def test_compressed_jsonl_equals_datasets(
    tmp_path: Path, suffix: str, compress: Callable[[bytes], bytes], datasets_rows: Oracle
) -> None:
    raw = (FIXTURES / "preference.jsonl").read_bytes()
    path = tmp_path / f"data.jsonl{suffix}"
    path.write_bytes(compress(raw))
    rows, info = read_all(path, "json")
    assert info.file_format == "json_lines"
    assert rows == datasets_rows(path, "json")["train"]


def test_zstd_jsonl_is_decoded(tmp_path: Path) -> None:
    raw = (FIXTURES / "preference.jsonl").read_bytes()
    sink = pa.BufferOutputStream()
    with pa.CompressedOutputStream(sink, "zstd") as out:
        out.write(raw)
    path = tmp_path / "data.jsonl.zst"
    path.write_bytes(sink.getvalue().to_pybytes())
    rows, _ = read_all(path, "json")
    assert rows == read_all(FIXTURES / "preference.jsonl", "json")[0]


def test_compression_bomb_stops_at_the_decompressed_quota(tmp_path: Path) -> None:
    path = tmp_path / "bomb.jsonl.gz"
    line = b'{"a": "' + b"0" * 1000 + b'"}\n'
    path.write_bytes(gzip.compress(line * 20_000))  # ~20 MB decompressed, ~30 KB on disk
    with pytest.raises(QuotaExceeded) as excinfo:
        read_all(path, "json", limits=ReaderLimits(max_decompressed_bytes=1 << 20))
    assert excinfo.value.limit == "max_decompressed_bytes"


def test_row_larger_than_the_row_quota_stops_reading(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "long.jsonl", [{"a": "short"}, {"a": "x" * 5000}, {"a": "z"}])
    with pytest.raises(QuotaExceeded) as excinfo:
        read_all(path, "json", limits=ReaderLimits(max_row_bytes=1000))
    assert excinfo.value.limit == "max_row_bytes"
    assert excinfo.value.detail["line"] == 2


def test_csv_row_quota(tmp_path: Path) -> None:
    path = tmp_path / "long.csv"
    path.write_text("text\nshort\n" + "y" * 5000 + "\n", encoding="utf-8")
    with pytest.raises(QuotaExceeded) as excinfo:
        read_all(path, "csv", limits=ReaderLimits(max_row_bytes=1000))
    assert excinfo.value.limit == "max_row_bytes"


def test_preview_budget_bounds_the_bytes_read(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "many.jsonl", [{"i": i, "pad": "p" * 100} for i in range(5000)])
    with pytest.raises(QuotaExceeded) as excinfo:
        read_all(path, "json", budget=200_000)
    assert excinfo.value.limit == "max_metadata_bytes"


def test_parquet_row_group_quota(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pq.write_table(pa.table({"text": ["x" * 2000] * 10}), path)
    with pytest.raises(QuotaExceeded) as excinfo:
        read_all(path, "parquet", limits=ReaderLimits(max_batch_bytes=1000))
    assert excinfo.value.limit == "max_batch_bytes"


def test_parquet_reads_every_row_group_in_order() -> None:
    path = FIXTURES / "prompt_completion.parquet"
    assert pq.ParquetFile(path).num_row_groups == 2
    rows, info = read_all(path, "parquet")
    assert [row["prompt"] for row in rows] == pq.read_table(path).column("prompt").to_pylist()
    assert info.rows == 5


@pytest.mark.parametrize("kind", ["stream", "file"])
def test_arrow_ipc_equals_datasets(tmp_path: Path, kind: str, datasets_rows: Oracle) -> None:
    table = pq.read_table(FIXTURES / "prompt_completion.parquet")
    path = tmp_path / f"data-{kind}.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        writer = pa.ipc.new_stream if kind == "stream" else pa.ipc.new_file
        with writer(sink, table.schema) as out:
            out.write_table(table, max_chunksize=2)
    rows, info = read_all(path, "arrow")
    assert info.file_format == f"arrow_{kind}"
    assert rows == datasets_rows(path, "arrow")["train"]


@pytest.mark.parametrize(
    ("module", "content", "error"),
    [
        ("parquet", b"PAR1" + b"\x00" * 64, ShardBroken),
        ("arrow", b"not arrow at all", ShardBroken),
    ],
)
def test_broken_columnar_files(tmp_path: Path, module: str, content: bytes, error: type) -> None:
    path = tmp_path / f"broken.{module}"
    path.write_bytes(content)
    with pytest.raises(error):
        read_all(path, module)


def test_compressed_columnar_files_are_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet.gz"
    path.write_bytes(gzip.compress((FIXTURES / "prompt_completion.parquet").read_bytes()))
    with pytest.raises(UnsupportedFormat):
        read_all(path, "parquet")


def test_csv_quotes_newlines_and_na_values_match_datasets(datasets_rows: Oracle) -> None:
    rows, _ = read_all(FIXTURES / "text.csv", "csv")
    assert rows[1]["text"] == 'Multi\nline text with "quotes"'
    assert rows[3]["text"] is None  # pandas default NA handling, exactly like datasets
    assert rows == datasets_rows(FIXTURES / "text.csv", "csv")["train"]


def test_tsv_separator_option(tmp_path: Path, datasets_rows: Oracle) -> None:
    path = tmp_path / "data.tsv"
    path.write_text("prompt\tcompletion\nhi\tthere\n안녕\t하세요\n", encoding="utf-8")
    rows, _ = read_all(path, "csv", options={"sep": "\t"})
    assert rows == [
        {"prompt": "hi", "completion": "there"},
        {"prompt": "안녕", "completion": "하세요"},
    ]
    assert rows == datasets_rows(path, "csv", sep="\t")["train"]


def test_byte_budget_is_exact(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "rows.jsonl", [{"i": i} for i in range(50)])
    size = path.stat().st_size
    rows, _ = read_all(path, "json", budget=size)  # exactly the file size: read completely
    assert len(rows) == 50
    with pytest.raises(QuotaExceeded):
        read_all(path, "json", budget=size - 1)


def test_agent_trace_shaped_json_lines_are_unsupported(tmp_path: Path) -> None:
    # datasets 5.0.1 routes these columns to its agent-trace converter (`teich`), not the rows.
    row = {"id": "1", "source": "s", "model": "m", "system_prompt": "x", "messages": []}
    path = write_jsonl(tmp_path / "traces.jsonl", [row])
    with pytest.raises(UnsupportedFormat) as excinfo:
        read_all(path, "json")
    assert excinfo.value.reason == "agent_traces"
    assert read_all(path, "json", options={"parse_agent_traces": False})[0] == [row]


def test_deprecated_block_size_overrides_chunksize() -> None:
    from vramforge_estimator.inspection.readers_json import JsonOptions

    assert JsonOptions.from_options({"block_size": 4096, "chunksize": 1 << 20}).chunksize == 4096
    assert JsonOptions.from_options({"block_size": None, "chunksize": 1 << 20}).chunksize == 1 << 20


def test_json_array_after_leading_whitespace_is_unsupported_like_datasets(
    tmp_path: Path, datasets_rows: Oracle
) -> None:
    path = tmp_path / "pretty.json"
    path.write_text('\n  [\n {"a": 1},\n {"a": 2}\n]\n', encoding="utf-8")
    with pytest.raises(UnsupportedFormat) as excinfo:
        read_all(path, "json")
    assert excinfo.value.reason == "json_array_leading_whitespace"
    with pytest.raises(Exception):  # noqa: B017 - datasets raises its own generation error
        datasets_rows(path, "json")


def test_pretty_printed_objects_follow_pyarrow_like_datasets(
    tmp_path: Path, datasets_rows: Oracle
) -> None:
    path = tmp_path / "pretty.jsonl"
    path.write_text('{\n  "a": 1\n}\n{\n  "a": 2\n}\n', encoding="utf-8")
    rows, _ = read_all(path, "json")
    assert rows == [{"a": 1}, {"a": 2}] == datasets_rows(path, "json")["train"]
