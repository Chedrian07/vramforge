"""RowStream: datasets row order, split isolation, complete-flag semantics and failure accounting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dataset_testkit import FIXTURES, file_entries, hf_source, local_source, write_jsonl

from vramforge_estimator.inspection.base import SourceRow
from vramforge_estimator.inspection.dataset_files import SourceFiles
from vramforge_estimator.inspection.dataset_layout import resolve_layout
from vramforge_estimator.inspection.dataset_stream import (
    DatasetRowStream,
    FailedSourceRow,
    is_failed_row,
)
from vramforge_estimator.inspection.readers import ReaderLimits
from vramforge_estimator.schemas import ErrorCode, FileEntry
from vramforge_estimator.sources import ResolvedSource, SourceAccess

Oracle = Callable[..., dict[str, list[dict[str, Any]]]]


def stream_for(
    source: ResolvedSource,
    split: str = "train",
    config: str | None = None,
    limits: ReaderLimits | None = None,
) -> DatasetRowStream:
    limits = limits or ReaderLimits()
    files = SourceFiles(source, SourceAccess(), limits)
    layout = resolve_layout(files)
    chosen = layout.config(config) if config else layout.configs[0]
    assert chosen is not None
    split_layout = chosen.split(split)
    assert split_layout is not None
    return DatasetRowStream(files, chosen, split_layout, limits)


def sharded_dir(root: Path) -> Path:
    for shard, start in (("00000", 0), ("00001", 3), ("00002", 6)):
        rows = [{"prompt": f"p{i}", "completion": f"c{i}", "n": i} for i in range(start, start + 3)]
        write_jsonl(root / "data" / f"train-{shard}-of-00003.jsonl", rows)
    write_jsonl(root / "data" / "test-00000-of-00001.jsonl", [{"prompt": "t", "completion": "x"}])
    return root


def test_row_order_and_values_equal_load_dataset(tmp_path: Path, datasets_rows: Oracle) -> None:
    root = sharded_dir(tmp_path / "ds")
    stream = stream_for(local_source(root))
    rows = list(stream)
    expected = datasets_rows(root)["train"]
    assert [r.row for r in rows] == expected
    assert [r.row_index for r in rows] == list(range(len(expected)))
    assert [r.shard_id for r in rows] == [
        f"data/train-0000{k}-of-00003.jsonl" for k in (0, 0, 0, 1, 1, 1, 2, 2, 2)
    ]
    assert stream.complete
    assert stream.shards_completed == len(stream.shards) == 3


@pytest.mark.parametrize("name", ["preference.jsonl", "messages.jsonl", "text.csv", "array.json"])
def test_fixture_streams_equal_load_dataset(name: str, datasets_rows: Oracle) -> None:
    path = FIXTURES / name
    module = {"jsonl": "json", "json": "json", "csv": "csv"}[path.suffix.lstrip(".")]
    stream = stream_for(local_source(path))
    assert [r.row for r in stream] == datasets_rows(path, module)["train"]
    assert stream.complete


def test_test_split_is_never_read_when_train_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    original = SourceFiles.scan_source

    def spy(self: SourceFiles, data_file: Any) -> Any:
        opened.append(data_file.shard_id)
        return original(self, data_file)

    monkeypatch.setattr(SourceFiles, "scan_source", spy)
    stream = stream_for(local_source(FIXTURES / "multi_split"), "train")
    prompts = [r.row["prompt"] for r in stream]
    assert prompts == ["train:t1", "train:t2", "train:t3"]
    assert opened == ["train.jsonl"]
    assert stream.shards == ["train.jsonl"]


def test_hub_files_are_downloaded_on_demand_and_only_for_the_split(
    tmp_path: Path, fake_hub: dict[str, Any]
) -> None:
    root = sharded_dir(tmp_path / "repo")
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    stream = stream_for(source)
    assert fake_hub["__log__"] == []  # opening the stream downloads nothing
    first = next(iter(stream))
    assert first.row_index == 0
    assert fake_hub["__log__"] == ["download:data/train-00000-of-00003.jsonl"]
    list(stream)
    assert not any("test-00000" in entry for entry in fake_hub["__log__"])
    assert stream.complete


def test_complete_only_after_eof_of_every_shard(tmp_path: Path) -> None:
    stream = stream_for(local_source(sharded_dir(tmp_path / "ds")))
    assert not stream.complete
    iterator = iter(stream)
    next(iterator)
    assert not stream.complete
    for _ in range(4):
        next(iterator)
    assert not stream.complete and stream.shards_completed == 1
    rest = list(iterator)
    assert len(rest) == 4
    assert stream.complete and stream.shards_completed == 3


def test_abandoned_iteration_is_not_complete(tmp_path: Path) -> None:
    stream = stream_for(local_source(sharded_dir(tmp_path / "ds")))
    for row in stream:
        if row.row_index == 4:
            break
    assert not stream.complete
    assert stream.shards_completed == 1


def test_reiteration_restarts_from_the_first_row(tmp_path: Path) -> None:
    stream = stream_for(local_source(sharded_dir(tmp_path / "ds")))
    first = [r.row for r in stream]
    second = [r.row for r in stream]
    assert first == second and stream.complete


def test_quota_stops_the_stream_and_later_shards_are_not_read(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    write_jsonl(root / "train-0.jsonl", [{"a": "short"}] * 2)
    write_jsonl(root / "train-1.jsonl", [{"a": "x" * 4000}])
    write_jsonl(root / "train-2.jsonl", [{"a": "never"}])
    stream = stream_for(local_source(root), limits=ReaderLimits(max_row_bytes=1000))
    rows = list(stream)
    assert [r.row["a"] for r in rows] == ["short", "short"]
    assert not stream.complete
    assert stream.shards_completed == 1
    issue = stream.issues[-1]
    assert issue.code == ErrorCode.SCAN_QUOTA_EXCEEDED
    assert issue.details["limit"] == "max_row_bytes"
    assert issue.details["shard_id"] == "train-1.jsonl"


def test_broken_shard_stops_the_stream(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    pq.write_table(pa.table({"prompt": ["a", "b"]}), root / "train-0.parquet")
    (root / "train-1.parquet").write_bytes(b"PAR1" + b"\x00" * 32)
    pq.write_table(pa.table({"prompt": ["c"]}), root / "train-2.parquet")
    stream = stream_for(local_source(root))
    assert [r.row["prompt"] for r in stream] == ["a", "b"]
    assert not stream.complete
    assert stream.shards_completed == 1
    assert stream.issues[-1].code == ErrorCode.SCAN_PARTIAL
    assert stream.issues[-1].details["shard_id"] == "train-1.parquet"


def test_malformed_records_are_failed_rows_with_positions() -> None:
    stream = stream_for(local_source(FIXTURES / "malformed.jsonl"))
    rows = list(stream)
    assert [r.row_index for r in rows] == [0, 1, 2, 3, 4]
    failed = [r for r in rows if is_failed_row(r)]
    assert len(failed) == 1
    bad = failed[0]
    assert isinstance(bad, FailedSourceRow) and isinstance(bad, SourceRow)
    assert (bad.row_index, bad.line, bad.reason, dict(bad.row)) == (2, 3, "json_parse", {})
    assert bad.error_code == ErrorCode.SCAN_FAILED_ROWS
    assert "3번째 줄" in bad.message and "malformed.jsonl" in bad.message
    assert "broken" not in bad.message  # raw data never leaks into messages
    assert (stream.rows_ok, stream.rows_failed) == (4, 1)
    assert stream.shards_completed == 1
    assert not stream.complete  # read to EOF, but a record is missing (plan §19.2)
    summary = stream.issues[-1]
    assert summary.code == ErrorCode.SCAN_FAILED_ROWS
    assert summary.details["sample"][0]["row_index"] == 2


def test_too_many_failed_rows_stops(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{oops\n" * 10 + '{"a": 1}\n', encoding="utf-8")
    stream = stream_for(local_source(path), limits=ReaderLimits(max_failed_rows=3))
    rows = list(stream)
    assert len(rows) == 4
    assert stream.issues[0].details["reason"] == "too_many_failed_rows"
    assert not stream.complete


def test_manifest_digest_mismatch_stops_before_reading(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "train.jsonl", [{"a": 1}])
    entry = FileEntry(path="train.jsonl", size=path.stat().st_size, sha256="0" * 64)
    stream = stream_for(local_source(path, [entry]))
    assert list(stream) == []
    assert stream.issues[-1].code == ErrorCode.SOURCE_REVISION_CHANGED
    assert stream.issues[-1].details["reason"] == "sha256_mismatch"
    assert not stream.complete


def test_hub_git_blob_id_is_verified(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    root = sharded_dir(tmp_path / "repo")
    entries = []
    for entry in file_entries(root):
        content = (root / entry.path).read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False)
        entries.append(FileEntry(path=entry.path, size=entry.size, blob_id=blob.hexdigest()))
    source = hf_source(root)
    source = ResolvedSource(
        kind="dataset",
        manifest=source.manifest.model_copy(update={"files": entries}),
        repo_id=source.repo_id,
        revision=source.revision,
    )
    fake_hub[source.repo_id] = root
    stream = stream_for(source)
    assert len(list(stream)) == 9 and stream.complete
    (root / "data/train-00001-of-00003.jsonl").write_text('{"prompt": "changed"}\n')
    stream = stream_for(source)
    assert len(list(stream)) == 3
    assert stream.issues[-1].details["reason"] in ("size_mismatch", "blob_id_mismatch")


def test_file_changed_while_reading_is_detected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "train.jsonl", [{"a": i} for i in range(3)])
    stream = stream_for(local_source(path, [FileEntry(path="train.jsonl")]))
    seen = []
    for row in stream:
        seen.append(row.row["a"])
        if row.row_index == 0:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"a": 99}) + "\n")
    assert not stream.complete
    assert stream.issues[-1].details["reason"] == "changed_while_reading"


def test_row_count_must_match_declared_metadata(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    write_jsonl(root / "train.jsonl", [{"a": 1}, {"a": 2}])
    (root / "README.md").write_text(
        "---\ndataset_info:\n  splits:\n  - name: train\n    num_examples: 3\n---\n",
        encoding="utf-8",
    )
    stream = stream_for(local_source(root))
    assert stream.total_rows == 3
    assert len(list(stream)) == 2
    assert not stream.complete
    assert stream.issues[-1].details["reason"] == "row_count_mismatch"


def test_total_rows_from_local_parquet_footers() -> None:
    stream = stream_for(local_source(FIXTURES / "prompt_completion.parquet"))
    assert stream.total_rows == 5
    assert len(list(stream)) == 5 and stream.complete


def test_unknown_total_rows_is_none() -> None:
    stream = stream_for(local_source(FIXTURES / "preference.jsonl"))
    assert stream.total_rows is None
    assert stream.split == "train" and stream.config == "default"
