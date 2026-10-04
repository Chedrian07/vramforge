"""inspect_dataset / open_rows end to end on local fixtures and an offline fake Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dataset_testkit import FIXTURES, hf_source, local_source, write_jsonl

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import inspect_dataset, open_rows
from vramforge_estimator.inspection.dataset import inspect_dataset_details
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    DatasetInspection,
    DatasetSourceRef,
    ErrorCode,
    Objective,
    SourceType,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

EXAMPLE_MAPPING = ColumnMapping(
    format=DatasetFormat.PREFERENCE,
    system="system",
    prompt="question",
    chosen="chosen",
    rejected="rejected",
)


def ref(**kwargs: Any) -> DatasetSourceRef:
    return DatasetSourceRef(source_type=SourceType.LOCAL, reference="local:fixtures/x", **kwargs)


def inspect(
    source: ResolvedSource,
    objective: Objective | None = None,
    access: SourceAccess | None = None,
    **ref_kwargs: Any,
) -> DatasetInspection:
    return inspect_dataset(source, ref(**ref_kwargs), access or SourceAccess(), objective)


def codes(inspection: DatasetInspection) -> list[ErrorCode]:
    return [issue.code for issue in inspection.issues]


@pytest.mark.parametrize("objective", [None, Objective.SFT, Objective.DPO, Objective.GRPO])
def test_example_shaped_dataset(objective: Objective | None) -> None:
    result = inspect(local_source(FIXTURES / "preference.jsonl"), objective)
    assert result.configs == ["default"] and result.selected_config == "default"
    assert [s.name for s in result.splits] == ["train"]
    assert result.selected_split == "train" and result.split_auto_selected
    assert [(c.name, c.kind) for c in result.columns] == [
        ("lang", "string"),
        ("vulnerability", "string"),
        ("system", "string"),
        ("question", "string"),
        ("chosen", "string"),
        ("rejected", "string"),
    ]
    assert result.detected_format == DatasetFormat.PREFERENCE
    assert result.suggested_mapping == EXAMPLE_MAPPING
    assert result.mapping_candidates == [EXAMPLE_MAPPING]
    assert not result.mapping_ambiguous
    assert result.issues == []


def test_json_lines_inside_a_json_file_is_detected() -> None:
    details = inspect_dataset_details(
        local_source(FIXTURES / "json_lines.json"), ref(), SourceAccess(), Objective.DPO
    )
    assert details.file_formats == {"json_lines.json": "json_lines"}
    assert any("JSON Lines" in note for note in details.inspection.manifest.notes)
    assert details.inspection.suggested_mapping == EXAMPLE_MAPPING


@pytest.mark.parametrize(
    ("name", "objective", "fmt"),
    [
        ("messages.jsonl", Objective.SFT, DatasetFormat.MESSAGES),
        ("prompt_completion.parquet", Objective.SFT, DatasetFormat.PROMPT_COMPLETION),
        ("text.csv", Objective.SFT, DatasetFormat.TEXT),
        ("array.json", Objective.GRPO, DatasetFormat.PROMPT_COMPLETION),
    ],
)
def test_formats_of_the_fixtures(name: str, objective: Objective, fmt: DatasetFormat) -> None:
    result = inspect(local_source(FIXTURES / name), objective)
    assert result.detected_format == fmt
    assert result.suggested_mapping is not None and result.suggested_mapping.format == fmt
    assert result.issues == []


def test_ambiguous_columns_need_input() -> None:
    result = inspect(local_source(FIXTURES / "ambiguous.jsonl"), Objective.DPO)
    assert result.mapping_ambiguous
    assert result.suggested_mapping is None
    assert len(result.mapping_candidates) >= 2
    assert codes(result) == [ErrorCode.COLUMN_MAPPING_REQUIRED]


def test_request_mapping_hint_resolves_the_ambiguity() -> None:
    result = inspect(
        local_source(FIXTURES / "ambiguous.jsonl"),
        Objective.DPO,
        mapping=ColumnMapping(prompt="question"),
    )
    assert not result.mapping_ambiguous
    assert result.suggested_mapping is not None
    assert result.suggested_mapping.prompt == "question"


def test_train_split_is_auto_selected_and_others_listed() -> None:
    result = inspect(local_source(FIXTURES / "multi_split"), Objective.SFT)
    assert [s.name for s in result.splits] == ["train", "test"]
    assert result.selected_split == "train" and result.split_auto_selected
    explicit = inspect(local_source(FIXTURES / "multi_split"), Objective.SFT, split="test")
    assert explicit.selected_split == "test" and not explicit.split_auto_selected


def test_unknown_split_and_bad_eval_split() -> None:
    source = local_source(FIXTURES / "multi_split")
    unknown = inspect(source, Objective.SFT, split="dev")
    assert unknown.selected_split is None
    issue = unknown.issues[0]
    assert issue.code == ErrorCode.DATASET_SPLIT_REQUIRED
    assert issue.details["options"] == ["train", "test"]
    assert [c.name for c in unknown.columns] == ["prompt", "completion"]  # still shown
    same = inspect(source, Objective.SFT, split="train", eval_split="train")
    assert codes(same) == [ErrorCode.DATASET_SPLIT_REQUIRED]
    assert same.issues[0].details["field"] == "dataset.eval_split"
    ok = inspect(source, Objective.SFT, eval_split="test")
    assert ok.issues == []


def test_no_train_split_and_several_splits_requires_a_choice(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    write_jsonl(root / "validation.jsonl", [{"prompt": "v", "completion": "1"}])
    write_jsonl(root / "test.jsonl", [{"prompt": "t", "completion": "2"}])
    result = inspect(local_source(root), Objective.SFT)
    assert result.selected_split is None and not result.split_auto_selected
    issue = next(i for i in result.issues if i.code == ErrorCode.DATASET_SPLIT_REQUIRED)
    assert issue.details["options"] == ["validation", "test"]
    assert issue.details["suggested"] is None
    assert [c.name for c in result.columns] == ["prompt", "completion"]  # columns still shown


def test_single_non_train_split_is_suggested_not_assumed(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    write_jsonl(root / "test.jsonl", [{"prompt": "t", "completion": "2"}])
    result = inspect(local_source(root), Objective.SFT)
    assert result.selected_split is None
    issue = next(i for i in result.issues if i.code == ErrorCode.DATASET_SPLIT_REQUIRED)
    assert issue.details["suggested"] == "test"


def readme_configs_dir(root: Path) -> Path:
    write_jsonl(root / "a" / "train.jsonl", [{"prompt": "a", "completion": "1"}])
    write_jsonl(root / "b" / "train.jsonl", [{"text": "b"}])
    (root / "README.md").write_text(
        "---\nconfigs:\n- config_name: alpha\n  data_files: a/*.jsonl\n"
        "- config_name: beta\n  data_files: b/*.jsonl\n---\n",
        encoding="utf-8",
    )
    return root


def test_several_configs_require_a_choice(tmp_path: Path) -> None:
    source = local_source(readme_configs_dir(tmp_path / "ds"))
    result = inspect(source, Objective.SFT)
    assert result.configs == ["alpha", "beta"]
    assert result.selected_config is None and result.splits == []
    issue = result.issues[0]
    assert issue.code == ErrorCode.DATASET_CONFIG_REQUIRED
    assert issue.details["options"] == ["alpha", "beta"]
    chosen = inspect(source, Objective.SFT, config="beta")
    assert chosen.selected_config == "beta"
    assert chosen.suggested_mapping == ColumnMapping(format=DatasetFormat.TEXT, text="text")
    missing = inspect(source, Objective.SFT, config="gamma")
    assert codes(missing) == [ErrorCode.DATASET_CONFIG_REQUIRED]


def test_malformed_rows_are_flagged_in_the_preview() -> None:
    result = inspect(local_source(FIXTURES / "malformed.jsonl"), Objective.SFT)
    assert codes(result) == [ErrorCode.SCAN_FAILED_ROWS]
    assert result.issues[0].severity.value == "warning"
    assert result.suggested_mapping is not None


def test_preview_reads_at_most_100_rows_and_respects_the_byte_budget(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "big.jsonl", [{"text": f"row {i} " + "x" * 50} for i in range(5000)]
    )
    details = inspect_dataset_details(local_source(path), ref(), SourceAccess(), Objective.SFT)
    assert details.preview is not None and len(details.preview.rows) == 100
    tight = SourceAccess(max_metadata_bytes=4 * 70 * 20)
    small = inspect_dataset_details(local_source(path), ref(), tight, Objective.SFT)
    assert small.preview is not None and 0 < len(small.preview.rows) < 100
    assert small.inspection.suggested_mapping == ColumnMapping(
        format=DatasetFormat.TEXT, text="text"
    )


def test_layout_errors_come_back_as_issues(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "demo.py").write_text("raise SystemExit\n", encoding="utf-8")
    write_jsonl(root / "train.jsonl", [{"text": "x"}])
    result = inspect(local_source(root), Objective.SFT)
    assert codes(result) == [ErrorCode.DATASET_FORMAT_UNSUPPORTED]
    assert result.configs == [] and result.manifest is not None


def test_unsupported_module_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "txt"
    root.mkdir()
    (root / "train.txt").write_text("hello\n", encoding="utf-8")
    result = inspect(local_source(root), Objective.SFT)
    assert ErrorCode.DATASET_FORMAT_UNSUPPORTED in codes(result)
    assert result.columns == []


def test_no_absolute_paths_in_results(tmp_path: Path) -> None:
    root = readme_configs_dir(tmp_path / "ds")
    write_jsonl(root / "stray.jsonl", [{"x": 1}])
    for source in (local_source(root), local_source(FIXTURES / "multi_split")):
        for kwargs in ({}, {"config": "alpha"}, {"split": "nope"}):
            dumped = inspect(source, Objective.SFT, **kwargs).model_dump_json()
            assert str(tmp_path) not in dumped
            assert str(FIXTURES) not in dumped


def test_hub_dataset_inspection_and_rows(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    root = tmp_path / "repo"
    rows = [
        json.loads(line) for line in (FIXTURES / "preference.jsonl").read_text("utf-8").splitlines()
    ]
    write_jsonl(root / "data" / "train-00000-of-00001.jsonl", rows)
    write_jsonl(root / "data" / "test-00000-of-00001.jsonl", rows[:1])
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    result = inspect_dataset(source, ref(), SourceAccess(), Objective.GRPO)
    assert result.suggested_mapping == EXAMPLE_MAPPING
    assert [s.name for s in result.splits] == ["train", "test"]
    assert fake_hub["__log__"] == ["download:data/train-00000-of-00001.jsonl"]
    stream = open_rows(source, config=None, split="train", access=SourceAccess())
    assert [r.row for r in stream] == rows
    assert stream.complete
    assert not any("test-00000" in entry for entry in fake_hub["__log__"])


def test_large_hub_file_preview_uses_ranged_reads(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    root = tmp_path / "repo"
    write_jsonl(root / "train.jsonl", [{"text": "t" * 200} for _ in range(2000)])
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    result = inspect_dataset(source, ref(), SourceAccess(max_metadata_bytes=64 * 1024), None)
    assert result.suggested_mapping == ColumnMapping(format=DatasetFormat.TEXT, text="text")
    assert fake_hub["__log__"] == ["open:train.jsonl"]  # no full download for the preview


def test_open_rows_contract_and_errors(tmp_path: Path) -> None:
    stream = open_rows(
        local_source(FIXTURES / "preference.jsonl"),
        config=None,
        split="train",
        access=SourceAccess(),
    )
    for attribute in ("split", "config", "shards", "total_rows", "complete", "shards_completed"):
        assert hasattr(stream, attribute)  # RowStream protocol (not runtime-checkable)
    assert (stream.split, stream.config, stream.shards) == (
        "train",
        "default",
        ["preference.jsonl"],
    )
    expected = [
        json.loads(line) for line in (FIXTURES / "preference.jsonl").read_text("utf-8").splitlines()
    ]
    assert [dict(r.row) for r in stream] == expected

    source = local_source(readme_configs_dir(tmp_path / "ds"))
    with pytest.raises(EstimatorError) as no_config:
        open_rows(source, config=None, split="train", access=SourceAccess())
    assert no_config.value.issue.code == ErrorCode.DATASET_CONFIG_REQUIRED
    with pytest.raises(EstimatorError) as no_split:
        open_rows(source, config="alpha", split="test", access=SourceAccess())
    assert no_split.value.issue.code == ErrorCode.DATASET_SPLIT_REQUIRED
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "train.txt").write_text("hello\n", encoding="utf-8")
    with pytest.raises(EstimatorError) as unsupported:
        open_rows(local_source(txt), config=None, split="train", access=SourceAccess())
    assert unsupported.value.issue.code == ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_local_parquet_split_rows_come_from_footers() -> None:
    result = inspect(local_source(FIXTURES / "prompt_completion.parquet"), Objective.SFT)
    assert [(s.name, s.num_rows, s.num_bytes) for s in result.splits] == [("train", 5, None)]


def test_hub_parquet_preview_falls_back_to_the_schema(
    tmp_path: Path, fake_hub: dict[str, Any]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "repo"
    root.mkdir()
    table = pa.table({"prompt": ["p" * 3000] * 300, "completion": ["c" * 3000] * 300})
    pq.write_table(table, root / "train.parquet", row_group_size=100, compression="none")
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    # Budget above the footer but below one row group: columns come from the Parquet schema.
    details = inspect_dataset_details(source, ref(), SourceAccess(max_metadata_bytes=80_000), None)
    assert fake_hub["__log__"] == ["open:train.parquet"]
    assert details.preview is not None and details.preview.rows == []
    assert [(c.name, c.kind) for c in details.inspection.columns] == [
        ("prompt", "string"),
        ("completion", "string"),
    ]
    assert details.inspection.suggested_mapping == ColumnMapping(
        format=DatasetFormat.PROMPT_COMPLETION, prompt="prompt", completion="completion"
    )
    # Budget below the footer: nothing is guessed, the reason is reported.
    tiny = inspect_dataset(source, ref(), SourceAccess(max_metadata_bytes=1_000), None)
    assert tiny.columns == [] and tiny.suggested_mapping is None
    assert codes(tiny) == [ErrorCode.SCAN_QUOTA_EXCEEDED]


def test_unexpected_preview_errors_become_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    import vramforge_estimator.inspection.dataset_schema as schema_module

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("library bug")
        yield  # pragma: no cover

    monkeypatch.setattr(schema_module, "iter_file", boom)
    result = inspect(local_source(FIXTURES / "preference.jsonl"), Objective.DPO)
    assert codes(result) == [ErrorCode.INTERNAL_ERROR]
    assert result.columns == [] and result.suggested_mapping is None
