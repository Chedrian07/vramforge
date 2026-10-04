"""Config/split/file resolution: identical to datasets 5.0.1 for local dirs and pinned Hub repos."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dataset_testkit import FIXTURES, file_entries, hf_source, local_source

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection.dataset_files import SourceFiles
from vramforge_estimator.inspection.dataset_layout import DatasetLayout, resolve_layout
from vramforge_estimator.inspection.readers import ReaderLimits
from vramforge_estimator.schemas import ErrorCode, FileEntry
from vramforge_estimator.sources import ResolvedSource, SourceAccess

LAYOUTS: dict[str, dict[str, str]] = {
    "single_file": {"data.jsonl": '{"a": 1}\n', "README.md": "# demo\n"},
    "named_splits": {
        "train.jsonl": '{"a": 1}\n',
        "test.jsonl": '{"a": 2}\n',
        "validation.jsonl": '{"a": 3}\n',
    },
    "sharded": {
        "data/train-00000-of-00002.jsonl": '{"a": 1}\n',
        "data/train-00001-of-00002.jsonl": '{"a": 2}\n',
        "data/test-00000-of-00001.jsonl": '{"a": 3}\n',
    },
    "split_dirs": {
        "data/train/b.jsonl": '{"a": 1}\n',
        "data/train/a.jsonl": '{"a": 2}\n',
        "data/test/a.jsonl": '{"a": 3}\n',
        ".hidden/x.jsonl": '{"a": 4}\n',
        "__pycache__/y.jsonl": '{"a": 5}\n',
    },
    "keywords": {
        "train_1.csv": "t\n1\n",
        "train_10.csv": "t\n10\n",
        "train_2.csv": "t\n2\n",
        "training.csv": "t\n3\n",
        "valid.csv": "t\n4\n",
        "eval.csv": "t\n5\n",
        "other.csv": "t\n6\n",
    },
    "readme_configs": {
        "README.md": (
            "---\nconfigs:\n- config_name: chat\n  data_files:\n  - split: train\n"
            "    path: chat/*.jsonl\n- config_name: tabular\n  data_files: tab/*.tsv\n"
            '  sep: "\\t"\n  default: true\ndataset_info:\n- config_name: chat\n'
            "  splits:\n  - name: train\n    num_bytes: 10\n    num_examples: 2\n---\n"
        ),
        "chat/b.jsonl": '{"a": 1}\n',
        "chat/a.jsonl": '{"a": 2}\n',
        "tab/x.tsv": "a\tb\n1\t2\n",
    },
}


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def layout_of(source: ResolvedSource) -> DatasetLayout:
    return resolve_layout(SourceFiles(source, SourceAccess(), ReaderLimits()))


def split_files(layout: DatasetLayout, config: str | None = None) -> dict[str, list[str]]:
    chosen = layout.config(config) if config else layout.configs[0]
    assert chosen is not None
    return {split.name: [f.shard_id for f in split.files] for split in chosen.splits}


def oracle(root: Path, config: str | None = None) -> tuple[dict[str, list[str]], Any]:
    import datasets

    datasets.disable_progress_bars()
    root = root.resolve()
    builder = datasets.load_dataset_builder(str(root), config, cache_dir=str(root / ".cache"))
    files = {
        str(split): [Path(os.path.relpath(f, root)).as_posix() for f in urls]
        for split, urls in builder.config.data_files.items()
    }
    return files, builder


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_local_dir_matches_datasets(tmp_path: Path, name: str) -> None:
    root = make_tree(tmp_path / name, LAYOUTS[name])
    layout = layout_of(local_source(root))
    expected, builder = oracle(root)
    assert [c.name for c in layout.configs] == list(builder.builder_configs)
    assert layout.default_config == builder.DEFAULT_CONFIG_NAME
    assert split_files(layout, builder.config.name) == expected


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_hub_manifest_listing_matches_datasets(
    tmp_path: Path, name: str, fake_hub: dict[str, Any]
) -> None:
    root = make_tree(tmp_path / name, LAYOUTS[name])
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    hub_layout = layout_of(source)
    local_layout = layout_of(local_source(root))
    assert [c.name for c in hub_layout.configs] == [c.name for c in local_layout.configs]
    for config in local_layout.configs:
        assert split_files(hub_layout, config.name) == split_files(local_layout, config.name)
    assert hub_layout.default_config == local_layout.default_config
    # Only card files were fetched while resolving: data files are downloaded on demand.
    assert set(fake_hub["__log__"]) <= {"README.md"}


def test_files_are_sorted_per_pattern_not_naturally(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "k", LAYOUTS["keywords"])
    layout = layout_of(local_source(root))
    assert split_files(layout)["train"] == [
        "train_1.csv",
        "train_10.csv",
        "train_2.csv",
        "training.csv",
    ]
    assert split_files(layout)["test"] == ["eval.csv"]
    assert any("other.csv" in note for note in layout.notes)  # unassigned file is reported


def test_readme_configs_options_infos_and_default(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "cfg", LAYOUTS["readme_configs"])
    layout = layout_of(local_source(root))
    assert [c.name for c in layout.configs] == ["chat", "tabular"]
    assert layout.default_config == "tabular"
    chat = layout.config("chat")
    tabular = layout.config("tabular")
    assert chat is not None and tabular is not None
    assert chat.module == tabular.module == "json"  # datasets infers one module per dataset
    assert split_files(layout, "chat") == {"train": ["chat/a.jsonl", "chat/b.jsonl"]}
    assert chat.splits[0].num_rows == 2 and chat.splits[0].num_bytes == 10
    assert "sep" not in tabular.options  # not a JsonConfig field: datasets ignores it too
    assert tabular.unsupported is not None  # .tsv files under the json module cannot be read
    assert tabular.unsupported.code == ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_duplicate_file_in_a_split_is_kept_and_noted(tmp_path: Path) -> None:
    root = make_tree(
        tmp_path / "dup", {"data/train-0.train.jsonl": "{}\n", "x/train.jsonl": "{}\n"}
    )
    layout = layout_of(local_source(root))
    expected, _ = oracle(root)
    assert split_files(layout) == expected
    assert split_files(layout)["train"].count("data/train-0.train.jsonl") == 2
    assert any("중복" in note for note in layout.notes)


def test_multi_split_fixture() -> None:
    layout = layout_of(local_source(FIXTURES / "multi_split"))
    assert split_files(layout) == {"train": ["train.jsonl"], "test": ["test.jsonl"]}


def test_single_local_file_is_one_train_split() -> None:
    layout = layout_of(local_source(FIXTURES / "prompt_completion.parquet"))
    assert layout.kind == "local_file"
    assert [(c.name, c.module) for c in layout.configs] == [("default", "parquet")]
    assert split_files(layout) == {"train": ["prompt_completion.parquet"]}


def test_upload_with_opaque_name_uses_the_manifest_file_name(tmp_path: Path) -> None:
    stored = tmp_path / "upload-7f3a"
    stored.write_bytes((FIXTURES / "preference.jsonl").read_bytes())
    entry = FileEntry(path="my_pairs.jsonl", size=stored.stat().st_size)
    layout = layout_of(local_source(stored, [entry]))
    assert layout.configs[0].module == "json"
    assert split_files(layout) == {"train": ["my_pairs.jsonl"]}


@pytest.mark.parametrize("hub", [False, True])
def test_loading_scripts_are_rejected(tmp_path: Path, hub: bool, fake_hub: dict[str, Any]) -> None:
    name = "demo-set"
    root = make_tree(tmp_path / name, {f"{name}.py": "print('never run')\n", "data.jsonl": "{}\n"})
    if hub:
        source = hf_source(root, repo_id=f"acme/{name}")
        fake_hub[source.repo_id] = root
    else:
        source = local_source(root)
    with pytest.raises(EstimatorError) as excinfo:
        layout_of(source)
    assert excinfo.value.issue.code == ErrorCode.DATASET_FORMAT_UNSUPPORTED
    assert excinfo.value.issue.details["reason"] == "loading_script"


def test_script_only_repo_is_unsupported(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "scripted", {"loader.py": "x = 1\n", "README.md": "# hi\n"})
    with pytest.raises(EstimatorError) as excinfo:
        layout_of(local_source(root))
    assert excinfo.value.issue.code == ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_no_data_files(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "empty", {"README.md": "# nothing\n", "LICENSE": "MIT\n"})
    with pytest.raises(EstimatorError) as excinfo:
        layout_of(local_source(root))
    assert excinfo.value.issue.code == ErrorCode.SOURCE_NOT_FOUND


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ({"a.txt": "plain text\n"}, "module_not_supported"),
        ({"train.jsonl": "{}\n", "train-more.zip": "PK"}, "archive_not_supported"),
    ],
)
def test_unsupported_configs_are_flagged(
    tmp_path: Path, files: dict[str, str], reason: str
) -> None:
    root = make_tree(tmp_path / "u", files)
    config = layout_of(local_source(root)).configs[0]
    assert config.unsupported is not None
    assert config.unsupported.details["reason"] == reason


def test_invalid_readme_yaml_is_reported(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "bad", {"README.md": "---\nconfigs: [\n---\n", "d.jsonl": "{}\n"})
    with pytest.raises(EstimatorError) as excinfo:
        layout_of(local_source(root))
    assert excinfo.value.issue.code == ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_symlink_escaping_the_dataset_dir_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"secret": 1}\n', encoding="utf-8")
    root = make_tree(tmp_path / "ds", {"train.jsonl": "{}\n"})
    (root / "test.jsonl").symlink_to(outside)
    with pytest.raises(EstimatorError) as excinfo:
        layout_of(local_source(root, file_entries(root)))
    assert excinfo.value.issue.code == ErrorCode.LOCAL_PATH_NOT_ALLOWED
    assert str(tmp_path) not in excinfo.value.issue.user_message


def test_hub_source_without_revision_is_rejected(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "r", {"d.jsonl": "{}\n"})
    source = hf_source(root)
    unpinned = ResolvedSource(kind="dataset", manifest=source.manifest, repo_id=source.repo_id)
    with pytest.raises(EstimatorError) as excinfo:
        SourceFiles(unpinned, SourceAccess(), ReaderLimits())
    assert excinfo.value.issue.code == ErrorCode.SOURCE_REVISION_CHANGED
