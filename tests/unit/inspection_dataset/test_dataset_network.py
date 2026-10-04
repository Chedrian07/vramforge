"""The real example dataset (plan.md §21) at its pinned revision.

Run with `VRAMFORGE_NETWORK_TESTS=1` (optionally `HF_HOME=<cache>` to reuse a warm Hub cache).
Facts asserted here are VERIFIED in docs/research/example-model-dataset.md §4: one JSON Lines file
named `.json`, 4,656 rows, six string columns, `system` empty in every row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vramforge_estimator.inspection import inspect_dataset, open_rows
from vramforge_estimator.inspection.dataset import inspect_dataset_details
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    DatasetSourceRef,
    FileEntry,
    Objective,
    SourceManifest,
    SourceType,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

pytestmark = pytest.mark.network

REPO = "CyberNative/Code_Vulnerability_Security_DPO"
REVISION = "81aeacf06cf43b16d7278a3a01f019a496a53c51"
DATA_FILE = "secure_programming_dpo.json"
COLUMNS = ["lang", "vulnerability", "system", "question", "chosen", "rejected"]
EXPECTED_MAPPING = ColumnMapping(
    format=DatasetFormat.PREFERENCE,
    system="system",
    prompt="question",
    chosen="chosen",
    rejected="rejected",
)


@pytest.fixture(scope="module")
def access() -> SourceAccess:
    home = os.environ.get("HF_HOME")
    return SourceAccess(hf_home=Path(home) if home else None)


@pytest.fixture(scope="module")
def source() -> ResolvedSource:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    entries = [
        FileEntry(
            path=item.path,
            size=item.size,
            blob_id=item.blob_id,
            sha256=item.lfs.sha256 if item.lfs else None,
        )
        for item in HfApi().list_repo_tree(
            REPO, repo_type="dataset", revision=REVISION, recursive=True, token=False
        )
        if isinstance(item, RepoFile)
    ]
    manifest = SourceManifest(
        kind="dataset",
        source_type=SourceType.HUGGINGFACE,
        reference=f"hf:{REPO}",
        repo_id=REPO,
        requested_revision=REVISION,
        resolved_revision=REVISION,
        files=entries,
        fingerprint=REVISION,
    )
    return ResolvedSource(kind="dataset", manifest=manifest, repo_id=REPO, revision=REVISION)


def ref() -> DatasetSourceRef:
    return DatasetSourceRef(reference=REPO, revision=REVISION)


@pytest.mark.parametrize("objective", [Objective.GRPO, Objective.DPO, Objective.SFT])
def test_example_dataset_inspection(
    source: ResolvedSource, access: SourceAccess, objective: Objective
) -> None:
    result = inspect_dataset(source, ref(), access, objective)
    assert result.configs == ["default"] and result.selected_config == "default"
    assert [s.name for s in result.splits] == ["train"]
    assert result.selected_split == "train" and result.split_auto_selected
    assert [(c.name, c.dtype, c.kind) for c in result.columns] == [
        (name, "string", "string") for name in COLUMNS
    ]
    assert result.detected_format == DatasetFormat.PREFERENCE
    assert result.suggested_mapping == EXPECTED_MAPPING
    assert not result.mapping_ambiguous
    assert result.issues == []


def test_example_dataset_is_detected_as_json_lines(
    source: ResolvedSource, access: SourceAccess
) -> None:
    details = inspect_dataset_details(source, ref(), access, Objective.GRPO)
    assert details.file_formats == {DATA_FILE: "json_lines"}
    assert details.layout is not None
    assert [f.shard_id for f in details.layout.configs[0].splits[0].files] == [DATA_FILE]
    assert any("JSON Lines" in note for note in details.inspection.manifest.notes)


def test_example_dataset_full_stream(source: ResolvedSource, access: SourceAccess) -> None:
    from huggingface_hub import hf_hub_download

    stream = open_rows(source, config=None, split="train", access=access)
    rows = list(stream)
    assert stream.complete
    assert stream.shards == [DATA_FILE] and stream.shards_completed == 1
    assert len(rows) == 4656
    assert [r.row_index for r in rows] == list(range(4656))
    assert all(list(r.row) == COLUMNS for r in rows)
    assert all(r.row["system"] == "" for r in rows)
    cache = access.hf_home / "hub" if access.hf_home else None
    path = hf_hub_download(
        REPO, DATA_FILE, repo_type="dataset", revision=REVISION, cache_dir=cache, token=False
    )
    expected = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert [dict(r.row) for r in rows] == expected
