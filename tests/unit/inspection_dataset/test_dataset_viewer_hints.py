"""Dataset viewer URLs (`/viewer/<config>/<split>`) select the config and split to analyze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dataset_testkit import hf_source, write_jsonl

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import open_rows
from vramforge_estimator.inspection.dataset import (
    DatasetInspectionDetails,
    RequestedSelection,
    inspect_dataset_details,
    requested_selection,
)
from vramforge_estimator.schemas import DatasetSourceRef, ErrorCode, Objective
from vramforge_estimator.sources import ResolvedSource, SourceAccess

VIEWER = "https://huggingface.co/datasets/acme/demo-set/viewer"
README = """---
configs:
- config_name: alpha
  data_files:
  - split: train
    path: alpha/train.jsonl
  - split: test
    path: alpha/test.jsonl
- config_name: beta
  data_files:
  - split: train
    path: beta/train.jsonl
  - split: test
    path: beta/test.jsonl
---
"""


@pytest.fixture
def two_configs(tmp_path: Path, fake_hub: dict[str, Any]) -> ResolvedSource:
    root = tmp_path / "repo"
    for split, count in (("train", 3), ("test", 2)):
        write_jsonl(
            root / "alpha" / f"{split}.jsonl",
            [{"prompt": f"alpha-{split}-{i}", "completion": "c"} for i in range(count)],
        )
        write_jsonl(
            root / "beta" / f"{split}.jsonl", [{"text": f"beta-{split}-{i}"} for i in range(count)]
        )
    (root / "README.md").write_text(README, encoding="utf-8")
    source = hf_source(root)
    fake_hub[source.repo_id] = root
    return source


def inspect_url(source: ResolvedSource, reference: str, **fields: Any) -> DatasetInspectionDetails:
    ref = DatasetSourceRef(reference=reference, **fields)
    return inspect_dataset_details(source, ref, SourceAccess(), Objective.SFT)


def data_reads(fake_hub: dict[str, Any]) -> list[str]:
    return [entry for entry in fake_hub["__log__"] if entry.endswith(".jsonl")]


def test_viewer_config_and_split_are_the_selection(
    two_configs: ResolvedSource, fake_hub: dict[str, Any]
) -> None:
    details = inspect_url(two_configs, f"{VIEWER}/beta/test?row=1")
    result = details.inspection
    assert (result.selected_config, result.selected_split) == ("beta", "test")
    assert not result.split_auto_selected  # the user chose it through the URL
    assert (details.config_origin, details.split_origin) == ("viewer_url", "viewer_url")
    assert result.issues == []
    assert [(c.name, c.kind) for c in result.columns] == [("text", "string")]
    assert data_reads(fake_hub) == ["download:beta/test.jsonl"]  # only the chosen split
    assert result.manifest is not None
    assert (
        "데이터셋 viewer 주소에 지정된 config 'beta', split 'test'을(를) 분석 대상으로 "
        "선택했습니다." in result.manifest.notes
    )
    # The selection feeds the full scan; the URL's ?row= never narrows it.
    stream = open_rows(
        two_configs,
        config=result.selected_config,
        split=result.selected_split,
        access=SourceAccess(),
    )
    assert [r.row["text"] for r in stream] == ["beta-test-0", "beta-test-1"]
    assert stream.complete


def test_viewer_config_only_keeps_the_train_auto_selection(two_configs: ResolvedSource) -> None:
    details = inspect_url(two_configs, f"{VIEWER}/beta")
    result = details.inspection
    assert (result.selected_config, result.selected_split) == ("beta", "train")
    assert result.split_auto_selected
    assert (details.config_origin, details.split_origin) == ("viewer_url", "auto")
    assert result.manifest is not None
    notes = [note for note in result.manifest.notes if "viewer 주소에 지정된" in note]
    assert notes == [
        "데이터셋 viewer 주소에 지정된 config 'beta'을(를) 분석 대상으로 선택했습니다."
    ]


def test_without_a_viewer_path_several_configs_still_need_a_choice(
    two_configs: ResolvedSource,
) -> None:
    details = inspect_url(two_configs, "https://huggingface.co/datasets/acme/demo-set")
    result = details.inspection
    assert result.selected_config is None and details.config_origin is None
    assert [issue.code for issue in result.issues] == [ErrorCode.DATASET_CONFIG_REQUIRED]
    assert "requested_from" not in result.issues[0].details
    assert result.manifest is not None
    assert not any("viewer 주소에 지정된" in note for note in result.manifest.notes)


def test_request_fields_take_precedence_over_matching_url_values(
    two_configs: ResolvedSource,
) -> None:
    both = inspect_url(two_configs, f"{VIEWER}/beta/test", config="beta", split="test")
    assert (both.config_origin, both.split_origin) == ("request", "request")
    assert both.inspection.manifest is not None
    assert not any("viewer 주소에 지정된" in note for note in both.inspection.manifest.notes)
    mixed = inspect_url(two_configs, f"{VIEWER}/beta/test", config="beta")
    assert (mixed.config_origin, mixed.split_origin) == ("request", "viewer_url")
    assert (mixed.inspection.selected_config, mixed.inspection.selected_split) == ("beta", "test")
    assert mixed.inspection.manifest is not None
    assert (
        "데이터셋 viewer 주소에 지정된 split 'test'을(를) 분석 대상으로 선택했습니다."
        in mixed.inspection.manifest.notes
    )


def test_url_and_field_that_disagree_are_never_resolved_silently(
    two_configs: ResolvedSource, fake_hub: dict[str, Any]
) -> None:
    with pytest.raises(EstimatorError) as excinfo:
        inspect_url(two_configs, f"{VIEWER}/beta/test", split="train")
    assert excinfo.value.issue.code == ErrorCode.CONFLICTING_OPTIONS
    assert data_reads(fake_hub) == []  # rejected before anything is read


def test_unknown_viewer_config_asks_for_a_config(two_configs: ResolvedSource) -> None:
    details = inspect_url(two_configs, f"{VIEWER}/gamma/train")
    result = details.inspection
    assert result.selected_config is None and result.splits == []
    assert details.config_origin is None and details.split_origin is None
    issue = result.issues[0]
    assert [i.code for i in result.issues] == [ErrorCode.DATASET_CONFIG_REQUIRED]
    assert issue.details["options"] == ["alpha", "beta"]
    assert (issue.details["requested"], issue.details["requested_from"]) == ("gamma", "viewer_url")
    assert "viewer 주소" in issue.user_message and "gamma" in issue.user_message


def test_unknown_viewer_split_asks_for_a_split(two_configs: ResolvedSource) -> None:
    details = inspect_url(two_configs, f"{VIEWER}/beta/validation")
    result = details.inspection
    assert result.selected_config == "beta" and details.config_origin == "viewer_url"
    assert result.selected_split is None and details.split_origin is None
    issue = result.issues[0]
    assert [i.code for i in result.issues] == [ErrorCode.DATASET_SPLIT_REQUIRED]
    assert issue.details["options"] == ["train", "test"]
    assert issue.details["field"] == "dataset.split"
    assert (issue.details["requested"], issue.details["requested_from"]) == (
        "validation",
        "viewer_url",
    )
    assert [c.name for c in result.columns] == ["text"]  # columns of the config still shown
    assert result.manifest is not None  # the config choice is recorded, the failed split is not
    assert (
        "데이터셋 viewer 주소에 지정된 config 'beta'을(를) 분석 대상으로 선택했습니다."
        in result.manifest.notes
    )


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, RequestedSelection("beta", "test", "viewer_url", "viewer_url")),
        (
            {"config": "  ", "split": ""},
            RequestedSelection("beta", "test", "viewer_url", "viewer_url"),
        ),
        ({"config": " beta "}, RequestedSelection("beta", "test", "request", "viewer_url")),
    ],
)
def test_requested_selection(fields: dict[str, str], expected: RequestedSelection) -> None:
    ref = DatasetSourceRef(reference=f"{VIEWER}/beta/test", **fields)
    assert requested_selection(ref) == expected


def test_requested_selection_without_url_hints() -> None:
    plain = DatasetSourceRef(reference="acme/demo-set", config="alpha")
    assert requested_selection(plain) == RequestedSelection("alpha", None, "request", None)
    assert requested_selection(DatasetSourceRef(reference="acme/demo-set")) == (
        RequestedSelection()
    )
