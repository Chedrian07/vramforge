"""Dataset inspection and the full-scan row stream (plan.md §7.1, §7.2, §7.6, §7.7, §18).

`inspect_dataset` resolves configs and splits exactly like datasets==5.0.1 (`dataset_layout`),
selects them only when the choice is unambiguous (otherwise DATASET_CONFIG_REQUIRED /
DATASET_SPLIT_REQUIRED with the options), previews at most 100 rows within
`SourceAccess.max_metadata_bytes` (`dataset_schema`) and ranks column-mapping candidates for the
objective (`dataset_mapping`). It raises `EstimatorError` only when there is nothing the user could
choose to analyze (no loadable layout, or the config that would be analyzed is unsupported); every
other problem is an issue of the returned inspection.

`open_rows` returns a `DatasetRowStream` over every record of one split. Failure signal and stop
contract (see `dataset_stream`): undecodable records arrive as `FailedSourceRow` (a `SourceRow`
with an empty `row`, `error_code`, `reason`, Korean `message` and its line/element position;
import it from the dependency-free `dataset_rows` module); quotas, broken shards, manifest
mismatches and download failures end the stream with an entry in `stream.issues`;
`stream.complete` is True only after a clean EOF of every shard with no failed record.

Heavy libraries (datasets, pyarrow, pandas, huggingface_hub) are imported lazily so that importing
the package stays cheap and works without the `analysis` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import (
    DatasetInspection,
    DatasetSourceRef,
    DatasetSplitInfo,
    ErrorCode,
    Issue,
    Objective,
    Severity,
    Stage,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .base import RowStream

if TYPE_CHECKING:
    from .dataset_layout import ConfigLayout, DatasetLayout
    from .dataset_schema import Preview
    from .dataset_stream import DatasetRowStream
    from .readers import ReaderLimits

__all__ = ["DatasetInspectionDetails", "inspect_dataset", "inspect_dataset_details", "open_rows"]


@dataclass
class DatasetInspectionDetails:
    """The public inspection plus what produced it (for callers that need more than the API)."""

    inspection: DatasetInspection
    layout: DatasetLayout | None = None
    preview: Preview | None = None
    file_formats: dict[str, str] = field(default_factory=dict)


def inspect_dataset(
    source: ResolvedSource,
    ref: DatasetSourceRef,
    access: SourceAccess,
    objective: Objective | None = None,
    *,
    limits: ReaderLimits | None = None,
) -> DatasetInspection:
    """Configs, splits, columns, detected format and ranked mapping candidates (preview only).

    Raises `EstimatorError` when nothing can be analyzed (see the module docstring), so callers
    fail instead of asking the user for a config, split or mapping that cannot exist.
    """
    return inspect_dataset_details(source, ref, access, objective, limits=limits).inspection


def inspect_dataset_details(
    source: ResolvedSource,
    ref: DatasetSourceRef,
    access: SourceAccess,
    objective: Objective | None = None,
    *,
    limits: ReaderLimits | None = None,
) -> DatasetInspectionDetails:
    from .dataset_files import SourceFiles
    from .dataset_layout import resolve_layout
    from .dataset_mapping import analyze_mapping
    from .dataset_schema import preview_columns, read_preview
    from .dataset_stream import footer_rows
    from .readers import ReaderLimits

    limits = limits or ReaderLimits()
    files = SourceFiles(source, access, limits)
    layout = resolve_layout(files)  # EstimatorError: no loadable layout at all
    issues: list[Issue] = []
    configs = [config.name for config in layout.configs]
    config, config_issue = _select_config(layout, ref.config)
    if config_issue is not None:
        issues.append(config_issue)
    splits: list[DatasetSplitInfo] = []
    selected_split: str | None = None
    auto_selected = False
    preview = None
    if config is not None and config.unsupported is not None:
        raise EstimatorError(config.unsupported)  # the only (or the requested) config
    if config is not None:
        splits = [
            DatasetSplitInfo(
                name=split.name,
                num_rows=split.num_rows
                if split.num_rows is not None
                else footer_rows(files, split),
                num_bytes=split.num_bytes,
            )
            for split in config.splits
        ]
        selected_split, auto_selected, split_issue = _select_split(config, ref.split)
        if split_issue is not None:
            issues.append(split_issue)
        eval_issue = _check_eval_split(config, ref.eval_split, selected_split)
        if eval_issue is not None:
            issues.append(eval_issue)
        # Columns come from the selected split; while the split is still undecided the first
        # split of the config stands in (datasets shares one schema across a config's splits).
        preview_split = config.split(selected_split) if selected_split else None
        if preview_split is None and config.splits:
            preview_split = config.splits[0]
        if preview_split is not None:
            preview = read_preview(files, config, preview_split, access.max_metadata_bytes, limits)
            issues.extend(preview.issues)
    columns = preview_columns(preview) if preview is not None else []
    mapping = None
    if columns:
        mapping = analyze_mapping(columns, objective, ref.mapping)
        issues.extend(mapping.issues)
    file_formats = dict(preview.file_formats) if preview is not None else {}
    notes = [*layout.notes, *_format_notes(file_formats)]
    manifest = source.manifest
    if notes:
        manifest = manifest.model_copy(update={"notes": [*manifest.notes, *notes]})
    inspection = DatasetInspection(
        manifest=manifest,
        configs=configs,
        selected_config=config.name if config is not None else None,
        splits=splits,
        selected_split=selected_split,
        split_auto_selected=auto_selected,
        columns=columns,
        detected_format=mapping.detected_format if mapping is not None else None,
        mapping_candidates=mapping.candidates if mapping is not None else [],
        suggested_mapping=mapping.suggested if mapping is not None else None,
        mapping_ambiguous=mapping.ambiguous if mapping is not None else False,
        issues=issues,
    )
    return DatasetInspectionDetails(inspection, layout, preview, file_formats)


def open_rows(
    source: ResolvedSource,
    *,
    config: str | None,
    split: str,
    access: SourceAccess,
    limits: ReaderLimits | None = None,
) -> RowStream:
    """Open a sequential reader over every row of `split` (never executes dataset scripts).

    Raises `EstimatorError` when the config/split cannot be selected or is unsupported; problems
    while reading are reported through the stream (see the module docstring).
    """
    return open_dataset_stream(source, config=config, split=split, access=access, limits=limits)


def open_dataset_stream(
    source: ResolvedSource,
    *,
    config: str | None,
    split: str,
    access: SourceAccess,
    limits: ReaderLimits | None = None,
) -> DatasetRowStream:
    """`open_rows` with the concrete return type (`issues`, `rows_failed`, `file_formats`)."""
    from .dataset_files import SourceFiles
    from .dataset_layout import resolve_layout
    from .dataset_stream import DatasetRowStream
    from .readers import ReaderLimits

    limits = limits or ReaderLimits()
    files = SourceFiles(source, access, limits)
    layout = resolve_layout(files)
    selected, issue = _select_config(layout, config)
    if issue is not None or selected is None:
        raise EstimatorError(issue or _config_required(layout, None))
    if selected.unsupported is not None:
        raise EstimatorError(selected.unsupported)
    split_layout = selected.split(split)
    if split_layout is None:
        raise EstimatorError(
            _required(
                ErrorCode.DATASET_SPLIT_REQUIRED,
                f"요청한 split '{split}'이(가) 없습니다. 분석할 split을 선택해 주세요.",
                field="dataset.split",
                options=[s.name for s in selected.splits],
            )
        )
    return DatasetRowStream(files, selected, split_layout, limits)


# ---------------------------------------------------------------- selection


def _select_config(
    layout: DatasetLayout, requested: str | None
) -> tuple[ConfigLayout | None, Issue | None]:
    names = [config.name for config in layout.configs]
    if requested is not None:
        config = layout.config(requested)
        if config is not None:
            return config, None
        return None, _required(
            ErrorCode.DATASET_CONFIG_REQUIRED,
            f"요청한 설정 '{requested}'이(가) 없습니다. 설정을 선택해 주세요.",
            field="dataset.config",
            options=names,
            suggested=layout.default_config,
        )
    if len(names) == 1:
        return layout.configs[0], None
    return None, _config_required(layout, requested)


def _config_required(layout: DatasetLayout, requested: str | None) -> Issue:
    return _required(
        ErrorCode.DATASET_CONFIG_REQUIRED,
        "데이터셋에 설정(config)이 여러 개 있습니다. 분석할 설정을 선택해 주세요.",
        field="dataset.config",
        options=[config.name for config in layout.configs],
        suggested=layout.default_config,
    )


def _select_split(
    config: ConfigLayout, requested: str | None
) -> tuple[str | None, bool, Issue | None]:
    names = [split.name for split in config.splits]
    if requested is not None:
        if requested in names:
            return requested, False, None
        return (
            None,
            False,
            _required(
                ErrorCode.DATASET_SPLIT_REQUIRED,
                f"요청한 split '{requested}'이(가) 없습니다. 분석할 split을 선택해 주세요.",
                field="dataset.split",
                options=names,
            ),
        )
    if "train" in names:
        return "train", True, None
    return (
        None,
        False,
        _required(
            ErrorCode.DATASET_SPLIT_REQUIRED,
            "train split이 없습니다. 학습에 사용할 split을 선택해 주세요.",
            field="dataset.split",
            options=names,
            suggested=names[0] if len(names) == 1 else None,
        ),
    )


def _check_eval_split(
    config: ConfigLayout, eval_split: str | None, train_split: str | None
) -> Issue | None:
    if eval_split is None:
        return None
    names = [split.name for split in config.splits]
    if eval_split not in names:
        return _required(
            ErrorCode.DATASET_SPLIT_REQUIRED,
            f"평가 split '{eval_split}'이(가) 없습니다. 평가 split을 다시 선택해 주세요.",
            field="dataset.eval_split",
            options=names,
        )
    if eval_split == train_split:
        return _required(
            ErrorCode.DATASET_SPLIT_REQUIRED,
            "학습 split과 평가 split이 같습니다. 서로 다른 split을 선택해 주세요.",
            field="dataset.eval_split",
            options=[name for name in names if name != train_split],
        )
    return None


def _required(code: ErrorCode, message: str, **details: Any) -> Issue:
    return make_issue(
        code,
        message,
        severity=Severity.ERROR,
        stage=Stage.INSPECTING,
        component="dataset",
        **details,
    )


def _format_notes(file_formats: dict[str, str]) -> list[str]:
    notes = []
    for shard_id, file_format in file_formats.items():
        if file_format == "json_lines" and shard_id.lower().endswith(".json"):
            notes.append(
                f"{shard_id}: 확장자는 .json이지만 내용이 JSON Lines(줄마다 객체 하나)라서 "
                "datasets와 같이 JSON Lines로 읽습니다."
            )
        elif file_format == "json_array" and shard_id.lower().endswith((".jsonl", ".ndjson")):
            notes.append(
                f"{shard_id}: 확장자는 JSON Lines지만 내용이 JSON 배열이라 배열로 읽습니다."
            )
    return notes
