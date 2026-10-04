"""Dataset inspection and the full-scan row stream (plan.md §7.1, §7.2, §7.6, §7.7, §18).

`inspect_dataset` resolves configs and splits exactly like datasets==5.0.1 (`dataset_layout`),
selects them only when the choice is unambiguous (otherwise DATASET_CONFIG_REQUIRED /
DATASET_SPLIT_REQUIRED with the options), previews at most 100 rows within
`SourceAccess.max_metadata_bytes` (`dataset_schema`) and ranks column-mapping candidates for the
objective (`dataset_mapping`). It raises `EstimatorError` only when there is nothing the user could
choose to analyze (no loadable layout, the config that would be analyzed is unsupported, the
preview read the only split of the only config to its end without a row: EMPTY_DATASET, or a Hub
request failed before any column was seen: the retryable or access issue of `remote_issue`);
every other problem is an issue of the returned inspection (an empty split among others, or a
failed request after some rows were previewed, included).

The requested config/split are the request fields, or, when a field is empty, the
`/viewer/<config>/<split>` path of a dataset viewer URL (`requested_selection`). A viewer value is
a user choice like a field: it is never auto-selected, a missing one is DATASET_CONFIG_REQUIRED /
DATASET_SPLIT_REQUIRED, and the applied choice is recorded as a manifest note. A URL value that
contradicts a field is rejected while the reference is normalized (CONFLICTING_OPTIONS).

`open_rows` returns a `DatasetRowStream` over every record of one split. Failure signal and stop
contract (see `dataset_stream`): undecodable records arrive as `FailedSourceRow` (a `SourceRow`
with an empty `row`, `error_code`, `reason`, Korean `message` and its line/element position;
import it from the dependency-free `dataset_rows` module); quotas, broken shards, manifest
mismatches and download failures end the stream with an entry in `stream.issues`;
`stream.complete` is True only after a clean EOF of every shard with no failed record. A split
without any record is complete and reports EMPTY_DATASET in `stream.issues`.

Heavy libraries (datasets, pyarrow, pandas, huggingface_hub) are imported lazily so that importing
the package stays cheap and works without the `analysis` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
from vramforge_estimator.sources.references import normalize_dataset_reference

from .base import RowStream

if TYPE_CHECKING:
    from .dataset_layout import ConfigLayout, DatasetLayout
    from .dataset_schema import Preview
    from .dataset_stream import DatasetRowStream
    from .readers import ReaderLimits

__all__ = [
    "DatasetInspectionDetails",
    "RequestedSelection",
    "SelectionOrigin",
    "inspect_dataset",
    "inspect_dataset_details",
    "open_rows",
    "requested_selection",
]

# Where a selected config/split came from: a request field, the dataset viewer URL path, or the
# inspector's unambiguous choice (the only config, the "train" split).
SelectionOrigin = Literal["request", "viewer_url", "auto"]


@dataclass(frozen=True)
class RequestedSelection:
    """The config/split a request asks for (see `requested_selection`); None = not asked."""

    config: str | None = None
    split: str | None = None
    config_origin: SelectionOrigin | None = None  # "request" or "viewer_url" when `config` is set
    split_origin: SelectionOrigin | None = None


@dataclass
class DatasetInspectionDetails:
    """The public inspection plus what produced it (for callers that need more than the API)."""

    inspection: DatasetInspection
    layout: DatasetLayout | None = None
    preview: Preview | None = None
    file_formats: dict[str, str] = field(default_factory=dict)
    # How `inspection.selected_config` / `selected_split` were chosen (None = not selected).
    config_origin: SelectionOrigin | None = None
    split_origin: SelectionOrigin | None = None


def requested_selection(ref: DatasetSourceRef) -> RequestedSelection:
    """Request fields first; an empty field takes the viewer URL path value (`/viewer/<c>/<s>`).

    Pure (no I/O). Raises `EstimatorError` for references the resolver rejects as well, e.g.
    CONFLICTING_OPTIONS when the URL path and a field name different configs or splits.
    """
    normalized = normalize_dataset_reference(ref)
    config, config_origin = _origin(_clean(ref.config), normalized.config_hint)
    split, split_origin = _origin(_clean(ref.split), normalized.split_hint)
    return RequestedSelection(config, split, config_origin, split_origin)


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
    wanted = requested_selection(ref)
    files = SourceFiles(source, access, limits)
    layout = resolve_layout(files)  # EstimatorError: no loadable layout at all
    issues: list[Issue] = []
    configs = [config.name for config in layout.configs]
    config, config_issue = _select_config(layout, wanted.config, wanted.config_origin)
    if config_issue is not None:
        issues.append(config_issue)
    config_origin: SelectionOrigin | None = None
    if config is not None:
        config_origin = wanted.config_origin or "auto"
    splits: list[DatasetSplitInfo] = []
    selected_split: str | None = None
    split_origin: SelectionOrigin | None = None
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
        selected_split, auto_selected, split_issue = _select_split(
            config, wanted.split, wanted.split_origin
        )
        if split_issue is not None:
            issues.append(split_issue)
        if selected_split is not None:
            split_origin = "auto" if auto_selected else wanted.split_origin
        eval_issue = _check_eval_split(config, _clean(ref.eval_split), selected_split)
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
            unreadable = _request_failure_before_columns(preview)
            if unreadable is not None:
                raise EstimatorError(unreadable)
            if selected_split is not None and preview.split_is_empty:
                others = len(layout.configs) > 1 or len(config.splits) > 1
                empty = _empty_split_issue(config.name, selected_split, others)
                if not others:
                    raise EstimatorError(empty)  # nothing else the user could choose
                issues.append(empty)
    columns = preview_columns(preview) if preview is not None else []
    mapping = None
    if columns:
        mapping = analyze_mapping(columns, objective, ref.mapping)
        issues.extend(mapping.issues)
    file_formats = dict(preview.file_formats) if preview is not None else {}
    selected_config = config.name if config is not None else None
    notes = [
        *layout.notes,
        *_viewer_notes(selected_config, config_origin, selected_split, split_origin),
        *_format_notes(file_formats),
    ]
    manifest = source.manifest
    if notes:
        manifest = manifest.model_copy(update={"notes": [*manifest.notes, *notes]})
    inspection = DatasetInspection(
        manifest=manifest,
        configs=configs,
        selected_config=selected_config,
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
    return DatasetInspectionDetails(
        inspection,
        layout,
        preview,
        file_formats,
        config_origin=config_origin,
        split_origin=split_origin,
    )


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


def _clean(value: str | None) -> str | None:
    """A blank field is no choice (like the reference normalizer's optional fields)."""
    if value is None:
        return None
    return value.strip() or None


def _origin(explicit: str | None, hint: str | None) -> tuple[str | None, SelectionOrigin | None]:
    if explicit is not None:
        return explicit, "request"
    if hint is not None:
        return hint, "viewer_url"
    return None, None


def _viewer_notes(
    config: str | None,
    config_origin: SelectionOrigin | None,
    split: str | None,
    split_origin: SelectionOrigin | None,
) -> list[str]:
    """Record a selection taken from the viewer URL path (applied choices only)."""
    choices = (("config", config, config_origin), ("split", split, split_origin))
    chosen = [
        f"{kind} '{name}'"
        for kind, name, origin in choices
        if name is not None and origin == "viewer_url"
    ]
    if not chosen:
        return []
    return [f"데이터셋 viewer 주소에 지정된 {', '.join(chosen)}을(를) 분석 대상으로 선택했습니다."]


def _select_config(
    layout: DatasetLayout, requested: str | None, origin: SelectionOrigin | None = None
) -> tuple[ConfigLayout | None, Issue | None]:
    names = [config.name for config in layout.configs]
    if requested is not None:
        config = layout.config(requested)
        if config is not None:
            return config, None
        if origin == "viewer_url":
            return None, _required(
                ErrorCode.DATASET_CONFIG_REQUIRED,
                f"viewer 주소의 설정 '{requested}'이(가) 데이터셋에 없습니다. "
                "설정을 선택해 주세요.",
                field="dataset.config",
                options=names,
                suggested=layout.default_config,
                requested=requested,
                requested_from="viewer_url",
            )
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
    config: ConfigLayout, requested: str | None, origin: SelectionOrigin | None = None
) -> tuple[str | None, bool, Issue | None]:
    names = [split.name for split in config.splits]
    if requested is not None:
        if requested in names:
            return requested, False, None
        if origin == "viewer_url":
            return (
                None,
                False,
                _required(
                    ErrorCode.DATASET_SPLIT_REQUIRED,
                    f"viewer 주소의 split '{requested}'이(가) 설정 '{config.name}'에 없습니다. "
                    "분석할 split을 선택해 주세요.",
                    field="dataset.split",
                    options=names,
                    requested=requested,
                    requested_from="viewer_url",
                ),
            )
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


def _request_failure_before_columns(preview: Preview) -> Issue | None:
    """A failed Hub request (`dataset_files.remote_issue`) before any column was seen.

    Retrying (or fixing access) is then the only way forward; returning the inspection would end
    in a column-mapping question about columns nobody could read.
    """
    if preview.rows or preview.features is not None:
        return None
    return next(
        (issue for issue in preview.issues if issue.details.get("request") in ("download", "read")),
        None,
    )


def _empty_split_issue(config: str, split: str, others: bool) -> Issue:
    message = f"선택한 split '{split}'의 데이터 파일에 row가 하나도 없어 분석할 데이터가 없습니다."
    if others:
        message += " 다른 split이나 설정을 선택해 주세요."
    return _required(ErrorCode.EMPTY_DATASET, message, reason="no_rows", config=config, split=split)


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
