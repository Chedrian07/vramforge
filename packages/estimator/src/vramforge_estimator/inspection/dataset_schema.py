"""Schema preview of one split: at most `ReaderLimits.preview_rows` rows read within a byte budget
(`SourceAccess.max_metadata_bytes`), never the full scan (plan §7.1). Columns get an Arrow dtype
string and a coarse kind used for mapping detection:

- `string`: every non-null value is a str
- `messages`: every non-null value is a non-empty list of dicts with a `role` (or ShareGPT `from`)
  key — the shape TRL treats as conversational (`trl.data_utils.is_conversational`)
- `number`: int/float (not bool)
- `other`: anything else; all-null columns fall back to their Arrow type
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import DatasetColumn, ErrorCode, Issue, Severity, Stage

from .dataset_files import SourceFiles
from .dataset_layout import ConfigLayout, SplitLayout
from .readers import (
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
    reason_message,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from datasets import Features

logger = logging.getLogger(__name__)
_PREVIEW_FILES = 4  # files tried when the first ones have no rows (one shared byte budget)


@dataclass
class Preview:
    rows: list[dict[str, Any]] = field(default_factory=list)
    features: Features | None = None
    file_formats: dict[str, str] = field(default_factory=dict)
    failed_rows: int = 0
    issues: list[Issue] = field(default_factory=list)


def read_preview(
    files: SourceFiles,
    config: ConfigLayout,
    split: SplitLayout,
    budget: int,
    limits: ReaderLimits,
) -> Preview:
    """Up to `limits.preview_rows` rows from the first files of `split`, `budget` bytes in total."""
    preview = Preview()
    schema = SchemaState()
    remaining = budget
    quota_hit = False
    for data_file in split.files[:_PREVIEW_FILES]:
        if remaining <= 0:
            break
        info = ReadInfo()
        source = None
        try:
            source = files.preview_source(data_file, remaining)
            spec = FileSpec(
                module=config.module,  # type: ignore[arg-type]
                options=config.options,
                compression=compression_of(source),
            )
            for item in iter_file(source, spec, limits, schema, info):
                if isinstance(item, RowError):
                    preview.failed_rows += 1
                    continue
                preview.rows.append(item)
                if len(preview.rows) >= limits.preview_rows:
                    break
        except QuotaExceeded:
            quota_hit = True  # the budget only bounds the preview; what was read stays valid
        except (ShardBroken, UnsupportedFormat) as exc:
            preview.issues.append(_preview_problem(data_file.shard_id, exc.reason))
            break
        except EstimatorError as exc:
            preview.issues.append(exc.issue)
            break
        except Exception as exc:  # unexpected library error: report, keep the inspection alive
            logger.error("unexpected %s in preview of %s", type(exc).__name__, data_file.shard_id)
            preview.issues.append(
                make_issue(
                    ErrorCode.INTERNAL_ERROR,
                    f"데이터 파일 {data_file.shard_id}을(를) 미리보기로 읽는 중 예기치 못한 오류가 "
                    "발생했습니다.",
                    stage=Stage.INSPECTING,
                    component="dataset",
                    error_type=type(exc).__name__,
                    shard_id=data_file.shard_id,
                )
            )
            break
        finally:
            if info.file_format is not None:
                preview.file_formats[data_file.shard_id] = info.file_format
            if source is not None:
                remaining -= source.bytes_read
        if len(preview.rows) >= limits.preview_rows or quota_hit:
            break
    preview.features = schema.features
    if preview.failed_rows:
        preview.issues.append(
            make_issue(
                ErrorCode.SCAN_FAILED_ROWS,
                f"미리보기에서 읽을 수 없는 레코드 {preview.failed_rows}개를 발견했습니다. "
                "전체 스캔에서 위치와 함께 집계됩니다.",
                severity=Severity.WARNING,
                stage=Stage.INSPECTING,
                component="dataset",
                rows_failed=preview.failed_rows,
            )
        )
    if quota_hit and not preview.rows and preview.features is None:
        preview.issues.append(
            make_issue(
                ErrorCode.SCAN_QUOTA_EXCEEDED,
                "미리보기 크기 한도 안에서 컬럼을 확인하지 못했습니다. "
                "첫 레코드나 파일 메타데이터가 한도보다 큽니다.",
                severity=Severity.WARNING,
                stage=Stage.INSPECTING,
                component="dataset",
                limit="max_metadata_bytes",
                limit_value=budget,
            )
        )
    return preview


def preview_columns(preview: Preview) -> list[DatasetColumn]:
    names: list[str] = list(preview.features) if preview.features is not None else []
    for row in preview.rows:
        names.extend(name for name in row if name not in names)
    arrow_schema = preview.features.arrow_schema if preview.features is not None else None
    columns = []
    for name in names:
        arrow_type = (
            arrow_schema.field(name).type
            if arrow_schema is not None and name in arrow_schema.names
            else None
        )
        values = [row.get(name) for row in preview.rows]
        columns.append(
            DatasetColumn(
                name=name,
                dtype=arrow_dtype(arrow_type) if arrow_type is not None else _dtype_of(values),
                kind=value_kind(values, arrow_type),
            )
        )
    return columns


def value_kind(values: list[Any], arrow_type: pa.DataType | None = None) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return _type_kind(arrow_type) if arrow_type is not None else "other"
    if all(isinstance(value, str) for value in present):
        return "string"
    if all(is_messages(value) for value in present):
        return "messages"
    if all(isinstance(value, int | float) and not isinstance(value, bool) for value in present):
        return "number"
    return "other"


def is_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(message, Mapping)
            and isinstance(message.get("role", message.get("from")), str)
            for message in value
        )
    )


def arrow_dtype(arrow_type: pa.DataType) -> str:
    """Compact Arrow type name; datasets' Json extension type reads as `json`."""
    import pyarrow as pa

    if isinstance(arrow_type, pa.JsonType):
        return "json"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return f"list<{arrow_dtype(arrow_type.value_type)}>"
    if pa.types.is_struct(arrow_type):
        inner = ", ".join(f"{f.name}: {arrow_dtype(f.type)}" for f in arrow_type)
        return f"struct<{inner}>"
    return str(arrow_type)


def _type_kind(arrow_type: pa.DataType) -> str:
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
        return "number"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        item = arrow_type.value_type
        if pa.types.is_struct(item) and ("role" in item.names or "from" in item.names):
            return "messages"
    return "other"


def _dtype_of(values: list[Any]) -> str:
    present = {type(value).__name__ for value in values if value is not None}
    if not present:
        return "null"
    if len(present) > 1:
        return "mixed"
    return {"str": "string", "int": "int64", "float": "double", "bool": "bool"}.get(
        present.pop(), "object"
    )


def _preview_problem(shard_id: str, reason: str) -> Issue:
    return make_issue(
        ErrorCode.DATASET_FORMAT_UNSUPPORTED,
        f"데이터 파일 {shard_id}을(를) 미리보기로 읽을 수 없습니다. {reason_message(reason)}",
        severity=Severity.ERROR,
        stage=Stage.INSPECTING,
        component="dataset",
        reason=reason,
        shard_id=shard_id,
    )
