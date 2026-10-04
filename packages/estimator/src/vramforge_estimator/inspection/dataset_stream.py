"""The sequential RowStream over one split (plan.md §7.6, §7.7, §19.2).

Iteration yields every record of every shard, shards in datasets' file order and records in file
order. `row_index` counts records within the split (0-based) and equals the index of the row in
`datasets.load_dataset(...)[split]` whenever datasets can load the split.

Failure signal: a record that cannot be decoded (e.g. a malformed JSON Lines line) is yielded as a
`FailedSourceRow` — a `SourceRow` subclass with an empty `row` plus `error_code`, `reason` and a
Korean `message` — at its own `row_index`, and reading continues. The scanner turns it into a
`FailedRow`; a consumer that ignores the subclass still fails the row (no columns), never counts
it as a success.

Stop conditions: a quota (`SCAN_QUOTA_EXCEEDED`), a structurally broken shard (`SCAN_PARTIAL`), a
file that differs from the manifest (`SOURCE_REVISION_CHANGED`), an unsupported file
(`DATASET_FORMAT_UNSUPPORTED`) or a download failure end the stream; later shards are not read so
row indexes never shift. The reason is appended to `issues`.

`complete` is True only after every shard reached EOF normally, no record failed, every shard
matched the manifest and the row count matched the split metadata when the dataset declares one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Issue, Severity, Stage

from .base import SourceRow
from .dataset_files import DataFile, SourceFiles
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

logger = logging.getLogger(__name__)
_FAILED_SAMPLE = 20

_ROW_MESSAGES = {
    "json_parse": "JSON으로 해석할 수 없는 레코드입니다",
    "not_an_object": "JSON 객체가 아닌 레코드입니다",
    "schema_mismatch": "다른 row와 컬럼 형식이 맞지 않는 레코드입니다",
    "invalid_utf8": "UTF-8로 해석할 수 없는 레코드입니다",
}


@dataclass(frozen=True)
class FailedSourceRow(SourceRow):
    """A record that could not be decoded. `row` is empty; position fields locate it."""

    error_code: ErrorCode = ErrorCode.SCAN_FAILED_ROWS
    reason: str = "json_parse"
    message: str = ""  # Korean, display-safe (no raw data)
    line: int | None = None  # 1-based line in the (decompressed) shard
    element: int | None = None  # 0-based element index in a JSON array shard


def is_failed_row(source_row: SourceRow) -> bool:
    return isinstance(source_row, FailedSourceRow)


class DatasetRowStream:
    """`RowStream` implementation (see module docstring for the failure and stop contract)."""

    def __init__(
        self,
        files: SourceFiles,
        config: ConfigLayout,
        split: SplitLayout,
        limits: ReaderLimits,
    ) -> None:
        self.split = split.name
        self.config: str | None = config.name
        self.shards = [data_file.shard_id for data_file in split.files]
        self.total_rows = (
            split.num_rows if split.num_rows is not None else footer_rows(files, split)
        )
        self.issues: list[Issue] = []
        self.file_formats: dict[str, str] = {}
        self.rows_ok = 0
        self.rows_failed = 0
        self._files = files
        self._config = config
        self._split = split
        self._limits = limits
        self._complete = False
        self._shards_completed = 0

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def shards_completed(self) -> int:
        return self._shards_completed

    @property
    def rows_seen(self) -> int:
        return self.rows_ok + self.rows_failed

    def __iter__(self) -> Iterator[SourceRow]:
        self.issues = []
        self.file_formats = {}
        self.rows_ok = self.rows_failed = 0
        self._complete = False
        self._shards_completed = 0
        if self._config.unsupported is not None:
            self.issues.append(self._config.unsupported)
            return
        schema = SchemaState()
        failed_sample: list[dict[str, object]] = []
        stopped = False
        for data_file in self._split.files:
            try:
                yield from self._read_shard(data_file, schema, failed_sample)
            except _StopStream:
                stopped = True
                break
            self._shards_completed += 1
        if self.rows_failed:
            self.issues.append(
                _issue(
                    ErrorCode.SCAN_FAILED_ROWS,
                    f"읽을 수 없는 레코드 {self.rows_failed}개가 있습니다. "
                    "위치는 실패 row 목록에 보존됩니다.",
                    severity=Severity.WARNING,
                    rows_failed=self.rows_failed,
                    sample=failed_sample,
                )
            )
        if stopped:
            return
        if self.total_rows is not None and self.rows_seen != self.total_rows:
            self.issues.append(
                _issue(
                    ErrorCode.SCAN_PARTIAL,
                    f"데이터셋 메타데이터의 row 수({self.total_rows})와 실제로 읽은 row 수"
                    f"({self.rows_seen})가 다릅니다. datasets로 불러올 때 split 크기 검증에 "
                    "실패할 수 있습니다.",
                    reason="row_count_mismatch",
                    rows_expected=self.total_rows,
                    rows_seen=self.rows_seen,
                )
            )
            return
        self._complete = self.rows_failed == 0

    def _read_shard(
        self, data_file: DataFile, schema: SchemaState, failed_sample: list[dict[str, object]]
    ) -> Iterator[SourceRow]:
        shard_id = data_file.shard_id
        try:
            source = self._files.scan_source(data_file)
            assert source.path is not None
            before = self._files.signature(source.path)
            problem = self._files.verify(data_file, source.path)
            if problem is not None:
                self._stop(problem)
            spec = FileSpec(
                module=self._config.module,  # type: ignore[arg-type]
                options=self._config.options,
                compression=compression_of(source),
            )
            info = ReadInfo()
            logger.info("reading shard %s (%s)", shard_id, spec.module)
            for item in iter_file(source, spec, self._limits, schema, info):
                index = self.rows_seen
                if isinstance(item, RowError):
                    self.rows_failed += 1
                    if len(failed_sample) < _FAILED_SAMPLE:
                        failed_sample.append(
                            {
                                "row_index": index,
                                "shard_id": shard_id,
                                "line": item.line,
                                "element": item.element,
                                "reason": item.reason,
                            }
                        )
                    yield _failed_row(index, shard_id, item)
                    if self.rows_failed > self._limits.max_failed_rows:
                        self._stop(
                            _issue(
                                ErrorCode.SCAN_PARTIAL,
                                "읽을 수 없는 레코드가 너무 많아 데이터 읽기를 중단했습니다.",
                                reason="too_many_failed_rows",
                                limit=self._limits.max_failed_rows,
                                shard_id=shard_id,
                            )
                        )
                else:
                    self.rows_ok += 1
                    yield SourceRow(row_index=index, shard_id=shard_id, row=item)
            if info.file_format is not None:
                self.file_formats[shard_id] = info.file_format
            if self._files.kind != "hf" and self._files.signature(source.path) != before:
                self._stop(
                    make_issue(
                        ErrorCode.SOURCE_REVISION_CHANGED,
                        "분석 중에 데이터 파일이 바뀌었습니다. 다시 분석해 주세요.",
                        stage=Stage.TOKENIZING,
                        component="dataset",
                        reason="changed_while_reading",
                        shard_id=shard_id,
                    )
                )
            logger.info("shard %s done: %d rows", shard_id, info.rows + info.errors)
        except QuotaExceeded as exc:
            self._stop(
                _issue(
                    ErrorCode.SCAN_QUOTA_EXCEEDED,
                    "자원 한도를 넘어 데이터 읽기를 중단했습니다. 이후 row는 분석되지 않았습니다.",
                    limit=exc.limit,
                    limit_value=exc.value,
                    shard_id=shard_id,
                    rows_read=self.rows_seen,
                    **_positions(exc.detail),
                )
            )
        except ShardBroken as exc:
            self._stop(
                _issue(
                    ErrorCode.SCAN_PARTIAL,
                    f"데이터 파일 {shard_id}을(를) 끝까지 읽을 수 없어 중단했습니다. "
                    f"{reason_message(exc.reason)}",
                    reason=exc.reason,
                    shard_id=shard_id,
                    rows_read=self.rows_seen,
                )
            )
        except UnsupportedFormat as exc:
            self._stop(
                _issue(
                    ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                    f"데이터 파일 {shard_id}을(를) 읽지 않았습니다. {reason_message(exc.reason)}",
                    reason=exc.reason,
                    shard_id=shard_id,
                )
            )
        except EstimatorError as exc:
            self._stop(exc.issue)

    def _stop(self, issue: Issue) -> None:
        self.issues.append(issue)
        logger.warning("dataset stream stopped: %s %s", issue.code, issue.details.get("reason"))
        raise _StopStream


class _StopStream(Exception):
    pass


def _failed_row(index: int, shard_id: str, error: RowError) -> FailedSourceRow:
    where = (
        f"{error.line}번째 줄"
        if error.line is not None
        else f"{error.element}번째 요소"
        if error.element is not None
        else "위치 미상"
    )
    message = (
        f"{_ROW_MESSAGES.get(error.reason, '읽을 수 없는 레코드입니다')} ({shard_id}, {where})"
    )
    return FailedSourceRow(
        row_index=index,
        shard_id=shard_id,
        row={},
        reason=error.reason,
        message=message,
        line=error.line,
        element=error.element,
    )


def _positions(detail: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in detail.items() if key in ("line", "element", "row_group")}


def _issue(
    code: ErrorCode, message: str, *, severity: Severity = Severity.ERROR, **details: object
) -> Issue:
    return make_issue(
        code, message, severity=severity, stage=Stage.TOKENIZING, component="dataset", **details
    )


def footer_rows(files: SourceFiles, split: SplitLayout) -> int | None:
    """Exact row count from local Parquet footers (cheap); None for anything else."""
    if files.kind == "hf" or not split.files:
        return None
    if not all(f.shard_id.endswith(".parquet") for f in split.files):
        return None
    import pyarrow.parquet as pq

    total = 0
    try:
        for data_file in split.files:
            total += pq.read_metadata(Path(data_file.location)).num_rows
    except (OSError, ValueError):
        return None
    return total
