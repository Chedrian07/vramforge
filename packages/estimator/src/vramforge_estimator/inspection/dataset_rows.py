"""The failure signal of the dataset row stream (no heavy imports; safe for any consumer).

`DatasetRowStream` yields a `FailedSourceRow` for a record it could not decode instead of
aborting: it is a `SourceRow` (same `row_index` / `shard_id` contract) with an empty `row`, the
`ErrorCode` to record, a stable `reason` code, a Korean display-safe `message` and the record's
position in its shard. Consumers turn it into a `FailedRow`; one that does not know the subclass
still sees an empty row, which fails mapping, so a failed record is never counted as a success.
"""

from __future__ import annotations

from dataclasses import dataclass

from vramforge_estimator.schemas import ErrorCode

from .base import SourceRow


@dataclass(frozen=True)
class FailedSourceRow(SourceRow):
    """A record that could not be decoded. `row` is empty; position fields locate it."""

    error_code: ErrorCode = ErrorCode.SCAN_FAILED_ROWS
    reason: str = "json_parse"  # json_parse | not_an_object | schema_mismatch | invalid_utf8
    message: str = ""  # Korean, display-safe (no raw data)
    line: int | None = None  # 1-based line in the (decompressed) shard
    element: int | None = None  # 0-based element index in a JSON array shard


def is_failed_row(source_row: SourceRow) -> bool:
    return isinstance(source_row, FailedSourceRow)
