"""Full dataset scan (plan.md §7.6, §7.7, §16.2, §18).

One sequential pass over the `RowStream`. Only integer lengths are kept in memory (exact
`LengthAccumulator` counts); per-row `LengthRecord` columns go to Parquet part files under
``ctx.artifact_dir/lengths/``. After every part the checkpoint (last processed ``row_index`` +
accumulators) is saved; a resumed scan re-reads the stream and skips processed indices.
Coverage (plan §7.7, §19.2 "손상 row"):

* COMPLETE — the reader confirmed EOF on every shard, the manifest row count matches and every
  row was tokenized.
* PARTIAL — rows were read but the maximum is not guaranteed: a limit, a read error or an
  unconfirmed stream left rows unread, or some rows could not be read/tokenized (their lengths
  are unknown even when every row was read; SCAN_FAILED_ROWS says how many were read and failed).
* FAILED — no row was read at all; a confirmed empty split adds EMPTY_DATASET.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vramforge_estimator.errors import CancelledError, EstimatorError, make_issue
from vramforge_estimator.inspection import RowStream, SourceRow
from vramforge_estimator.preprocessing import PreprocessingAdapter, TokenizedRecord
from vramforge_estimator.schemas import (
    Branch,
    BranchStats,
    ColumnMapping,
    DatasetScanResult,
    ErrorCode,
    FailedRow,
    Issue,
    JobProgress,
    JobStatus,
    Objective,
    ScanCoverage,
    Severity,
    ShardProgress,
    Stage,
)

from .artifact import (
    COLUMNS,
    LENGTHS_DIR,
    SCHEMA_ID,
    iter_columns,
    part_paths,
    remove_parts,
    write_manifest,
    write_part,
)
from .base import LengthTable, ScanContext, ScanLimits, ScanOutcome
from .stats import BRANCHES, LengthAccumulator, branch_values, primary_branch

CHECKPOINT_KIND = "vramforge.scan"
CHECKPOINT_VERSION = 1
FAILED_SAMPLE_LIMIT = 100
ROW_ID_SAMPLE_LIMIT = 20
CANCEL_CHECK_ROWS = 16

# Monotonic clock; tests may replace it.
clock: Callable[[], float] = time.monotonic

_DIAGNOSTICS = ("template_loss", "prefix_mismatch", "special_token", "system_omitted")
_FINAL_MESSAGES = {
    ScanCoverage.COMPLETE: "데이터셋 전체 토큰화를 마쳤습니다.",
    ScanCoverage.PARTIAL: "데이터셋을 끝까지 확인하지 못했습니다. 통계는 처리한 row까지의 값이며 "
    "전체 최대 길이가 아닙니다.",
    ScanCoverage.FAILED: "데이터를 읽지 못해 토큰화한 row가 없습니다.",
    ScanCoverage.NOT_STARTED: "데이터셋 토큰화를 시작하지 않았습니다.",
}
_FAILED_ROWS_MESSAGE = (
    "모든 row를 읽었지만 {failed:,}개 row를 읽거나 토큰화하지 못했습니다. 통계는 처리한 row의 "
    "값이며 전체 최대 길이로 보장되지 않습니다."
)
_EMPTY_MESSAGE = "선택한 split에 row가 없어 토큰화한 row가 없습니다."
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def _safe_shard(shard_id: str) -> str:
    """Never let a host absolute path into results or artifacts."""
    if os.path.isabs(shard_id) or _WINDOWS_ABS.match(shard_id):
        return Path(shard_id.replace("\\", "/")).name
    return shard_id


def _row_chars(value: Any, budget: int) -> int:
    """Characters in a raw row (strings anywhere in it); stops counting once over `budget`."""
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str | bytes):
            total += len(item)
            if total > budget:
                return total
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list | tuple):
            stack.extend(item)
    return total


@dataclass
class _State:
    last_row_index: int = -1
    rows_seen: int = 0
    rows_ok: int = 0
    rows_failed: int = 0
    context_exceeded_rows: int = 0
    elapsed: float = 0.0
    parts: list[str] = field(default_factory=list)
    accumulators: dict[Branch, LengthAccumulator] = field(default_factory=dict)
    failed_sample: list[dict[str, Any]] = field(default_factory=list)
    diag_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_DIAGNOSTICS, 0))
    diag_samples: dict[str, list[str]] = field(
        default_factory=lambda: {k: [] for k in _DIAGNOSTICS}
    )

    def note(self, kind: str, row_id: str) -> None:
        self.diag_counts[kind] += 1
        if len(self.diag_samples[kind]) < ROW_ID_SAMPLE_LIMIT:
            self.diag_samples[kind].append(row_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_row_index": self.last_row_index,
            "rows_seen": self.rows_seen,
            "rows_ok": self.rows_ok,
            "rows_failed": self.rows_failed,
            "context_exceeded_rows": self.context_exceeded_rows,
            "elapsed": self.elapsed,
            "parts": list(self.parts),
            "accumulators": {b.value: a.to_state() for b, a in self.accumulators.items()},
            "failed_sample": list(self.failed_sample),
            "diag_counts": dict(self.diag_counts),
            "diag_samples": {k: list(v) for k, v in self.diag_samples.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _State:
        return cls(
            last_row_index=int(data["last_row_index"]),
            rows_seen=int(data["rows_seen"]),
            rows_ok=int(data["rows_ok"]),
            rows_failed=int(data["rows_failed"]),
            context_exceeded_rows=int(data["context_exceeded_rows"]),
            elapsed=float(data["elapsed"]),
            parts=[str(p) for p in data["parts"]],
            accumulators={
                Branch(b): LengthAccumulator.from_state(s) for b, s in data["accumulators"].items()
            },
            failed_sample=list(data["failed_sample"]),
            diag_counts={k: int(data["diag_counts"].get(k, 0)) for k in _DIAGNOSTICS},
            diag_samples={k: list(data["diag_samples"].get(k, [])) for k in _DIAGNOSTICS},
        )


@dataclass(frozen=True)
class _Ending:
    """How reading ended; decides coverage and which issues explain it."""

    stop_reason: str | None  # our own limit: "max_rows" | "max_seconds"
    read_error: BaseException | None
    read_all: bool  # the reader confirmed EOF on every shard and the manifest count matches
    manifest_mismatch: bool
    unprocessed: int | None  # rows known to exist that were never read; None = unknown
    rows_seen: int
    rows_failed: int

    @property
    def all_rows_seen(self) -> bool:
        """Every row of the split was read, even if the reader did not call the stream complete
        (e.g. it flags undecodable records): known only when the manifest row count is known."""
        finished = self.stop_reason is None and self.read_error is None
        return self.read_all or (finished and not self.manifest_mismatch and self.unprocessed == 0)

    @property
    def empty(self) -> bool:
        return self.read_all and self.rows_seen == 0

    @property
    def coverage(self) -> ScanCoverage:
        if self.rows_seen == 0:
            return ScanCoverage.FAILED
        if self.read_all and self.rows_failed == 0:
            return ScanCoverage.COMPLETE
        return ScanCoverage.PARTIAL

    def final_message(self) -> str:
        coverage = self.coverage
        if coverage is ScanCoverage.FAILED and self.empty:
            return _EMPTY_MESSAGE
        if coverage is ScanCoverage.PARTIAL and self.rows_failed and self.all_rows_seen:
            return _FAILED_ROWS_MESSAGE.format(failed=self.rows_failed)
        return _FINAL_MESSAGES[coverage]


def _ending(
    state: _State, stream: RowStream, stop_reason: str | None, read_error: BaseException | None
) -> _Ending:
    reader_complete = stop_reason is None and read_error is None and stream.complete
    total = stream.total_rows
    manifest_mismatch = reader_complete and total is not None and total != state.rows_seen
    read_all = reader_complete and not manifest_mismatch
    if read_all:
        unprocessed: int | None = 0
    elif total is not None:
        unprocessed = max(0, total - state.rows_seen)
    else:
        unprocessed = None
    return _Ending(
        stop_reason=stop_reason,
        read_error=read_error,
        read_all=read_all,
        manifest_mismatch=manifest_mismatch,
        unprocessed=unprocessed,
        rows_seen=state.rows_seen,
        rows_failed=state.rows_failed,
    )


def _identity(
    stream: RowStream,
    adapter: PreprocessingAdapter,
    *,
    objective: Objective,
    preprocess_key: str,
    tokenizer_fingerprint: str,
    template_fingerprint: str | None,
    context_limit: int | None,
) -> dict[str, Any]:
    return {
        "kind": CHECKPOINT_KIND,
        "version": CHECKPOINT_VERSION,
        "preprocess_key": preprocess_key,
        "split": stream.split,
        "config": stream.config,
        "objective": objective.value,
        "adapter": adapter.name,
        "adapter_version": adapter.version,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "template_fingerprint": template_fingerprint,
        "context_limit": context_limit,
    }


def _restore(ctx: ScanContext, identity: dict[str, Any], lengths_dir: Path) -> _State:
    """Resume from a matching checkpoint; otherwise start clean (stale parts are removed)."""
    data = ctx.load_checkpoint()
    if data and all(data.get(k) == v for k, v in identity.items()):
        try:
            state = _State.from_dict(data["state"])
        except (KeyError, TypeError, ValueError):
            state = None
        if state is not None and all((lengths_dir / p).is_file() for p in state.parts):
            remove_parts(lengths_dir, keep=set(state.parts))  # orphans after the last checkpoint
            return state
    remove_parts(lengths_dir)
    return _State()


class _RowBuffer:
    def __init__(self) -> None:
        self.columns: dict[str, list[Any]] = {c: [] for c in COLUMNS}

    def __len__(self) -> int:
        return len(self.columns["row_id"])

    def append(self, values: dict[str, Any]) -> None:
        for c in COLUMNS:
            self.columns[c].append(values.get(c))

    def clear(self) -> None:
        for values in self.columns.values():
            values.clear()


def full_scan(
    stream: RowStream,
    adapter: PreprocessingAdapter,
    ctx: ScanContext,
    *,
    objective: Objective,
    preprocess_key: str,
    tokenizer_fingerprint: str,
    template_fingerprint: str | None,
    context_limit: int | None,
) -> ScanOutcome:
    """Tokenize every row of the stream, write the Parquet row-length artifact under
    `ctx.artifact_dir`, compute exact counts/max and statistics, and resume from the checkpoint
    when present.

    Coverage is COMPLETE only when the reader confirmed every shard, the manifest count matches
    and every row was tokenized. Rows that could not be read or tokenized make it PARTIAL even
    when every row was read: their lengths are unknown, so the maximum is not guaranteed and the
    complete badge is withheld (plan §19.2); SCAN_FAILED_ROWS states how many rows were read and
    how many failed. A scan that read no row is FAILED, with EMPTY_DATASET for a confirmed empty
    split."""
    if adapter.objective is not objective:
        raise EstimatorError(
            make_issue(
                ErrorCode.CONFLICTING_OPTIONS,
                "전처리 어댑터의 학습 방식이 요청과 다릅니다.",
                stage=Stage.TOKENIZING,
                adapter_objective=str(adapter.objective),
                objective=objective.value,
            )
        )
    limits = ctx.limits
    lengths_dir = ctx.artifact_dir / LENGTHS_DIR
    identity = _identity(
        stream,
        adapter,
        objective=objective,
        preprocess_key=preprocess_key,
        tokenizer_fingerprint=tokenizer_fingerprint,
        template_fingerprint=template_fingerprint,
        context_limit=context_limit,
    )
    state = _restore(ctx, identity, lengths_dir)
    for branch, _getter in BRANCHES[objective]:
        state.accumulators.setdefault(branch, LengthAccumulator())
    metadata = {"vramforge.schema": SCHEMA_ID, "vramforge.preprocess_key": preprocess_key}
    constant = {
        "split": stream.split,
        "objective": objective.value,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "template_fingerprint": template_fingerprint,
    }
    buffer = _RowBuffer()
    started = clock()
    elapsed_before = state.elapsed
    last_report = float("-inf")
    since_cancel_check = CANCEL_CHECK_ROWS  # check before the first row

    def elapsed() -> float:
        return elapsed_before + (clock() - started)

    def report(*, coverage: ScanCoverage | None = None, final_message: str = "") -> None:
        """Progress while scanning; the final report (coverage given) never calls a partial or
        failed scan finished (plan §7.7, §19.2)."""
        nonlocal last_report
        last_report = clock()
        partial: dict[str, Any] = {
            "status": coverage.value if coverage else "partial",
            "rows_ok": state.rows_ok,
            "rows_failed": state.rows_failed,
        }
        for branch, acc in state.accumulators.items():
            if acc.max_value is not None:
                partial[f"max_{branch.value}"] = acc.max_value
        if coverage is None:
            code, message = "scan_progress", "데이터셋 전체를 토큰화하는 중입니다."
        else:
            code, message = f"scan_{coverage.value}", final_message or _FINAL_MESSAGES[coverage]
        ctx.report(
            JobProgress(
                stage=JobStatus.TOKENIZING,
                processed_rows=state.rows_seen,
                total_rows=stream.total_rows,
                shard_progress=ShardProgress(
                    completed=stream.shards_completed, total=len(stream.shards) or None
                ),
                message_code=code,
                message=message,
            ),
            partial,
        )

    def flush() -> None:
        if not len(buffer):
            return
        state.parts.append(write_part(lengths_dir, len(state.parts), buffer.columns, metadata))
        buffer.clear()
        state.elapsed = elapsed()
        ctx.save_checkpoint({**identity, "state": state.to_dict()})

    def check_cancelled() -> None:
        nonlocal since_cancel_check
        since_cancel_check += 1
        if since_cancel_check >= CANCEL_CHECK_ROWS:
            since_cancel_check = 0
            if ctx.cancelled():
                raise CancelledError(Stage.TOKENIZING)

    stop_reason: str | None = None
    read_error: BaseException | None = None
    iterator: Iterator[SourceRow] = iter(stream)
    report()
    while True:
        check_cancelled()
        try:
            src = next(iterator)
        except StopIteration:
            break
        except CancelledError:
            raise
        except Exception as exc:  # reader failure: keep what was verified so far (PARTIAL)
            read_error = exc
            break
        if src.row_index <= state.last_row_index:
            continue  # processed before the checkpoint we resumed from
        if state.rows_seen >= limits.max_rows:
            stop_reason = "max_rows"
            break
        if elapsed() >= limits.max_seconds:
            stop_reason = "max_seconds"
            break
        rec = _tokenize(adapter, src, f"{stream.split}:{src.row_index}", limits.max_record_chars)
        buffer.append({**constant, **_account(state, rec, src, objective, context_limit)})
        if len(buffer) >= limits.checkpoint_every_rows:
            flush()
        if clock() - last_report >= limits.progress_interval_s:
            report()
    flush()
    state.elapsed = elapsed()

    ending = _ending(state, stream, stop_reason, read_error)
    coverage = ending.coverage

    paths = [lengths_dir / p for p in state.parts]
    duplicates = _count_duplicates(paths)
    write_manifest(
        lengths_dir,
        {
            "schema": SCHEMA_ID,
            "preprocess_key": preprocess_key,
            "objective": objective.value,
            "split": stream.split,
            "parts": state.parts,
            "rows_seen": state.rows_seen,
            "rows_ok": state.rows_ok,
            "rows_failed": state.rows_failed,
            "coverage": coverage.value,
        },
    )
    report(coverage=coverage, final_message=ending.final_message())

    issues = _issues(state, stream, ending, limits)
    mapping = getattr(adapter, "mapping", None)
    note = adapter.transformation_note()
    omitted = state.diag_counts["system_omitted"]
    if omitted:
        note += f" 빈 system 값 때문에 {omitted:,}개 row에서 system 메시지를 생략했습니다."
    result = DatasetScanResult(
        coverage=coverage,
        objective=objective,
        config=stream.config,
        split=stream.split,
        mapping_applied=mapping if isinstance(mapping, ColumnMapping) else None,
        transformation_note=note,
        rows_expected=stream.total_rows,
        rows_seen=state.rows_seen,
        rows_ok=state.rows_ok,
        rows_failed=state.rows_failed,
        rows_unprocessed=ending.unprocessed,
        shards_total=len(stream.shards) or None,
        shards_completed=stream.shards_completed,
        branches=_branch_stats(state, objective),
        failed_rows_sample=[FailedRow.model_validate(f) for f in state.failed_sample],
        duplicate_rows=duplicates,
        context_exceeded_rows=state.context_exceeded_rows,
        preprocess_key=preprocess_key,
        artifact_id=LENGTHS_DIR,
        tokenizer_fingerprint=tokenizer_fingerprint,
        template_fingerprint=template_fingerprint,
        preprocessing_adapter=adapter.name,
        preprocessing_adapter_version=adapter.version,
        elapsed_seconds=state.elapsed,
    )
    return ScanOutcome(
        result=result,
        artifact_path=lengths_dir,
        issues=issues,
        template_content_loss_rows=state.diag_counts["template_loss"],
    )


def _source_failure(src: SourceRow) -> tuple[ErrorCode, str] | None:
    """A record the reader could not decode (e.g. inspection's FailedSourceRow): it carries its
    own error code and display-safe Korean message and has no columns to tokenize."""
    code = getattr(src, "error_code", None)
    if not isinstance(code, ErrorCode):
        return None
    message = getattr(src, "message", None)
    return code, message if isinstance(message, str) and message else "레코드를 읽지 못했습니다."


def _tokenize(
    adapter: PreprocessingAdapter, src: SourceRow, row_id: str, max_chars: int
) -> TokenizedRecord:
    failure = _source_failure(src)
    if failure is not None:
        return TokenizedRecord(
            row_id=row_id,
            objective=adapter.objective,
            ok=False,
            error_code=failure[0],
            error_message=failure[1],
        )
    if _row_chars(src.row, max_chars) > max_chars:
        return TokenizedRecord(
            row_id=row_id,
            objective=adapter.objective,
            ok=False,
            error_code=ErrorCode.SCAN_QUOTA_EXCEEDED,
            error_message=f"row 크기가 서비스 한도({max_chars:,}자)를 넘어 처리하지 않았습니다.",
        )
    try:
        return adapter.process(src.row, row_id)
    except Exception as exc:  # adapters return failures; this guards third-party adapters
        return TokenizedRecord(
            row_id=row_id,
            objective=adapter.objective,
            ok=False,
            error_code=ErrorCode.INTERNAL_ERROR,
            error_message=f"전처리 중 예기치 않은 오류가 발생했습니다 ({type(exc).__name__}).",
        )


def _account(
    state: _State,
    rec: TokenizedRecord,
    src: SourceRow,
    objective: Objective,
    context_limit: int | None,
) -> dict[str, Any]:
    """Update counters/accumulators for one row and return its per-row LengthRecord values."""
    row_id = rec.row_id
    shard = _safe_shard(src.shard_id)
    state.rows_seen += 1
    state.last_row_index = src.row_index
    context_status = "unknown"
    if rec.ok:
        state.rows_ok += 1
        for branch, value in branch_values(objective, rec):
            state.accumulators[branch].add(value, src.row_index, row_id)
        longest = rec.max_sequence_tokens()
        if context_limit is not None and longest is not None:
            context_status = "exceeded" if longest > context_limit else "ok"
            if context_status == "exceeded":
                state.context_exceeded_rows += 1
        if rec.template_preserved is False:
            state.note("template_loss", row_id)
        if rec.extras.get("prefix_mismatch"):
            state.note("prefix_mismatch", row_id)
        if rec.extras.get("duplicate_bos") or rec.extras.get("duplicate_eos"):
            state.note("special_token", row_id)
        if rec.extras.get("system_omitted"):
            state.note("system_omitted", row_id)
    else:
        state.rows_failed += 1
        if len(state.failed_sample) < FAILED_SAMPLE_LIMIT:
            state.failed_sample.append(
                {
                    "row_id": row_id,
                    "shard_id": shard,
                    "error_code": (rec.error_code or ErrorCode.INTERNAL_ERROR).value,
                    "message": rec.error_message or "전처리에 실패했습니다.",
                }
            )
    return {
        "row_id": row_id,
        "row_index": src.row_index,
        "shard_id": shard,
        "prompt_tokens": rec.prompt_tokens,
        "completion_tokens": rec.completion_tokens,
        "sequence_tokens": rec.sequence_tokens,
        "loss_token_count": rec.loss_token_count,
        "chosen_total_tokens": rec.chosen_total_tokens,
        "rejected_total_tokens": rec.rejected_total_tokens,
        "chosen_completion_tokens": rec.chosen_completion_tokens,
        "rejected_completion_tokens": rec.rejected_completion_tokens,
        "content_digest": rec.content_digest,
        "processing_status": "ok" if rec.ok else "failed",
        "context_status": context_status,
        "error_code": rec.error_code.value if rec.error_code else None,
    }


def _count_duplicates(paths: list[Path]) -> int:
    """Rows whose mapped record repeats an earlier ok row (first occurrence not counted)."""
    seen: set[bytes] = set()
    ok = 0
    for cols in iter_columns(paths, ["content_digest", "processing_status"]):
        for digest, status in zip(cols["content_digest"], cols["processing_status"], strict=True):
            if status == "ok" and digest:
                ok += 1
                seen.add(bytes.fromhex(digest[:32]))  # 128-bit prefix
    return ok - len(seen)


def _branch_stats(state: _State, objective: Objective) -> list[BranchStats]:
    primary = primary_branch(objective)
    out = []
    for branch, _getter in BRANCHES[objective]:
        acc = state.accumulators[branch]
        if acc.n or branch is primary:
            out.append(BranchStats(branch=branch, stats=acc.stats(), top_rows=acc.top_rows()))
    return out


def _issue(code: ErrorCode, severity: Severity, message: str, **details: object) -> Issue:
    return Issue(
        code=code,
        severity=severity,
        stage=Stage.TOKENIZING,
        user_message=message,
        details=dict(details),
    )


def _failed_rows_issue(state: _State, ending: _Ending) -> Issue:
    """Rows that were read but could not be decoded or tokenized: their lengths are unknown, so
    the scan is PARTIAL even when every row was read (plan §19.2)."""
    failed, seen = state.rows_failed, state.rows_seen
    if ending.all_rows_seen:
        message = (
            f"데이터셋의 row {seen:,}개를 모두 읽었지만 그중 {failed:,}개를 읽거나 토큰화하지 "
            "못했습니다. 이 row들의 길이를 알 수 없어 최대 길이를 보장할 수 없으므로 전체 스캔 "
            "완료로 표시하지 않습니다(부분 스캔). 실패한 row의 위치는 실패 row 목록에 있습니다."
        )
    else:
        message = (
            f"읽은 row {seen:,}개 중 {failed:,}개를 읽거나 토큰화하지 못했습니다. 이 row들의 "
            "길이는 통계에 포함되지 않았습니다. 실패한 row의 위치는 실패 row 목록에 있습니다."
        )
    return _issue(
        ErrorCode.SCAN_FAILED_ROWS,
        Severity.ERROR,
        message,
        rows_seen=seen,
        rows_ok=state.rows_ok,
        rows_failed=failed,
        all_rows_read=ending.all_rows_seen,
        row_ids=[f["row_id"] for f in state.failed_sample[:ROW_ID_SAMPLE_LIMIT]],
    )


def _issues(state: _State, stream: RowStream, ending: _Ending, limits: ScanLimits) -> list[Issue]:
    # The reader's own stop/failure reasons (e.g. inspection's DatasetRowStream.issues) are the
    # most precise explanation of an incomplete stream, so they come first.
    stream_issues = [i for i in (getattr(stream, "issues", None) or []) if isinstance(i, Issue)]
    issues: list[Issue] = list(stream_issues)
    seen = {"rows_seen": state.rows_seen, "rows_expected": stream.total_rows}
    if ending.stop_reason == "max_rows":
        issues.append(
            _issue(
                ErrorCode.SCAN_QUOTA_EXCEEDED,
                Severity.ERROR,
                f"서비스 한도(최대 {limits.max_rows:,} row)에 도달해 스캔을 멈췄습니다. 나머지 "
                "row는 처리하지 않았으며 전체 스캔 완료로 표시하지 않습니다.",
                limit="max_rows",
                **seen,
            )
        )
    elif ending.stop_reason == "max_seconds":
        issues.append(
            _issue(
                ErrorCode.SCAN_QUOTA_EXCEEDED,
                Severity.ERROR,
                f"서비스 시간 한도({limits.max_seconds:,.0f}초)에 도달해 스캔을 멈췄습니다. "
                "나머지 row는 처리하지 않았으며 전체 스캔 완료로 표시하지 않습니다.",
                limit="max_seconds",
                **seen,
            )
        )
    read_error = ending.read_error
    if read_error is not None:
        cause = None
        if isinstance(read_error, EstimatorError):
            # The reader's own issue is display-safe (errors.py) and says what went wrong.
            cause = read_error.issue.code.value
            if read_error.issue not in issues:
                issues.append(read_error.issue)
        issues.append(
            _issue(
                ErrorCode.SCAN_PARTIAL,
                Severity.ERROR,
                f"데이터를 읽는 중 오류가 발생해 스캔이 중단되었습니다 "
                f"({type(read_error).__name__}). 통계는 읽은 row까지의 값입니다.",
                cause_code=cause,
                **seen,
            )
        )
    elif ending.manifest_mismatch:
        issues.append(
            _issue(
                ErrorCode.SCAN_PARTIAL,
                Severity.ERROR,
                "읽은 row 수가 데이터셋 manifest의 row 수와 다릅니다. 전체 스캔 완료로 표시하지 "
                "않습니다.",
                **seen,
            )
        )
    elif not ending.read_all and ending.stop_reason is None and not stream_issues:
        issues.append(
            _issue(
                ErrorCode.SCAN_PARTIAL,
                Severity.WARNING,
                "데이터셋 reader가 모든 shard를 끝까지 정상적으로 읽었다고 확인하지 못해 부분 "
                "스캔으로 표시합니다. 최대 길이와 통계는 읽은 row까지의 값이며 전체 최대 길이가 "
                "아닙니다.",
                **seen,
            )
        )
    if ending.empty and not any(i.code is ErrorCode.EMPTY_DATASET for i in issues):
        issues.append(
            _issue(
                ErrorCode.EMPTY_DATASET,
                Severity.ERROR,
                "선택한 split에 row가 없습니다(0개). 분석할 데이터가 없어 길이 통계와 batch 계획을 "
                "만들 수 없습니다.",
                split=stream.split,
                **seen,
            )
        )
    if state.rows_failed:
        issues.append(_failed_rows_issue(state, ending))
    loss = state.diag_counts["template_loss"]
    if loss:
        issues.append(
            _issue(
                ErrorCode.TEMPLATE_CONTENT_LOSS,
                Severity.ERROR,
                f"chat template이 매핑된 content를 원문 그대로 렌더링하지 않은 row가 {loss:,}개 "
                "있습니다.",
                rows=loss,
                row_ids=state.diag_samples["template_loss"],
            )
        )
    mismatch = state.diag_counts["prefix_mismatch"]
    if mismatch:
        issues.append(
            _issue(
                ErrorCode.TEMPLATE_CONTENT_LOSS,
                Severity.WARNING,
                f"prompt만 토큰화한 결과가 prompt+응답 토큰화 결과의 앞부분과 다른 row가 "
                f"{mismatch:,}개 있습니다. TRL도 경고만 하고 같은 방식으로 자르므로 길이는 "
                "TRL과 동일하게 계산했지만 loss 경계가 어긋날 수 있습니다.",
                kind="prompt_boundary_mismatch",
                rows=mismatch,
                row_ids=state.diag_samples["prefix_mismatch"],
            )
        )
    special = state.diag_counts["special_token"]
    if special:
        issues.append(
            _issue(
                ErrorCode.TEMPLATE_CONTENT_LOSS,
                Severity.WARNING,
                f"BOS 또는 EOS 토큰이 연속으로 두 번 들어간 row가 {special:,}개 있습니다 "
                "(원문에 special token 문자열이 포함된 경우). TRL과 같은 길이로 계산했습니다.",
                kind="duplicate_special_token",
                rows=special,
                row_ids=state.diag_samples["special_token"],
            )
        )
    return issues


def load_lengths(artifact_path: Path) -> LengthTable:
    """Read the row-length artifact written by `full_scan` (successful rows, source order)."""
    wanted = [
        "row_id",
        "processing_status",
        "prompt_tokens",
        "completion_tokens",
        "sequence_tokens",
        "loss_token_count",
        "chosen_total_tokens",
        "rejected_total_tokens",
    ]
    table = LengthTable()
    for cols in iter_columns(part_paths(artifact_path), wanted):
        for i, status in enumerate(cols["processing_status"]):
            if status != "ok":
                continue
            table.row_ids.append(cols["row_id"][i])
            table.prompt_tokens.append(cols["prompt_tokens"][i])
            table.completion_tokens.append(cols["completion_tokens"][i])
            table.sequence_tokens.append(cols["sequence_tokens"][i])
            table.loss_token_count.append(cols["loss_token_count"][i])
            table.chosen_total_tokens.append(cols["chosen_total_tokens"][i])
            table.rejected_total_tokens.append(cols["rejected_total_tokens"][i])
    return table
