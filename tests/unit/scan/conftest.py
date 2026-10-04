"""In-memory fakes for the scanner: RowStream, ScanContext and a deterministic adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.inspection import SourceRow
from vramforge_estimator.preprocessing import TokenizedRecord
from vramforge_estimator.preprocessing.mapping import content_digest
from vramforge_estimator.scan import ScanLimits
from vramforge_estimator.schemas import ErrorCode, Issue, JobProgress, Objective, Severity


@dataclass(frozen=True)
class DecodeFailedRow(SourceRow):
    """Shape of inspection's FailedSourceRow: no columns, a code and a Korean message."""

    error_code: ErrorCode = ErrorCode.SCAN_FAILED_ROWS
    message: str = ""


class WorkerDied(RuntimeError):
    """Simulates the worker process dying right after a checkpoint was saved."""


class FakeStream:
    """A finite split. `complete` turns True only after a full, error-free iteration."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        split: str = "train",
        shards: list[str] | None = None,
        total_rows: int | None = -1,
        stop_early_at: int | None = None,
        fail_at: int | None = None,
        rows_per_shard: int | None = None,
        decode_fail_at: int | None = None,
    ) -> None:
        self.rows = rows
        self.split = split
        self.config: str | None = None
        self.shards = shards if shards is not None else ["data/train-00000.jsonl"]
        self.total_rows = len(rows) if total_rows == -1 else total_rows
        self.stop_early_at = stop_early_at
        self.fail_at = fail_at
        self.rows_per_shard = rows_per_shard
        self.decode_fail_at = decode_fail_at
        self.issues: list[Issue] = []
        self._complete = False
        self._shards_done = 0

    def __iter__(self) -> Iterator[SourceRow]:
        self._complete = False
        self.issues = []
        for i, row in enumerate(self.rows):
            if self.fail_at is not None and i == self.fail_at:
                raise OSError("simulated read failure")
            if self.stop_early_at is not None and i == self.stop_early_at:
                return  # quota/cancel inside the reader: EOF not reached
            shard_index = i // self.rows_per_shard if self.rows_per_shard else 0
            self._shards_done = shard_index
            if i == self.decode_fail_at:
                yield DecodeFailedRow(
                    row_index=i,
                    shard_id=self.shards[shard_index],
                    row={},
                    message="JSON으로 해석할 수 없는 레코드입니다",
                )
                continue
            yield SourceRow(row_index=i, shard_id=self.shards[shard_index], row=row)
        self._shards_done = len(self.shards)
        if self.decode_fail_at is not None:
            self.issues.append(
                Issue(
                    code=ErrorCode.SCAN_FAILED_ROWS,
                    severity=Severity.ERROR,
                    user_message="해석할 수 없는 레코드가 있어 전체 읽기를 완료로 보지 않습니다.",
                )
            )
            return
        self._complete = True

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def shards_completed(self) -> int:
        return self._shards_done


@dataclass
class FakeCtx:
    artifact_dir: Path
    limits: ScanLimits = field(default_factory=ScanLimits)
    reports: list[tuple[JobProgress, dict[str, Any] | None]] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None
    saves: int = 0
    crash_after_saves: int | None = None
    cancel_after_calls: int | None = None
    cancel_calls: int = 0

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        self.reports.append((progress, partial))

    def cancelled(self) -> bool:
        self.cancel_calls += 1
        return self.cancel_after_calls is not None and self.cancel_calls > self.cancel_after_calls

    def load_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        self.checkpoint = json.loads(json.dumps(data))  # must be JSON-serializable
        self.saves += 1
        if self.crash_after_saves is not None and self.saves >= self.crash_after_saves:
            raise WorkerDied("simulated worker death")


class FakeSftAdapter:
    """Row {"len": n, "prompt": p} -> SFT record; {"fail": True} fails; {"lossy": True} loses
    template content; {"system": ""} counts as one omitted system message, {"omitted": k} as k."""

    name = "fake-sft"
    version = "1"
    objective = Objective.SFT

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transformation_note(self) -> str:
        return "테스트 변환"

    def process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        self.calls.append(row_id)
        if row.get("fail"):
            return TokenizedRecord(
                row_id=row_id,
                objective=self.objective,
                ok=False,
                error_code=ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                error_message="손상된 row입니다.",
            )
        length = int(row["len"])
        prompt = int(row.get("prompt", length // 4))
        extras: dict[str, Any] = {}
        if row.get("system") == "":
            extras["system_omitted"] = 1
        if "omitted" in row:
            extras["system_omitted"] = int(row["omitted"])
        return TokenizedRecord(
            row_id=row_id,
            objective=self.objective,
            ok=True,
            prompt_tokens=prompt,
            completion_tokens=length - prompt,
            sequence_tokens=length,
            loss_token_count=length - prompt,
            content_digest=content_digest({"text": row.get("text", str(length))}),
            template_preserved=not row.get("lossy", False),
            template_issue="loss" if row.get("lossy") else None,
            extras=extras,
        )


@pytest.fixture
def ctx(tmp_path: Path) -> FakeCtx:
    return FakeCtx(artifact_dir=tmp_path / "artifacts")


@pytest.fixture
def adapter() -> FakeSftAdapter:
    return FakeSftAdapter()


@pytest.fixture
def make_stream():
    return FakeStream


@pytest.fixture
def make_ctx(tmp_path: Path):
    def factory(**kwargs: Any) -> FakeCtx:
        return FakeCtx(artifact_dir=tmp_path / "artifacts", **kwargs)

    return factory


@pytest.fixture
def make_adapter():
    return FakeSftAdapter
