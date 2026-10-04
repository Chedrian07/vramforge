"""Full scan: coverage, artifact, checkpoint/resume, cancellation, limits (plan §7.6-7.7, §16.2)."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from vramforge_estimator.errors import CancelledError
from vramforge_estimator.scan import ScanLimits, full_scan, load_lengths
from vramforge_estimator.scan import scanner as scanner_module
from vramforge_estimator.schemas import (
    Branch,
    ErrorCode,
    JobStatus,
    LengthRecord,
    Objective,
    ScanCoverage,
)

LENGTHS = [120, 300, 40, 300, 75, 9, 210]


def rows(lengths: list[int] = LENGTHS) -> list[dict]:
    return [{"len": n, "text": f"row-{i}", "lang": "meta"} for i, n in enumerate(lengths)]


def scan(stream, adapter, ctx, *, key: str = "pre_a", limit: int | None = 4096):
    return full_scan(
        stream,
        adapter,
        ctx,
        objective=Objective.SFT,
        preprocess_key=key,
        tokenizer_fingerprint="tok_fp",
        template_fingerprint="tpl_fp",
        context_limit=limit,
    )


def branch(result, name: Branch):
    return next(b for b in result.branches if b.branch is name)


def test_complete_scan_counts_every_row_exactly(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows()), adapter, ctx)
    res = out.result
    assert res.coverage is ScanCoverage.COMPLETE
    assert (res.rows_expected, res.rows_seen, res.rows_ok, res.rows_failed) == (7, 7, 7, 0)
    assert res.rows_unprocessed == 0 and out.issues == []
    seq = branch(res, Branch.SEQUENCE).stats
    assert (seq.count, seq.min, seq.max, seq.total_tokens) == (7, 9, 300, sum(LENGTHS))
    assert seq.max_row_id == "train:1"  # first of the two 300-token rows
    assert [r.row_id for r in branch(res, Branch.SEQUENCE).top_rows[:2]] == ["train:1", "train:3"]
    assert res.preprocessing_adapter == "fake-sft" and res.preprocessing_adapter_version == "1"
    assert res.preprocess_key == "pre_a" and res.artifact_id == "lengths"
    assert res.transformation_note == "테스트 변환"


def test_artifact_has_length_record_columns_and_round_trips(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows()), adapter, ctx)
    parts = sorted(out.artifact_path.glob("part-*.parquet"))
    table = pq.read_table(parts)
    assert table.column_names == list(LengthRecord.model_fields)
    records = [LengthRecord.model_validate(r) for r in table.to_pylist()]
    assert [r.row_id for r in records] == [f"train:{i}" for i in range(7)]
    assert {r.split for r in records} == {"train"}
    assert {r.tokenizer_fingerprint for r in records} == {"tok_fp"}
    assert all(r.processing_status == "ok" and r.context_status == "ok" for r in records)
    lengths = load_lengths(out.artifact_path)
    assert lengths.sequence_tokens == LENGTHS and len(lengths) == 7
    assert (out.artifact_path / "manifest.json").is_file()


def test_artifact_keeps_the_token_digest_of_ok_rows_only(make_stream, adapter, ctx) -> None:
    data = [{"len": 5, "digest": "ab" * 32}, {"fail": True, "digest": "cd" * 32}, {"len": 6}]
    out = scan(make_stream(data), adapter, ctx)
    table = pq.read_table(sorted(out.artifact_path.glob("part-*.parquet"))).to_pylist()
    assert [r["token_digest"] for r in table] == ["ab" * 32, None, None]


def test_artifact_never_stores_raw_text(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows()), adapter, ctx)
    blob = b"".join(p.read_bytes() for p in out.artifact_path.iterdir())
    assert b"row-3" not in blob and b"meta" not in blob


def test_corrupt_row_is_counted_with_position_and_blocks_verification(
    make_stream, adapter, ctx
) -> None:
    data = rows()
    data[4] = {"fail": True}
    out = scan(make_stream(data, shards=["/abs/host/path/train.jsonl"]), adapter, ctx)
    res = out.result
    assert res.rows_failed == 1 and res.rows_ok == 6
    # every row was read, but one length is unknown: the maximum is not guaranteed (plan §19.2)
    assert res.coverage is ScanCoverage.PARTIAL and res.rows_unprocessed == 0
    failed = res.failed_rows_sample[0]
    assert failed.row_id == "train:4" and failed.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED
    assert failed.shard_id == "train.jsonl"  # host path stripped
    codes = [i.code for i in out.issues]
    assert codes == [ErrorCode.SCAN_FAILED_ROWS]  # the reader is not blamed for an unread stream
    issue = out.issues[0]
    assert issue.details["row_ids"] == ["train:4"]
    assert (issue.details["rows_seen"], issue.details["rows_failed"]) == (7, 1)
    assert issue.details["all_rows_read"] is True
    assert "row 7개를 모두 읽었지만 그중 1개를" in issue.user_message
    table = pq.read_table(sorted(out.artifact_path.glob("part-*.parquet"))).to_pylist()
    assert table[4]["processing_status"] == "failed" and table[4]["error_code"] is not None
    assert len(load_lengths(out.artifact_path)) == 6


def test_failed_sample_is_capped_at_100(make_stream, adapter, ctx) -> None:
    out = scan(make_stream([{"fail": True}] * 130), adapter, ctx)
    assert out.result.rows_failed == 130 and len(out.result.failed_rows_sample) == 100
    assert out.result.coverage is ScanCoverage.PARTIAL  # rows were read; none has a length


def test_failed_rows_in_a_partial_read_say_how_many_were_read(make_stream, adapter, ctx) -> None:
    data = rows()
    data[1] = {"fail": True}
    out = scan(make_stream(data, stop_early_at=4), adapter, ctx)
    assert out.result.coverage is ScanCoverage.PARTIAL and out.result.rows_unprocessed == 3
    failed = next(i for i in out.issues if i.code is ErrorCode.SCAN_FAILED_ROWS)
    assert failed.details["all_rows_read"] is False
    assert "읽은 row 4개 중 1개를" in failed.user_message
    assert ErrorCode.SCAN_PARTIAL in [i.code for i in out.issues]  # the unread rows, separately


def test_final_report_never_calls_failed_rows_finished(make_stream, adapter, ctx) -> None:
    data = rows()
    data[2] = {"fail": True}
    scan(make_stream(data), adapter, ctx)
    final, final_partial = ctx.reports[-1]
    assert final.message_code == "scan_partial" and final_partial["status"] == "partial"
    assert final_partial["rows_failed"] == 1
    assert "모든 row를 읽었지만 1개 row를" in final.message
    assert "데이터셋 전체 토큰화를 마쳤습니다" not in final.message


def test_empty_split_is_failed_with_empty_dataset(make_stream, adapter, ctx) -> None:
    out = scan(make_stream([]), adapter, ctx)
    res = out.result
    assert res.coverage is ScanCoverage.FAILED
    assert (res.rows_expected, res.rows_seen, res.rows_unprocessed) == (0, 0, 0)
    assert [i.code for i in out.issues] == [ErrorCode.EMPTY_DATASET]
    assert out.issues[0].details["split"] == "train"
    seq = branch(res, Branch.SEQUENCE).stats
    assert seq.count == 0 and seq.max is None  # nothing invented
    final, final_partial = ctx.reports[-1]
    assert final.message_code == "scan_failed" and "row가 없어" in final.message
    assert final_partial["status"] == "failed" and final_partial["rows_ok"] == 0
    assert len(load_lengths(out.artifact_path)) == 0


def test_reader_empty_dataset_issue_is_not_repeated(make_stream, adapter, ctx) -> None:
    from vramforge_estimator.schemas import Issue, Severity

    class EmptyStream(make_stream):
        def __iter__(self):
            yield from super().__iter__()
            self.issues.append(
                Issue(
                    code=ErrorCode.EMPTY_DATASET, severity=Severity.ERROR, user_message="빈 split"
                )
            )

    out = scan(EmptyStream([]), adapter, ctx)
    assert [i.code for i in out.issues] == [ErrorCode.EMPTY_DATASET]
    assert out.issues[0].user_message == "빈 split"  # the reader's own wording is kept


def test_nothing_read_is_failed_but_not_called_empty(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows(), stop_early_at=0), adapter, ctx)
    assert out.result.coverage is ScanCoverage.FAILED and out.result.rows_unprocessed == 7
    codes = [i.code for i in out.issues]
    assert ErrorCode.EMPTY_DATASET not in codes and codes == [ErrorCode.SCAN_PARTIAL]


def test_oversized_row_is_reported_not_truncated(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows([100, 50_000, 30])), adapter, ctx, limit=4096)
    res = out.result
    assert res.context_exceeded_rows == 1
    assert branch(res, Branch.SEQUENCE).stats.max == 50_000  # full length kept
    table = pq.read_table(sorted(out.artifact_path.glob("part-*.parquet"))).to_pylist()
    assert [r["context_status"] for r in table] == ["ok", "exceeded", "ok"]


@pytest.mark.parametrize("limit", [None, 10**30])  # no limit, or transformers' "no limit" value
def test_unknown_context_limit_is_never_counted_as_zero(make_stream, adapter, ctx, limit) -> None:
    out = scan(make_stream(rows([100, 50_000])), adapter, ctx, limit=limit)
    assert out.result.context_exceeded_rows is None
    table = pq.read_table(sorted(out.artifact_path.glob("part-*.parquet"))).to_pylist()
    assert {r["context_status"] for r in table} == {"unknown"}


def test_partial_stream_is_never_complete(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows(), stop_early_at=4), adapter, ctx)
    res = out.result
    assert res.coverage is ScanCoverage.PARTIAL
    assert res.rows_seen == 4 and res.rows_unprocessed == 3
    assert [i.code for i in out.issues] == [ErrorCode.SCAN_PARTIAL]
    assert branch(res, Branch.SEQUENCE).stats.max == 300  # max of the rows read so far
    unknown_total = scan(
        make_stream(rows(), stop_early_at=4, total_rows=None), adapter, ctx, key="pre_c"
    )
    assert unknown_total.result.rows_unprocessed is None
    assert unknown_total.result.coverage is ScanCoverage.PARTIAL


def test_manifest_row_count_mismatch_is_partial(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows(), total_rows=10), adapter, ctx)
    assert out.result.coverage is ScanCoverage.PARTIAL
    assert out.result.rows_unprocessed == 3
    assert out.issues[0].code is ErrorCode.SCAN_PARTIAL


def test_read_error_keeps_verified_rows(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows(), fail_at=5), adapter, ctx)
    assert out.result.coverage is ScanCoverage.PARTIAL and out.result.rows_seen == 5
    assert out.issues[0].code is ErrorCode.SCAN_PARTIAL
    assert "OSError" in out.issues[0].user_message
    first = scan(make_stream(rows(), fail_at=0), adapter, ctx, key="pre_d")
    assert first.result.coverage is ScanCoverage.FAILED and first.result.rows_seen == 0


def test_row_quota_stops_partial_and_exact_quota_completes(make_stream, adapter, make_ctx) -> None:
    out = scan(make_stream(rows()), adapter, make_ctx(limits=ScanLimits(max_rows=5)))
    assert out.result.coverage is ScanCoverage.PARTIAL and out.result.rows_seen == 5
    assert out.result.rows_unprocessed == 2
    assert out.issues[0].code is ErrorCode.SCAN_QUOTA_EXCEEDED
    exact = scan(make_stream(rows()), adapter, make_ctx(limits=ScanLimits(max_rows=7)), key="x")
    assert exact.result.coverage is ScanCoverage.COMPLETE


def test_time_quota_stops_partial(make_stream, adapter, make_ctx, monkeypatch) -> None:
    ticks = iter(range(1000))
    monkeypatch.setattr(scanner_module, "clock", lambda: float(next(ticks)))
    out = scan(make_stream(rows()), adapter, make_ctx(limits=ScanLimits(max_seconds=4)))
    assert out.result.coverage is ScanCoverage.PARTIAL
    assert 0 < out.result.rows_seen < 7
    assert out.issues[0].code is ErrorCode.SCAN_QUOTA_EXCEEDED
    assert out.issues[0].details["limit"] == "max_seconds"


def test_oversized_record_fails_that_row_only(make_stream, adapter, make_ctx) -> None:
    data = rows()
    data[2]["blob"] = "x" * 500
    out = scan(make_stream(data), adapter, make_ctx(limits=ScanLimits(max_record_chars=200)))
    assert out.result.rows_failed == 1 and out.result.coverage is ScanCoverage.PARTIAL
    assert out.result.failed_rows_sample[0].error_code is ErrorCode.SCAN_QUOTA_EXCEEDED
    assert "train:2" not in adapter.calls  # never tokenized


def test_progress_is_throttled_and_flags_partial_maxima(
    make_stream, adapter, make_ctx, monkeypatch
) -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 0.1
        return now[0]

    monkeypatch.setattr(scanner_module, "clock", clock)
    ctx = make_ctx(limits=ScanLimits(progress_interval_s=1.0))
    scan(make_stream(rows(list(range(1, 41)))), adapter, ctx)
    progress = [p for p, _ in ctx.reports]
    assert all(p.stage is JobStatus.TOKENIZING for p in progress)
    assert 3 <= len(progress) <= 10  # 40 rows x 0.1s at 1s interval, not one per row
    _, partial = ctx.reports[1]
    assert partial["status"] == "partial" and partial["max_sequence"] >= 1
    final, final_partial = ctx.reports[-1]
    assert final.processed_rows == 40 and final.total_rows == 40
    assert final.shard_progress.completed == 1 and final.shard_progress.total == 1
    assert final_partial["status"] == "complete" and final_partial["max_sequence"] == 40


@pytest.mark.parametrize(
    ("stream_kwargs", "coverage"),
    [({"stop_early_at": 4}, "partial"), ({"fail_at": 0}, "failed"), ({}, "complete")],
)
def test_final_report_states_the_real_coverage(
    make_stream, adapter, ctx, stream_kwargs, coverage
) -> None:
    scan(make_stream(rows(), **stream_kwargs), adapter, ctx)
    final, final_partial = ctx.reports[-1]
    assert final.message_code == f"scan_{coverage}" and final_partial["status"] == coverage
    finished_all = "데이터셋 전체 토큰화를 마쳤습니다" in (final.message or "")
    assert finished_all is (coverage == "complete")  # partial maxima never sold as complete


def test_reader_estimator_error_keeps_its_own_issue(make_stream, adapter, ctx) -> None:
    from vramforge_estimator.errors import EstimatorError, make_issue

    reader_issue = make_issue(ErrorCode.SOURCE_REVISION_CHANGED, "데이터 파일이 바뀌었습니다.")

    class ChangedStream(make_stream):
        def __iter__(self):
            for i, source_row in enumerate(super().__iter__()):
                if i == 3:  # e.g. the file changed under the reader
                    raise EstimatorError(reader_issue)
                yield source_row

    out = scan(ChangedStream(rows()), adapter, ctx)
    assert out.result.coverage is ScanCoverage.PARTIAL and out.result.rows_seen == 3
    assert [i.code for i in out.issues] == [
        ErrorCode.SOURCE_REVISION_CHANGED,
        ErrorCode.SCAN_PARTIAL,
    ]
    assert out.issues[0] == reader_issue
    assert out.issues[1].details["cause_code"] == "SOURCE_REVISION_CHANGED"


def test_unknown_total_never_invents_a_percentage(make_stream, adapter, ctx) -> None:
    scan(make_stream(rows(), total_rows=None), adapter, ctx)
    assert all(p.total_rows is None for p, _ in ctx.reports)


def test_cancellation_raises(make_stream, adapter, make_ctx) -> None:
    ctx = make_ctx(cancel_after_calls=1)
    with pytest.raises(CancelledError) as err:
        scan(make_stream(rows(list(range(1, 100)))), adapter, ctx)
    assert err.value.issue.code is ErrorCode.CANCELLED
    assert len(adapter.calls) < 99


def test_resume_after_worker_death_matches_a_clean_run(
    make_stream, make_ctx, make_adapter, tmp_path
) -> None:
    data = rows(list(range(10, 260, 10)))  # 25 rows
    limits = ScanLimits(checkpoint_every_rows=4)

    clean_ctx = make_ctx(limits=limits)
    clean_ctx.artifact_dir = tmp_path / "clean"
    clean_adapter = make_adapter()
    clean = scan(make_stream(data), clean_adapter, clean_ctx)

    crashing = make_ctx(limits=limits, crash_after_saves=3)  # dies after 12 rows are saved
    first_adapter = make_adapter()
    with pytest.raises(RuntimeError, match="simulated worker death"):
        scan(make_stream(data), first_adapter, crashing)
    assert crashing.checkpoint["state"]["last_row_index"] == 11
    # an orphan part (written after the last checkpoint) must be ignored on resume
    (crashing.artifact_dir / "lengths" / "part-00009.parquet").write_bytes(b"garbage")

    resumed_ctx = make_ctx(limits=limits, checkpoint=crashing.checkpoint)
    second_adapter = make_adapter()
    resumed = scan(make_stream(data), second_adapter, resumed_ctx)
    assert second_adapter.calls == [f"train:{i}" for i in range(12, 25)]
    assert resumed.result.rows_seen == 25 and resumed.result.coverage is ScanCoverage.COMPLETE
    assert resumed.result.branches == clean.result.branches
    assert load_lengths(resumed.artifact_path).sequence_tokens == [r["len"] for r in data]
    assert not (resumed.artifact_path / "part-00009.parquet").exists()


def test_checkpoint_for_another_preprocess_key_is_not_reused(
    make_stream, make_ctx, make_adapter
) -> None:
    limits = ScanLimits(checkpoint_every_rows=2)
    first = make_ctx(limits=limits)
    scan(make_stream(rows()), make_adapter(), first, key="pre_template_a")
    again = make_ctx(limits=limits, checkpoint=first.checkpoint)
    adapter_b = make_adapter()
    out = scan(make_stream(rows()), adapter_b, again, key="pre_template_b")
    assert len(adapter_b.calls) == 7  # template change -> full rescan, no cache reuse
    assert out.result.preprocess_key == "pre_template_b"


def test_duplicates_counted_by_content_digest(make_stream, adapter, ctx) -> None:
    data = [{"len": 5, "text": "same"}, {"len": 6, "text": "other"}, {"len": 5, "text": "same"}]
    data.append({"len": 5, "text": "same"})
    assert scan(make_stream(data), adapter, ctx).result.duplicate_rows == 2


def test_split_isolation(make_stream, adapter, make_ctx, tmp_path) -> None:
    train = make_stream(rows([10, 20, 30]), split="train")
    out = scan(train, adapter, make_ctx())
    assert out.result.split == "train" and out.result.rows_seen == 3
    table = pq.read_table(sorted(out.artifact_path.glob("part-*.parquet"))).to_pylist()
    assert {r["split"] for r in table} == {"train"}
    assert all(r["row_id"].startswith("train:") for r in table)
    assert all(call.startswith("train:") for call in adapter.calls)


def test_template_loss_and_system_omission_are_reported(make_stream, adapter, ctx) -> None:
    data = [{"len": 10, "system": ""}, {"len": 11, "lossy": True}, {"len": 12, "system": ""}]
    out = scan(make_stream(data), adapter, ctx)
    assert out.template_content_loss_rows == 1
    loss = [i for i in out.issues if i.code is ErrorCode.TEMPLATE_CONTENT_LOSS]
    assert loss and loss[0].details["row_ids"] == ["train:1"]
    assert out.result.omitted_system_messages == 2
    assert out.result.transformation_note == "테스트 변환"  # the count is a field, not prose


def test_omitted_system_messages_are_counted_per_message(make_stream, adapter, ctx) -> None:
    data = [{"len": 10, "omitted": 3}, {"len": 11}, {"len": 12, "system": ""}]
    assert scan(make_stream(data), adapter, ctx).result.omitted_system_messages == 4


@pytest.mark.parametrize(
    ("mapping", "data", "expected"),
    [
        ({"system": "s", "prompt": "p", "completion": "c"}, [{"len": 5}], 0),
        ({"messages": "m"}, [{"len": 5}], 0),  # conversations may carry system messages
        ({"prompt": "p", "completion": "c"}, [{"len": 5}], None),  # no system role mapped
        ({"prompt": "p", "completion": "c"}, [{"len": 5, "omitted": 1}], 1),  # list prompt
        (  # KEEP renders even an empty system value: nothing to count
            {"system": "s", "prompt": "p", "completion": "c", "empty_system_policy": "keep"},
            [{"len": 5, "omitted": 1}],
            None,
        ),
    ],
)
def test_omitted_system_messages_applicability(
    make_stream, adapter, ctx, mapping, data, expected
) -> None:
    from vramforge_estimator.schemas import ColumnMapping

    adapter.mapping = ColumnMapping(**mapping)
    out = scan(make_stream(data), adapter, ctx)
    assert out.result.omitted_system_messages == expected
    assert out.result.mapping_applied == adapter.mapping


def test_reader_decode_failures_keep_their_reason(make_stream, adapter, ctx) -> None:
    out = scan(make_stream(rows(), decode_fail_at=2), adapter, ctx)
    res = out.result
    assert res.rows_seen == 7 and res.rows_failed == 1 and res.rows_ok == 6
    failed = res.failed_rows_sample[0]
    assert failed.row_id == "train:2" and failed.error_code is ErrorCode.SCAN_FAILED_ROWS
    assert failed.message == "JSON으로 해석할 수 없는 레코드입니다"
    assert "train:2" not in adapter.calls  # nothing to tokenize
    assert res.coverage is ScanCoverage.PARTIAL  # the reader did not confirm a full read
    assert res.rows_unprocessed == 0  # but every expected row was read
    codes = [i.code for i in out.issues]
    assert codes == [ErrorCode.SCAN_FAILED_ROWS, ErrorCode.SCAN_FAILED_ROWS]
    assert out.issues[0].user_message.startswith("해석할 수 없는 레코드")  # reader's own issue
    assert out.issues[1].details["all_rows_read"] is True
