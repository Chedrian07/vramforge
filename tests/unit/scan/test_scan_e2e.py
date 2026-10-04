"""Real adapters + scanner + planner on the offline MiMo-template tokenizer (plan §19.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vramforge_estimator import keys
from vramforge_estimator.batching import plan_batches
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.scan import audit_preservation, full_scan, load_lengths, validate_context
from vramforge_estimator.schemas import (
    Branch,
    ColumnMapping,
    DataPreservation,
    EffectiveDtypes,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
    OptimizerResolved,
    PreservationCheckName,
    QuantizationResolved,
    ResolvedConfig,
    ScanCoverage,
    ScopeConfig,
    Strategy,
    TokenizerManifest,
    WorkspaceAssumptions,
)

TOKENIZERS = Path(__file__).parents[2] / "fixtures" / "golden" / "tokenizers"
MAPPING = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROWS = [
    {
        "system": "",
        "question": "C 함수",
        "chosen": "```c\nint f(void){return 0;}\n```",
        "rejected": "몰라요",
        "lang": "c",
        "vulnerability": "meta",
    },
    {
        "system": "",
        "question": "파이썬 eval",
        "chosen": "ast.literal_eval 사용",
        "rejected": "B" * 3000,
        "lang": "python",
        "vulnerability": "meta",
    },
    {
        "system": "보안 리뷰어",
        "question": "Go",
        "chosen": "x" * 20_000,
        "rejected": "no",
        "lang": "go",
        "vulnerability": "meta",
    },
]


@pytest.fixture(scope="module")
def handle() -> TokenizerHandle:
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZERS / "mimo_bytelevel")
    manifest = TokenizerManifest(
        tokenizer_class=type(tok).__name__,
        vocab_size=len(tok),
        chat_template_present=True,
        has_generation_markers=True,
        fingerprint="fixture:mimo_bytelevel",
    )
    return TokenizerHandle(tokenizer=tok, manifest=manifest)


def resolved(objective: Objective, microbatch: int) -> ResolvedConfig:
    dt = "bfloat16"
    return ResolvedConfig(
        objective=objective,
        strategy=Strategy.LORA,
        loading_scope="text_only",
        load_dtype=dt,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=dt,
            compute=dt,
            adapter=dt,
            gradient=dt,
            optimizer_state=dt,
            logits=dt,
            loss=dt,
            kv_cache=dt,
            recurrent_state=dt,
        ),
        quantization=QuantizationResolved(enabled=False),
        optimizer=OptimizerResolved(name="adamw_torch", states_per_param=2, state_dtype=dt),
        microbatch=microbatch,
        accumulation=1,
        gradient_checkpointing=True,
        checkpointing_granularity="per_decoder_layer",
        loss_path="chunked_nll",
        workspace=WorkspaceAssumptions(
            cuda_context_bytes=(1, 1),
            library_workspace_bytes=(1, 1),
            allocator_slack_fraction=(0.0, 0.0),
        ),
    )


@pytest.mark.parametrize("objective", [Objective.SFT, Objective.DPO, Objective.GRPO])
def test_scan_records_exactly_what_the_adapter_computes(
    objective, handle, make_stream, make_ctx
) -> None:
    adapter = get_adapter(objective, handle, MAPPING)
    expected = [adapter.process(row, f"train:{i}") for i, row in enumerate(ROWS)]
    out = full_scan(
        make_stream(ROWS),
        adapter,
        make_ctx(),
        objective=objective,
        preprocess_key="pre_e2e",
        tokenizer_fingerprint=handle.manifest.fingerprint,
        template_fingerprint="tpl",
        context_limit=262_144,
    )
    assert out.result.coverage is ScanCoverage.COMPLETE and out.issues == []
    assert out.result.mapping_applied == MAPPING
    table = load_lengths(out.artifact_path)
    assert table.prompt_tokens == [r.prompt_tokens for r in expected]
    assert table.sequence_tokens == [r.sequence_tokens for r in expected]
    assert table.rejected_total_tokens == [r.rejected_total_tokens for r in expected]
    assert out.result.omitted_system_messages == 2  # the two empty system values
    assert "omit" in out.result.transformation_note  # the policy itself is still stated


def test_keep_policy_has_no_omitted_count(handle, make_stream, make_ctx) -> None:
    keep = MAPPING.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP})
    adapter = get_adapter(Objective.GRPO, handle, keep, empty_system_policy=EmptySystemPolicy.KEEP)
    out = full_scan(
        make_stream(ROWS),
        adapter,
        make_ctx(),
        objective=Objective.GRPO,
        preprocess_key="pre_keep",
        tokenizer_fingerprint="t",
        template_fingerprint=None,
        context_limit=None,
    )
    assert out.result.omitted_system_messages is None
    assert out.result.context_exceeded_rows is None  # no limit known


def test_dpo_rejected_outlier_reaches_the_batch_shape(handle, make_stream, make_ctx) -> None:
    adapter = get_adapter(Objective.DPO, handle, MAPPING)
    out = full_scan(
        make_stream(ROWS),
        adapter,
        make_ctx(),
        objective=Objective.DPO,
        preprocess_key="pre_dpo",
        tokenizer_fingerprint="t",
        template_fingerprint=None,
        context_limit=None,
    )
    stats = {b.branch: b.stats for b in out.result.branches}
    assert stats[Branch.REJECTED_SEQUENCE].max_row_id == "train:1"  # rejected-only outlier
    plan = plan_batches(load_lengths(out.artifact_path), resolved(Objective.DPO, 2), seed=42)
    longest = max(stats[Branch.PAIR_MAX].max, 0)
    assert plan.worst_case.padded_length == longest
    assert plan.worst_case.token_slots == 4 * longest
    assert "train:1" in plan.worst_case.source_row_ids


def test_long_rows_are_kept_whole_and_flagged_against_a_small_context(
    handle, make_stream, make_ctx
) -> None:
    adapter = get_adapter(Objective.SFT, handle, MAPPING)
    out = full_scan(
        make_stream(ROWS),
        adapter,
        make_ctx(),
        objective=Objective.SFT,
        preprocess_key="pre_ctx",
        tokenizer_fingerprint="t",
        template_fingerprint=None,
        context_limit=4096,
    )
    seq = next(b for b in out.result.branches if b.branch is Branch.SEQUENCE).stats
    assert seq.max > 20_000 and out.result.context_exceeded_rows == 1


@pytest.mark.parametrize("decode_failure", [False, True])
def test_corrupt_row_is_kept_located_and_withholds_verification(
    handle, make_stream, make_ctx, decode_failure
) -> None:
    """plan §19.2 "손상 row": count and position kept, nothing reported as verified/ok."""
    rows = [dict(r) for r in ROWS]
    if not decode_failure:
        rows[1]["chosen"] = None  # decodes, but the mapped record cannot be built
    stream = make_stream(rows, decode_fail_at=1 if decode_failure else None)
    out = full_scan(
        stream,
        get_adapter(Objective.DPO, handle, MAPPING),
        make_ctx(),
        objective=Objective.DPO,
        preprocess_key="pre_corrupt",
        tokenizer_fingerprint="t",
        template_fingerprint=None,
        context_limit=262_144,
    )
    res = out.result
    assert (res.rows_seen, res.rows_ok, res.rows_failed) == (3, 2, 1)
    assert [f.row_id for f in res.failed_rows_sample] == ["train:1"]
    assert ErrorCode.SCAN_FAILED_ROWS in [i.code for i in out.issues]
    # every row was read, but one length is unknown: never complete (no complete badge)
    assert res.coverage is ScanCoverage.PARTIAL and res.rows_unprocessed == 0
    context = validate_context(
        res, model_declared_max=262_144, tokenizer=None, backend_verified_max=None
    )
    assert context.status == "unknown"  # the failed row's length is not known
    plan = plan_batches(load_lengths(out.artifact_path), resolved(Objective.DPO, 1), seed=42)
    assert len(load_lengths(out.artifact_path)) == 2 and plan.sampler.covers_all_rows
    audit = audit_preservation(
        res,
        context=context,
        batch_plan=plan,
        packing=False,
        scope=ScopeConfig(),
        template_content_loss_rows=out.template_content_loss_rows,
    )
    assert audit.status is DataPreservation.UNKNOWN
    full_read = next(c for c in audit.checks if c.name is PreservationCheckName.FULL_READ)
    assert full_read.passed is False
    codes = [v.code for v in audit.violations]
    assert ErrorCode.SCAN_FAILED_ROWS in codes and ErrorCode.SCAN_PARTIAL not in codes


def test_template_change_changes_the_preprocess_key(handle) -> None:
    adapter = get_adapter(Objective.SFT, handle, MAPPING)
    common = {
        "source_key": "src_x",
        "config": None,
        "split": "train",
        "mapping": MAPPING,
        "objective": "sft",
        "tokenizer_fingerprint": "tok",
        "adapter_name": adapter.name,
        "adapter_version": adapter.version,
        "dependency_lock_digest": "lock",
    }
    a = keys.preprocess_key(chat_template_sha256="59a64ebb", template_kwargs={}, **common)
    b = keys.preprocess_key(chat_template_sha256="00000000", template_kwargs={}, **common)
    c = keys.preprocess_key(
        chat_template_sha256="59a64ebb", template_kwargs={"enable_thinking": False}, **common
    )
    keep = keys.preprocess_key(
        chat_template_sha256="59a64ebb",
        template_kwargs={},
        **{
            **common,
            "mapping": MAPPING.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP}),
        },
    )
    assert len({a, b, c, keep}) == 4
