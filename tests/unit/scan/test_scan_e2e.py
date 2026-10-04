"""Real adapters + scanner + planner on the offline MiMo-template tokenizer (plan §19.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vramforge_estimator import keys
from vramforge_estimator.batching import plan_batches
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.scan import full_scan, load_lengths
from vramforge_estimator.schemas import (
    Branch,
    ColumnMapping,
    EffectiveDtypes,
    EmptySystemPolicy,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ResolvedConfig,
    ScanCoverage,
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
    assert "2개 row에서 system 메시지를 생략" in out.result.transformation_note


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
