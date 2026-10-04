"""Golden example stats (docs/research/example-model-dataset.md §5.8) reproduced by OUR adapters,
scanner and planner with the real MiMo tokenizer over the full CyberNative dataset.

Network test (Hugging Face Hub or a warm HF cache); no torch/trl involved. Run with:
``VRAMFORGE_NETWORK_TESTS=1 uv run --no-sync pytest tests/preprocessing_parity -m network``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.batching import plan_batches
from vramforge_estimator.inspection import SourceRow, TokenizerHandle
from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.scan import LengthTable, ScanLimits, full_scan, load_lengths
from vramforge_estimator.schemas import (
    Branch,
    BranchStats,
    ColumnMapping,
    EffectiveDtypes,
    EmptySystemPolicy,
    JobProgress,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ResolvedConfig,
    ScanCoverage,
    Strategy,
    TokenizerManifest,
    WorkspaceAssumptions,
)

pytestmark = pytest.mark.network

GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "golden" / "example_stats.json").read_text("utf-8")
)
MODEL, DATASET = GOLDEN["model"], GOLDEN["dataset"]


class ListStream:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.split = "train"
        self.config: str | None = None
        self.shards = [DATASET["file"]]
        self.total_rows: int | None = len(rows)
        self._done = False

    def __iter__(self) -> Iterator[SourceRow]:
        for i, row in enumerate(self.rows):
            yield SourceRow(row_index=i, shard_id=DATASET["file"], row=row)
        self._done = True

    @property
    def complete(self) -> bool:
        return self._done

    @property
    def shards_completed(self) -> int:
        return 1 if self._done else 0


@dataclass
class Ctx:
    artifact_dir: Path
    limits: ScanLimits = field(default_factory=ScanLimits)
    checkpoint: dict[str, Any] | None = None

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        pass

    def cancelled(self) -> bool:
        return False

    def load_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        self.checkpoint = data


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerHandle:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL["id"], revision=MODEL["revision"])
    template_sha = hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest()
    assert template_sha == MODEL["sha256"]["chat_template.jinja"]
    assert type(tok).__name__ == MODEL["tokenizer_class"] and len(tok) == MODEL["len_tokenizer"]
    manifest = TokenizerManifest(
        tokenizer_class=type(tok).__name__,
        vocab_size=len(tok),
        eos_token=tok.eos_token,
        pad_token=tok.pad_token,
        model_max_length=tok.model_max_length,
        chat_template_present=True,
        chat_template_sha256=template_sha,
        has_generation_markers=True,
        fingerprint=f"golden:{MODEL['revision']}",
    )
    return TokenizerHandle(tokenizer=tok, manifest=manifest)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET["id"], DATASET["file"], repo_type="dataset", revision=DATASET["revision"]
    )
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == DATASET["file_sha256"]
    parsed = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    assert len(parsed) == DATASET["rows"]
    return parsed


@pytest.fixture(scope="module")
def scans(tokenizer, rows, tmp_path_factory) -> dict[tuple[Objective, str], Any]:
    out = {}
    for policy in (EmptySystemPolicy.OMIT, EmptySystemPolicy.KEEP):
        mapping = ColumnMapping(**GOLDEN["mapping"], empty_system_policy=policy)
        for objective in (Objective.SFT, Objective.DPO, Objective.GRPO):
            adapter = get_adapter(objective, tokenizer, mapping, empty_system_policy=policy)
            ctx = Ctx(tmp_path_factory.mktemp(f"{objective.value}-{policy.value}"))
            out[objective, policy.value] = full_scan(
                ListStream(rows),
                adapter,
                ctx,
                objective=objective,
                preprocess_key=f"golden-{objective.value}-{policy.value}",
                tokenizer_fingerprint=tokenizer.manifest.fingerprint,
                template_fingerprint=tokenizer.manifest.chat_template_sha256,
                context_limit=MODEL["max_position_embeddings"],
            )
    return out


def branch(outcome: Any, name: Branch) -> BranchStats:
    return next(b for b in outcome.result.branches if b.branch is name)


def assert_matches(stats: BranchStats, golden: dict[str, Any], *, full: bool) -> None:
    s = stats.stats
    assert (s.count, s.min, s.max, s.total_tokens) == (
        golden["count"],
        golden["min"],
        golden["max"],
        golden["total"],
    )
    assert s.max_row_id == f"train:{golden['max_rows'][0]}"  # first row reaching the max
    if full:
        assert round(s.mean, 3) == golden["mean"]
        nearest = golden["nearest_rank"]
        assert (s.p50, s.p90, s.p95, s.p99) == (
            nearest["p50"],
            nearest["p90"],
            nearest["p95"],
            nearest["p99"],
        )
        assert [r.row_id for r in stats.top_rows] == [f"train:{i}" for i in golden["top10_rows"]]


def test_every_row_is_processed_completely(scans) -> None:
    for outcome in scans.values():
        res = outcome.result
        assert res.coverage is ScanCoverage.COMPLETE and res.rows_ok == DATASET["rows"]
        assert res.rows_failed == 0 and res.context_exceeded_rows == 0
        assert outcome.template_content_loss_rows == 0 and outcome.issues == []


def test_grpo_prompt_stats(scans) -> None:
    g = GOLDEN["grpo_prompt"]
    assert_matches(branch(scans[Objective.GRPO, "omit"], Branch.PROMPT), g["omit"], full=True)
    assert_matches(branch(scans[Objective.GRPO, "keep"], Branch.PROMPT), g["keep"], full=False)


def test_sft_stats(scans) -> None:
    g = GOLDEN["sft"]
    omit = scans[Objective.SFT, "omit"]
    assert_matches(branch(omit, Branch.SEQUENCE), g["total_omit"], full=True)
    assert_matches(branch(omit, Branch.COMPLETION), g["completion_loss_tokens"], full=True)
    assert_matches(branch(omit, Branch.LOSS_TOKENS), g["completion_loss_tokens"], full=True)
    keep = scans[Objective.SFT, "keep"]
    assert_matches(branch(keep, Branch.SEQUENCE), g["total_keep"], full=False)


def test_dpo_stats(scans) -> None:
    g = GOLDEN["dpo"]
    omit = scans[Objective.DPO, "omit"]
    assert_matches(branch(omit, Branch.PROMPT), g["prompt_omit"], full=False)
    assert_matches(branch(omit, Branch.CHOSEN_SEQUENCE), g["chosen_total_omit"], full=True)
    assert_matches(branch(omit, Branch.REJECTED_SEQUENCE), g["rejected_total_omit"], full=True)
    assert_matches(branch(omit, Branch.PAIR_MAX), g["pair_max_omit"], full=True)
    keep = scans[Objective.DPO, "keep"]
    assert_matches(branch(keep, Branch.CHOSEN_SEQUENCE), g["chosen_total_keep"], full=False)
    assert_matches(branch(keep, Branch.REJECTED_SEQUENCE), g["rejected_total_keep"], full=False)
    assert_matches(branch(keep, Branch.PAIR_MAX), g["pair_max_keep"], full=False)


def test_rows_over_trl_default_lengths(scans) -> None:
    sft = load_lengths(scans[Objective.SFT, "omit"].artifact_path)
    dpo = load_lengths(scans[Objective.DPO, "omit"].artifact_path)
    pair = [
        max(c, r) for c, r in zip(dpo.chosen_total_tokens, dpo.rejected_total_tokens, strict=True)
    ]
    for limit, expected in GOLDEN["rows_over_length"].items():
        assert sum(1 for n in sft.sequence_tokens if n > int(limit)) == expected["sft_total"]
        assert sum(1 for n in pair if n > int(limit)) == expected["dpo_pair_max"]


def test_layout_invariants_hold_for_every_row(scans, tokenizer, rows) -> None:
    tok = tokenizer.tokenizer
    c = GOLDEN["constants_tokens"]
    grpo = load_lengths(scans[Objective.GRPO, "omit"].artifact_path).prompt_tokens
    grpo_keep = load_lengths(scans[Objective.GRPO, "keep"].artifact_path).prompt_tokens
    sft = load_lengths(scans[Objective.SFT, "omit"].artifact_path)
    for i, row in enumerate(rows):
        n_q = len(tok(row["question"], add_special_tokens=False)["input_ids"])
        n_c = len(tok(row["chosen"], add_special_tokens=False)["input_ids"])
        assert grpo[i] == n_q + c["prompt_overhead_omit"]
        assert grpo_keep[i] == grpo[i] + c["empty_system_block"]
        assert sft.completion_tokens[i] == n_c + c["completion_overhead"]
        assert sft.sequence_tokens[i] == n_q + n_c + c["sft_total_minus_content"]


def _subset(table: LengthTable, indices: list[int]) -> LengthTable:
    pick = [table.row_ids.index(f"train:{i}") for i in indices]
    out = LengthTable()
    for name in (
        "row_ids",
        "prompt_tokens",
        "completion_tokens",
        "sequence_tokens",
        "loss_token_count",
        "chosen_total_tokens",
        "rejected_total_tokens",
    ):
        setattr(out, name, [getattr(table, name)[j] for j in pick])
    return out


def _resolved(objective: Objective, microbatch: int) -> ResolvedConfig:
    dt = "bfloat16"
    return ResolvedConfig(
        objective=objective,
        strategy=Strategy.QLORA,
        loading_scope="text_only",
        load_dtype=dt,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=dt,
            compute=dt,
            adapter=dt,
            gradient=dt,
            optimizer_state=dt,
            logits="float32",
            loss="float32",
            kv_cache=dt,
            recurrent_state="float32",
        ),
        quantization=QuantizationResolved(enabled=True, method="bnb_nf4"),
        optimizer=OptimizerResolved(name="adamw_torch_fused", states_per_param=2, state_dtype=dt),
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


def test_collator_shapes(scans) -> None:
    shapes = GOLDEN["collator_shapes"]
    sft = load_lengths(scans[Objective.SFT, "omit"].artifact_path)
    dpo = load_lengths(scans[Objective.DPO, "omit"].artifact_path)
    for table, objective, indices, key in (
        (sft, Objective.SFT, [2355, 2905], "sft_rows_2355_2905"),
        (dpo, Objective.DPO, [2355, 3169], "dpo_rows_2355_3169"),
        (dpo, Objective.DPO, [227], "dpo_row_227"),
    ):
        plan = plan_batches(_subset(table, indices), _resolved(objective, len(indices)), seed=42)
        worst = plan.worst_case
        assert [worst.sequences_per_forward, worst.padded_length] == shapes[key]
