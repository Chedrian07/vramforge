"""Factories for ResolvedConfig and LengthTable used by the batch-planner tests."""

from __future__ import annotations

from typing import Any

import pytest

from vramforge_estimator.scan import LengthTable
from vramforge_estimator.schemas import (
    EffectiveDtypes,
    GrpoResolved,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ResolvedConfig,
    RewardKind,
    RolloutBackend,
    Strategy,
    WorkspaceAssumptions,
)


def resolved(
    objective: Objective,
    *,
    microbatch: int = 1,
    accumulation: int = 1,
    pad_to_multiple_of: int | None = None,
    loss_path: str = "chunked_nll",
    grpo: GrpoResolved | None = None,
) -> ResolvedConfig:
    bf16 = "bfloat16"
    return ResolvedConfig(
        objective=objective,
        strategy=Strategy.LORA,
        loading_scope="text_only",
        load_dtype=bf16,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=bf16,
            compute=bf16,
            adapter="float32",
            gradient="float32",
            optimizer_state="float32",
            logits="float32",
            loss="float32",
            kv_cache=bf16,
            recurrent_state="float32",
        ),
        quantization=QuantizationResolved(enabled=False),
        optimizer=OptimizerResolved(
            name="adamw_torch_fused", states_per_param=2, state_dtype="float32"
        ),
        microbatch=microbatch,
        accumulation=accumulation,
        pad_to_multiple_of=pad_to_multiple_of,
        gradient_checkpointing=True,
        checkpointing_granularity="per_decoder_layer",
        loss_path=loss_path,
        grpo=grpo,
        workspace=WorkspaceAssumptions(
            cuda_context_bytes=(1, 2),
            library_workspace_bytes=(1, 2),
            allocator_slack_fraction=(0.0, 0.1),
        ),
    )


def grpo(
    *,
    num_generations: int = 4,
    generation_batch_size: int = 4,
    steps_per_generation: int = 4,
    update_microbatch: int = 1,
    accumulation: int = 4,
    budgets: list[int] | None = None,
    live_sequences: int | None = None,
) -> GrpoResolved:
    return GrpoResolved(
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        steps_per_generation=steps_per_generation,
        num_iterations=1,
        completion_budgets=[1024, 2048, 4096, 8192] if budgets is None else budgets,
        budget_explicit=False,
        beta=0.0,
        reference_needed=False,
        reward_kind=RewardKind.UNSPECIFIED,
        rollout_backend=RolloutBackend.TRANSFORMERS_SHARED_POLICY,
        live_sequences=(update_microbatch * steps_per_generation)
        if live_sequences is None
        else live_sequences,
        update_microbatch=update_microbatch,
        accumulation=accumulation,
    )


def table(**columns: list[Any]) -> LengthTable:
    n = len(next(iter(columns.values())))
    out = LengthTable(row_ids=[f"train:{i}" for i in range(n)])
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "sequence_tokens",
        "loss_token_count",
        "chosen_total_tokens",
        "rejected_total_tokens",
    ):
        setattr(out, name, list(columns.get(name, [None] * n)))
    return out


@pytest.fixture
def make_resolved():
    return resolved


@pytest.fixture
def make_grpo():
    return grpo


@pytest.fixture
def make_table():
    return table
