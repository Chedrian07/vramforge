"""Batch plan contract (plan.md §8)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Issue, Objective, VFModel


class SequenceShape(VFModel):
    """One forward pass as seen by an architecture adapter."""

    batch: int  # sequences in the forward (DPO: 2 × pairs)
    seq_len: int  # padded length T
    logits_positions_per_sequence: int | None = None  # logits_to_keep; None = all positions


class BatchShape(VFModel):
    name: str  # "worst_case", "sampler_max", "budget_1024", ...
    objective: Objective
    rows_per_microbatch: int  # SFT samples / DPO pairs / GRPO completions per update microbatch
    sequences_per_forward: int
    padded_length: int
    token_slots: int  # sequences_per_forward × padded_length
    logits_positions: int  # total positions projected by the LM head per forward
    prompt_length: int | None = None
    completion_length: int | None = None
    source_row_ids: list[str] = Field(default_factory=list)
    description: str


class SamplerPlan(VFModel):
    kind: str  # e.g. "random", "repeat_sampler"
    seed: int | None = None
    drop_last: bool = False
    covers_all_rows: bool
    dropped_rows: int = 0
    note: str = ""


class GrpoBatchPlan(VFModel):
    """Plan §8.3 symbols: U, G, C, B_update, K."""

    unique_prompts_per_generation: int  # U
    num_generations: int  # G
    live_sequences: int  # C (concurrently generated sequences per device)
    update_microbatch: int  # B_update
    accumulation: int  # K
    generation_batch_size: int
    steps_per_generation: int
    num_iterations: int
    completion_budgets: list[int]
    max_prompt_length: int


class BatchPlan(VFModel):
    objective: Objective
    unit: Literal["samples", "pairs", "completions"]
    microbatch: int
    accumulation: int
    effective_batch: int
    pad_to_multiple_of: int | None = None
    padding_side: Literal["left", "right"] = "right"
    sampler: SamplerPlan
    worst_case: BatchShape  # structural conservative shape (longest rows together)
    sampler_max: BatchShape | None = None  # largest shape in the seeded sampler order
    scenarios: list[BatchShape] = Field(default_factory=list)  # GRPO: one per completion budget
    grpo: GrpoBatchPlan | None = None
    issues: list[Issue] = Field(default_factory=list)
    batch_key: str
