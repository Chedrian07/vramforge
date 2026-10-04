"""PreprocessingAdapter contract (plan.md §7.3, §8).

An adapter reproduces the pinned trainer's preprocessing (TRL 1.14.1, see docs/research/) for one
objective: mapping → conversational structure → the model's real chat template → tokenization
WITHOUT truncation. It returns only lengths/masks summaries, never token ids to persist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from vramforge_estimator.schemas import ErrorCode, Objective


@dataclass
class TokenizedRecord:
    row_id: str
    objective: Objective
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    sequence_tokens: int | None = None  # SFT full sequence fed to the model
    loss_token_count: int | None = None
    chosen_total_tokens: int | None = None  # DPO prompt + chosen as fed to the model
    rejected_total_tokens: int | None = None
    chosen_completion_tokens: int | None = None
    rejected_completion_tokens: int | None = None
    content_digest: str = ""
    # Template content-loss check (plan §7.4): False if a mapped message content does not appear
    # verbatim in the rendered text.
    template_preserved: bool | None = None
    template_issue: str | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def max_sequence_tokens(self) -> int | None:
        """The longest sequence this row contributes to a forward pass."""
        values = [
            v
            for v in (
                self.sequence_tokens,
                self.chosen_total_tokens,
                self.rejected_total_tokens,
                self.prompt_tokens,
            )
            if v is not None
        ]
        return max(values) if values else None


class PreprocessingAdapter(Protocol):
    name: str  # e.g. "trl-1.14.1-sft"
    version: str  # bump when the reproduced behavior changes (part of preprocess_key)
    objective: Objective

    def transformation_note(self) -> str:
        """Korean description of the data transformation applied (shown in the UI)."""
        ...

    def process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        """Tokenize one mapped row with no truncation; failures are returned, not raised."""
        ...
