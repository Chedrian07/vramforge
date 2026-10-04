"""ArchitectureAdapter contract (plan.md §9.3–9.5, §9.7, §14.1).

An architecture adapter knows the MODEL STRUCTURE only: which tensors are resident and in which
format, which modules are trainable, and what a forward/backward or a generation of a given shape
allocates. It does not know the trainer's procedure; the trainer adapter decides which passes
happen in which phase and hands the adapter the timepoint ids to attach allocations to.

Every returned `AllocationSpec` must carry evidence and a formula reference; sizes that cannot be
derived are `None` with a note (never 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from vramforge_estimator.schemas import (
    AllocationSpec,
    ArchitectureFacts,
    LinearModule,
    ModelInventory,
    ResolvedConfig,
    SequenceShape,
)


@dataclass(frozen=True)
class StepTimepoints:
    """Timepoint ids of one training forward+backward, chosen by the trainer adapter."""

    forward: str  # end of forward: every saved activation of the step is alive
    loss: str  # LM-head/loss computation (saved activations still alive)
    backward: str  # backward through the first (recomputed) layer: largest working set


@dataclass(frozen=True)
class GenerationTimepoints:
    prefill: str  # prompt prefill of all live sequences
    decode: str  # last decode step: caches at maximum length


@dataclass(frozen=True)
class TrainableGroup:
    """A group of trainable parameters (sizes drive gradients and optimizer states)."""

    name: str  # e.g. "lora", "modules_to_save:lm_head", "full"
    kind: Literal["lora", "modules_to_save", "full", "bias"]
    numel: int
    dtype: str  # parameter dtype while training
    tensor_count: int
    note: str = ""
    # False for parameters that never run on this data (e.g. vision LoRA on text-only data):
    # they are allocated but get no gradient and no optimizer state (research Q8.4).
    # None = not reported; trainer adapters then map the group onto the inventory themselves.
    receives_grad: bool | None = None
    component: str = "text"
    # nn.Embedding parameters (bnb 8-bit optimizers keep 32-bit state for them).
    is_embedding: bool = False


FINAL_HIDDEN_SUFFIX = "final_hidden"


def final_hidden_alias(prefix: str) -> str:
    """Storage-alias group of the final hidden state produced by `train_step_ledger(prefix=...)`.

    Trainer adapters reuse it so a trainable LM head's saved input is counted once."""
    return f"{prefix}.{FINAL_HIDDEN_SUFFIX}"


class ArchitectureAdapter(Protocol):
    adapter_id: str  # "dense_decoder" | "qwen3_5_hybrid"

    def supports(self, facts: ArchitectureFacts) -> bool: ...

    def lora_target_modules(
        self,
        inventory: ModelInventory,
        target: str | list[str],
        exclude: list[str],
    ) -> list[LinearModule]:
        """Resolve "auto_verified" / "all-linear" / explicit names (or one regex) to modules."""
        ...

    def resident_weights(
        self, inventory: ModelInventory, cfg: ResolvedConfig, live_at: list[str]
    ) -> list[AllocationSpec]:
        """Base model weights as resident on the device (4-bit payload + metadata, non-quantized
        modules in their effective dtype, fp32 upcasts, vision tower if in loading scope)."""
        ...

    def load_transient(
        self, inventory: ModelInventory, cfg: ResolvedConfig, live_at: list[str]
    ) -> list[AllocationSpec]:
        """Extra memory that exists only while loading/quantizing."""
        ...

    def trainable_groups(
        self, inventory: ModelInventory, cfg: ResolvedConfig
    ) -> list[TrainableGroup]:
        """LoRA adapters (from real module dims), modules_to_save copies, full fine-tune params."""
        ...

    def train_step_ledger(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        shape: SequenceShape,
        tps: StepTimepoints,
        prefix: str,
    ) -> list[AllocationSpec]:
        """Activations of one forward+backward from the embedding through the final norm,
        honoring gradient checkpointing and the per-layer-type kernel paths. Excludes the LM
        head and loss (trainer adapters account for those)."""
        ...

    def no_grad_forward_ledger(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        shape: SequenceShape,
        live_at: list[str],
        prefix: str,
    ) -> list[AllocationSpec]:
        """Working set of an inference-mode forward (e.g. reference or old-policy log-probs),
        excluding the LM head and loss."""
        ...

    def generation_ledger(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        num_sequences: int,
        prompt_len: int,
        new_tokens: int,
        tps: GenerationTimepoints,
        prefix: str,
    ) -> list[AllocationSpec]:
        """KV cache of attention layers, conv/recurrent state of linear-attention layers and the
        prefill/decode working set for `num_sequences` live sequences (excluding logits)."""
        ...

    def lm_head_dims(self, inventory: ModelInventory) -> tuple[int, int]:
        """(hidden_size, vocab rows of the output projection)."""
        ...

    def loading_budget_bytes(self, inventory: ModelInventory, cfg: ResolvedConfig) -> int:
        """S_load: device bytes from_pretrained must fit before training starts (4-bit modules at
        their packed size, everything else in the load dtype). With bnb 4-bit + device_map="auto"
        on one GPU and no max_memory, loading needs free memory x 0.81 >= S_load
        (docs/research/loading-quantization-peft.md §4.5)."""
        ...
