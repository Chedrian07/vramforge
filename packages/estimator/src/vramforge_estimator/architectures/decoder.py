"""The two registered adapters: `dense_decoder` and `qwen3_5_hybrid` (plan.md §2.1, §6.4, §11.4).

Both share one implementation; they differ in the structure they accept (validated from module
shapes), their verified LoRA preset and the per-layer formulas (Llama-style RMSNorm vs Qwen3.5
zero-centered RMSNorm, gated attention, Gated DeltaNet layers).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

from vramforge_estimator.schemas import (
    AllocationSpec,
    ArchitectureFacts,
    LinearModule,
    ModelInventory,
    ResolvedConfig,
    SequenceShape,
    TensorRole,
)

from . import ledger
from .base import GenerationTimepoints, StepTimepoints, TrainableGroup
from .structure import (
    ATTENTION_PROJ,
    FULL_ATTENTION,
    GDN_PROJ,
    LINEAR_ATTENTION,
    MLP_PROJ,
    SILU_ACTS,
    SLIDING_ATTENTION,
    Family,
    ModelStructure,
    is_moe,
    structure_error,
    text_layer_types,
)
from .trainable import build_trainability, resolve_lora_targets, trainable_group_list
from .weights import (
    device_map_load_bytes,
    load_transient_allocations,
    resident_weight_allocations,
)

_LINEAR_ATTENTION_KEYS = (
    "num_key_heads",
    "num_value_heads",
    "key_head_dim",
    "value_head_dim",
    "conv_kernel_dim",
)


def _common_supports(facts: ArchitectureFacts) -> bool:
    if is_moe(facts) or facts.extra.get("is_encoder_decoder"):
        return False
    if not facts.num_attention_heads or facts.num_hidden_layers <= 0:
        return False
    act = facts.extra.get("hidden_act")
    if act is not None and str(act).lower() not in SILU_ACTS:
        return False  # the verified MLP is a SiLU-gated MLP
    return not (facts.layer_types and len(facts.layer_types) != facts.num_hidden_layers)


class DecoderAdapter:
    """Shared implementation of `ArchitectureAdapter` for pre-norm decoder-only text models."""

    adapter_id: str
    family: ClassVar[Family]
    verified_kinds: ClassVar[frozenset[str]]
    _cache: OrderedDict[tuple[str, int, int], ModelStructure]

    def __init__(self) -> None:
        self._cache = OrderedDict()

    def supports(self, facts: ArchitectureFacts) -> bool:
        raise NotImplementedError

    def structure(self, inventory: ModelInventory) -> ModelStructure:
        """Parsed structure (small LRU keyed by inventory hash and size)."""
        key = (inventory.inventory_hash, len(inventory.tensors), len(inventory.linear_modules))
        cached = self._cache.get(key)
        if cached is not None and cached.inventory is inventory:
            return cached
        if not self.supports(inventory.facts):
            raise structure_error(
                "등록된 adapter가 이 모델 구조를 지원하지 않습니다.", adapter=self.adapter_id
            )
        parsed = ModelStructure(inventory, self.family)
        self._cache[key] = parsed
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return parsed

    # ------------------------------------------------------------------ protocol

    def lora_target_modules(
        self, inventory: ModelInventory, target: str | list[str], exclude: list[str]
    ) -> list[LinearModule]:
        return resolve_lora_targets(self.structure(inventory), target, exclude, self.verified_kinds)

    def resident_weights(
        self, inventory: ModelInventory, cfg: ResolvedConfig, live_at: list[str]
    ) -> list[AllocationSpec]:
        return resident_weight_allocations(self.structure(inventory), cfg, live_at)

    def load_transient(
        self, inventory: ModelInventory, cfg: ResolvedConfig, live_at: list[str]
    ) -> list[AllocationSpec]:
        return load_transient_allocations(self.structure(inventory), cfg, live_at)

    def trainable_groups(
        self, inventory: ModelInventory, cfg: ResolvedConfig
    ) -> list[TrainableGroup]:
        return trainable_group_list(self.structure(inventory), cfg)

    def train_step_ledger(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        shape: SequenceShape,
        tps: StepTimepoints,
        prefix: str,
    ) -> list[AllocationSpec]:
        st = self.structure(inventory)
        return ledger.train_step(st, cfg, build_trainability(st, cfg), shape, tps, prefix)

    def no_grad_forward_ledger(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        shape: SequenceShape,
        live_at: list[str],
        prefix: str,
    ) -> list[AllocationSpec]:
        st = self.structure(inventory)
        return ledger.no_grad_forward(st, cfg, build_trainability(st, cfg), shape, live_at, prefix)

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
        st = self.structure(inventory)
        tr = build_trainability(st, cfg)
        return ledger.generation(st, cfg, tr, num_sequences, prompt_len, new_tokens, tps, prefix)

    def lm_head_dims(self, inventory: ModelInventory) -> tuple[int, int]:
        st = self.structure(inventory)
        head = st.linear_by_name.get(st.output_embedding or "")
        if head is not None:
            return head.in_features, head.out_features
        emb = next(
            (
                t
                for t in st.tensors
                if t.role is TensorRole.EMBEDDING
                and len(t.shape) == 2
                and t.shape[1] == st.dims.hidden
            ),
            None,
        )
        if emb is None:
            raise structure_error("출력 projection(lm_head)의 크기를 결정할 수 없습니다.")
        return emb.shape[1], emb.shape[0]  # tied lm_head shares the embedding

    # ------------------------------------------------------------------ extras

    def loading_budget_bytes(self, inventory: ModelInventory, cfg: ResolvedConfig) -> int:
        """`S_load` of transformers' `caching_allocator_warmup` / bnb `device_map="auto"` check
        (single GPU without max_memory loads only if free × 0.81 >= S_load, research §3.5)."""
        return device_map_load_bytes(self.structure(inventory), cfg)


class DenseDecoderAdapter(DecoderAdapter):
    adapter_id = "dense_decoder"
    family: ClassVar[Family] = "dense"
    verified_kinds = frozenset(ATTENTION_PROJ + MLP_PROJ)

    def supports(self, facts: ArchitectureFacts) -> bool:
        if not _common_supports(facts) or facts.linear_attention:
            return False
        types = set(text_layer_types(facts))
        if not types <= {FULL_ATTENTION, SLIDING_ATTENTION}:
            return False
        return not (SLIDING_ATTENTION in types and facts.sliding_window is None)


class Qwen35HybridAdapter(DecoderAdapter):
    adapter_id = "qwen3_5_hybrid"
    family: ClassVar[Family] = "qwen3_5"
    verified_kinds = frozenset(ATTENTION_PROJ + GDN_PROJ + MLP_PROJ)

    def supports(self, facts: ArchitectureFacts) -> bool:
        if not _common_supports(facts):
            return False
        types = set(facts.layer_types)
        if LINEAR_ATTENTION not in types or not types <= {LINEAR_ATTENTION, FULL_ATTENTION}:
            return False
        dims = facts.linear_attention
        return all(int(dims.get(k, 0) or 0) > 0 for k in _LINEAR_ATTENTION_KEYS)
