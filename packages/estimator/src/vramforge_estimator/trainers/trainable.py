"""Which trainable parameters receive gradients and optimizer state.

The architecture adapter reports trainable groups (sizes and dtypes). They are aggregated per
(kind, dtype) into one ledger entry each, and the part that actually executes is determined:
TRL never freezes a vision tower and text-only data never runs it, so such parameters keep their
weights resident but get no gradient and no AdamW state (docs/research/loading-quantization-peft.md
§Q8.4, trl-sft-dpo.md V-19, trl-grpo.md verification #15).

The contract flags drive the split: `TrainableGroup.receives_grad` says whether a group executes
and `is_embedding` marks nn.Embedding parameters (bnb 8-bit optimizers keep 32-bit state for
them); the architecture adapters of this repository set both. Groups whose `receives_grad` is None
are mapped onto inventory modules (tensor roles decide embeddings). When neither works the whole
bucket is treated as executed (conservative) and the slice says so.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from vramforge_estimator.architectures import TrainableGroup
from vramforge_estimator.schemas import (
    LinearModule,
    ModelComponent,
    ModelInventory,
    ResolvedConfig,
    TensorInfo,
    TensorRole,
)

# Components that a text-only training step executes. MTP layers are not built by transformers
# 5.18 (docs/research/example-model-dataset.md O7); vision/audio/projector never run on text data.
EXECUTED_COMPONENTS = frozenset({ModelComponent.TEXT, ModelComponent.OTHER})


@dataclass(frozen=True)
class TrainableSlice:
    name: str  # "lora", "full", "modules_to_save", "bias" (+ ":dtype" when a kind mixes dtypes)
    kind: str
    dtype: str
    numel: int  # all parameters of the bucket (resident weights)
    tensor_count: int
    executed_numel: int
    executed_tensors: int
    # (numel, is_embedding) per executed tensor when derivable, else None
    tensors: tuple[tuple[int, bool], ...] | None
    exact: bool
    note: str = ""


def rank_for(module_name: str, r: int, rank_pattern: dict[str, int]) -> int:
    """PEFT 0.21.2 `get_pattern_key`: first rank_pattern key with re.match(r"(.*\\.)?(key)$")."""
    for key, rank in rank_pattern.items():
        if re.match(rf"(.*\.)?({key})$", module_name):
            return rank
    return r


def lora_rank(module_name: str, cfg: ResolvedConfig) -> int:
    assert cfg.lora is not None
    return rank_for(module_name, cfg.lora.r, cfg.lora.rank_pattern)


def lora_targets(inventory: ModelInventory, cfg: ResolvedConfig) -> list[tuple[LinearModule, int]]:
    """Resolved LoRA target modules with their rank (A: r x in, B: out x r)."""
    if cfg.lora is None:
        return []
    by_name = {m.name: m for m in inventory.linear_modules}
    return [(by_name[n], lora_rank(n, cfg)) for n in cfg.lora.target_modules if n in by_name]


def _unique_tensors(inventory: ModelInventory) -> list[TensorInfo]:
    """Tensors in serialization order with tied groups counted once."""
    tied_followers = {name for group in inventory.tied_groups for name in group[1:]}
    return [t for t in inventory.tensors if t.name not in tied_followers]


def _in_scope(t: TensorInfo, cfg: ResolvedConfig) -> bool:
    if t.component is ModelComponent.MTP:
        return False
    return cfg.loading_scope == "full_checkpoint" or t.component is ModelComponent.TEXT


def _module_matches(module: str, targets: list[str]) -> bool:
    # PEFT `_set_trainable` wraps modules whose name ends with the key (no dot boundary).
    return any(module.endswith(key) for key in targets)


def _skipped_note(kind: str, skipped: int) -> str:
    if not skipped:
        return ""
    return (
        f"{kind}: 텍스트 전용 데이터에서 실행되지 않는 {skipped:,}개 학습 파라미터는 가중치만 "
        "상주하고 gradient·optimizer state가 없습니다."
    )


def _from_flags(
    name: str, kind: str, dtype: str, groups: Sequence[TrainableGroup]
) -> TrainableSlice:
    executed = [g for g in groups if g.receives_grad is not False]
    tensors: list[tuple[int, bool]] = []
    per_tensor_known = True
    for g in executed:
        # The adapters group tensors of one shape, so every tensor of a group has numel/count.
        if g.tensor_count > 0 and g.numel % g.tensor_count == 0:
            tensors += [(g.numel // g.tensor_count, g.is_embedding)] * g.tensor_count
        else:
            per_tensor_known = False
    numel = sum(g.numel for g in groups)
    executed_numel = sum(g.numel for g in executed)
    return TrainableSlice(
        name=name,
        kind=kind,
        dtype=dtype,
        numel=numel,
        tensor_count=sum(g.tensor_count for g in groups),
        executed_numel=executed_numel,
        executed_tensors=sum(g.tensor_count for g in executed),
        tensors=tuple(tensors) if per_tensor_known else None,
        exact=True,
        note=_skipped_note(kind, numel - executed_numel),
    )


def _derived(
    kind: str, inventory: ModelInventory, cfg: ResolvedConfig
) -> tuple[int | None, list[tuple[int, bool]]]:
    """(total numel, executed (numel, is_embedding) tensors) of a kind from the inventory."""
    if kind == "lora":
        total = 0
        executed: list[tuple[int, bool]] = []
        for module, rank in lora_targets(inventory, cfg):
            pair = (rank * module.in_features, module.out_features * rank)
            total += sum(pair)
            if module.component in EXECUTED_COMPONENTS:
                executed += [(pair[0], False), (pair[1], False)]
        return total, executed
    unique = [t for t in _unique_tensors(inventory) if _in_scope(t, cfg)]
    if kind == "full":
        tensors = unique
    elif kind == "modules_to_save" and cfg.lora is not None:
        tensors = [t for t in unique if _module_matches(t.module, cfg.lora.modules_to_save)]
    else:
        return None, []
    executed = [
        (t.numel, t.role is TensorRole.EMBEDDING)
        for t in tensors
        if t.component in EXECUTED_COMPONENTS
    ]
    return sum(t.numel for t in tensors), executed


def _from_inventory(
    name: str,
    kind: str,
    dtype: str,
    groups: Sequence[TrainableGroup],
    inventory: ModelInventory,
    cfg: ResolvedConfig,
) -> TrainableSlice:
    numel = sum(g.numel for g in groups)
    count = sum(g.tensor_count for g in groups)
    total, executed = _derived(kind, inventory, cfg)
    executed_numel = sum(n for n, _ in executed)
    if total is not None and numel in (total, executed_numel):
        return TrainableSlice(
            name=name,
            kind=kind,
            dtype=dtype,
            numel=numel,
            tensor_count=count,
            executed_numel=executed_numel,
            executed_tensors=len(executed),
            tensors=tuple(executed),
            exact=True,
            note=_skipped_note(kind, numel - executed_numel),
        )
    return TrainableSlice(
        name=name,
        kind=kind,
        dtype=dtype,
        numel=numel,
        tensor_count=count,
        executed_numel=numel,
        executed_tensors=count,
        tensors=None,
        exact=False,
        note=f"{kind}: 학습 파라미터를 inventory 모듈과 대응시키지 못해 전부 gradient를 받는 "
        "것으로 계산했습니다 (보수적).",
    )


def executed_slices(
    groups: list[TrainableGroup], inventory: ModelInventory, cfg: ResolvedConfig
) -> list[TrainableSlice]:
    buckets: dict[tuple[str, str], list[TrainableGroup]] = {}
    for g in groups:
        buckets.setdefault((g.kind, g.dtype), []).append(g)
    dtypes_per_kind: dict[str, int] = {}
    for kind, _ in buckets:
        dtypes_per_kind[kind] = dtypes_per_kind.get(kind, 0) + 1
    out: list[TrainableSlice] = []
    for (kind, dtype), members in buckets.items():
        name = kind if dtypes_per_kind[kind] == 1 else f"{kind}:{dtype}"
        if all(isinstance(g.receives_grad, bool) for g in members):
            out.append(_from_flags(name, kind, dtype, members))
        else:
            out.append(_from_inventory(name, kind, dtype, members, inventory, cfg))
    return out


__all__ = [
    "EXECUTED_COMPONENTS",
    "TrainableSlice",
    "executed_slices",
    "lora_rank",
    "lora_targets",
    "rank_for",
]
