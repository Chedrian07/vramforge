"""Which trainable parameters receive gradients and optimizer state.

The architecture adapter reports trainable groups (sizes and dtypes); this module maps each group
onto inventory tensors to find the part that is actually executed. TRL never freezes a vision
tower, and text-only data never runs it, so such trainable parameters keep their weights resident
but get no gradient and no AdamW state (docs/research/loading-quantization-peft.md §Q8.4,
trl-sft-dpo.md V-19, trl-grpo.md verification #15). When a group cannot be mapped exactly, the
whole group is treated as executed (conservative) and the slice says so.
"""

from __future__ import annotations

import re
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
    group: TrainableGroup
    executed_numel: int
    executed_tensors: int
    # (numel, is_embedding) per executed tensor when derivable from the inventory, else None
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


def _conservative(group: TrainableGroup, what: str) -> TrainableSlice:
    return TrainableSlice(
        group,
        group.numel,
        group.tensor_count,
        None,
        False,
        f"{what}: 그룹 크기를 inventory 모듈과 대응시키지 못해 전부 gradient를 받는 것으로 "
        "계산했습니다 (보수적).",
    )


def _slice(
    group: TrainableGroup,
    executed: list[tuple[int, bool]],
    total: int,
    what: str,
) -> TrainableSlice:
    executed_numel = sum(n for n, _ in executed)
    if group.numel == total:
        skipped = total - executed_numel
        note = (
            f"{what}: 텍스트 전용 데이터에서 실행되지 않는 {skipped:,}개 학습 파라미터는 "
            "가중치만 상주하고 gradient·optimizer state가 없습니다."
            if skipped
            else ""
        )
        return TrainableSlice(group, executed_numel, len(executed), tuple(executed), True, note)
    if group.numel == executed_numel:
        return TrainableSlice(group, executed_numel, len(executed), tuple(executed), True)
    return _conservative(group, what)


def _lora_slice(
    group: TrainableGroup, inventory: ModelInventory, cfg: ResolvedConfig
) -> TrainableSlice:
    total = 0
    executed: list[tuple[int, bool]] = []
    for module, rank in lora_targets(inventory, cfg):
        pair = (rank * module.in_features, module.out_features * rank)
        total += sum(pair)
        if module.component in EXECUTED_COMPONENTS:
            executed += [(pair[0], False), (pair[1], False)]
    executed_numel = sum(n for n, _ in executed)
    if 0 < group.numel == total - executed_numel != total:
        # The group is exactly the non-executed (e.g. vision) part of the adapter.
        return TrainableSlice(
            group,
            0,
            0,
            (),
            True,
            "LoRA: 텍스트 전용 데이터에서 실행되지 않는 모듈의 adapter입니다.",
        )
    return _slice(group, executed, total, "LoRA")


def _tensor_slice(group: TrainableGroup, tensors: list[TensorInfo], what: str) -> TrainableSlice:
    executed = [
        (t.numel, t.role is TensorRole.EMBEDDING)
        for t in tensors
        if t.component in EXECUTED_COMPONENTS
    ]
    return _slice(group, executed, sum(t.numel for t in tensors), what)


def executed_slices(
    groups: list[TrainableGroup], inventory: ModelInventory, cfg: ResolvedConfig
) -> list[TrainableSlice]:
    out: list[TrainableSlice] = []
    unique = [t for t in _unique_tensors(inventory) if _in_scope(t, cfg)]
    for group in groups:
        if group.kind == "lora":
            out.append(_lora_slice(group, inventory, cfg))
        elif group.kind == "full":
            out.append(_tensor_slice(group, unique, "Full fine-tuning"))
        elif group.kind == "modules_to_save" and cfg.lora is not None:
            wanted = [t for t in unique if _module_matches(t.module, cfg.lora.modules_to_save)]
            out.append(_tensor_slice(group, wanted, "modules_to_save"))
        else:
            out.append(_conservative(group, group.kind))
    return out


__all__ = [
    "EXECUTED_COMPONENTS",
    "TrainableSlice",
    "executed_slices",
    "lora_rank",
    "lora_targets",
    "rank_for",
]
