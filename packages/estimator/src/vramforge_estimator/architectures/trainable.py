"""LoRA target resolution, trainability and trainable parameter groups (plan.md §9.4).

PEFT 0.21.2 semantics (docs/research/loading-quantization-peft.md §5): A `(r, in)` and
B `(out, r)` per targeted Linear from the real module dims, `all-linear` = every Linear except the
output embedding (vision tower included), modules_to_save = frozen original + trainable deep copy,
DoRA adds an `(out,)` magnitude vector per targeted Linear. TRL 1.14.1 casts every trainable
parameter of a quantized model to bf16 (§Q6.1).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    ErrorCode,
    LinearModule,
    ModelComponent,
    ResolvedConfig,
    Stage,
    Strategy,
    TensorInfo,
    TensorRole,
)
from vramforge_estimator.units import canonical_dtype

from . import matching
from .base import TrainableGroup
from .structure import EXECUTED_COMPONENTS, ModelStructure, arch_issue, module_ancestors
from .weights import quantized_modules, resident_dtype

# Characters that only occur in a regex, never in a module name (dots alone stay suffixes).
_REGEX_SYNTAX = re.compile(r"[*+?\[\](){}|^$\\]")

NO_GRAD_NOTE = (
    "텍스트 전용 데이터에서는 실행되지 않아 gradient와 optimizer state가 생기지 않습니다 "
    "(가중치만 상주)."
)


@dataclass(frozen=True)
class ArchTrainableGroup(TrainableGroup):
    """`TrainableGroup` plus whether the parameters ever receive a gradient.

    Parameters of modules that never run on text-only data (vision tower) are allocated but get
    no gradient and no optimizer state (research §Q8.4, verified E5d/V6).
    """

    receives_grad: bool = True
    component: str = "text"


def config_error(message: str, **details: object) -> EstimatorError:
    return arch_issue(ErrorCode.CONFLICTING_OPTIONS, message, Stage.ESTIMATING, "lora", details)


# ---------------------------------------------------------------- LoRA target resolution


def resolve_lora_targets(
    structure: ModelStructure,
    target: str | list[str],
    exclude: Iterable[str],
    verified_kinds: frozenset[str],
) -> list[LinearModule]:
    """Resolve a LoRA target spec against the WHOLE inventory (full-checkpoint names).

    - "auto_verified": the adapter's verified text-decoder Linear kinds (no lm_head, embeddings
      or vision tower)
    - "all-linear": every Linear except the output embedding, vision Linear included
      (tuners_utils.py:2430-2492)
    - other str: one regex, `re.fullmatch` on module names; list: exact name or `.suffix`
    - a one-element list holding regex-only syntax (`*+?[](){}|^$\\`) is that single regex: the
      request schema carries "explicit suffixes or a single regex string" in one list field
    - `exclude` (list semantics) is applied last
    Callers apply the loading scope (vision modules do not exist in a text-only load).
    """
    if isinstance(target, list) and len(target) == 1 and _REGEX_SYNTAX.search(target[0]):
        target = target[0]
    if target == "auto_verified":
        chosen = [m for m in structure.text_linears() if m.kind in verified_kinds]
    elif target == "all-linear":
        chosen = [m for m in structure.linear_modules if m.name != structure.output_embedding]
    else:
        if isinstance(target, str):
            matched = sorted(
                n for n in structure.module_names if matching.peft_regex_match(n, target)
            )
        else:
            matched = sorted(
                n for n in structure.module_names if matching.peft_name_match(n, target)
            )
        non_linear = [n for n in matched if n not in structure.linear_by_name]
        if non_linear:
            raise config_error(
                "LoRA 대상에 Linear가 아닌 모듈이 포함되어 있습니다. 현재는 Linear 모듈만 "
                "계산할 수 있습니다.",
                modules=non_linear[:10],
            )
        names = set(matched)
        chosen = [m for m in structure.linear_modules if m.name in names]
    excluded = list(exclude)
    if excluded:
        chosen = [m for m in chosen if not matching.peft_name_match(m.name, excluded)]
    if not chosen:
        raise config_error("LoRA 대상 모듈을 하나도 찾지 못했습니다.", target=str(target)[:80])
    return chosen


# ---------------------------------------------------------------- trainability


@dataclass(frozen=True)
class Trainability:
    """Which parameters train in this configuration (shared by the activation formulas)."""

    full: bool  # strategy FULL: every loaded parameter trains
    lora_ranks: Mapping[str, int] = field(default_factory=dict)  # module -> rank
    saved_modules: frozenset[str] = frozenset()  # modules_to_save matches (outermost)
    dora: bool = False

    def module_trainable(self, module: str) -> bool:
        """The module's own weights train (full FT or inside a modules_to_save copy)."""
        if self.full:
            return True
        return any(module == m or module.startswith(m + ".") for m in self.saved_modules)

    def lora_rank(self, module: str) -> int | None:
        return self.lora_ranks.get(module)


def _outermost(names: Iterable[str]) -> list[str]:
    ordered = sorted(set(names), key=lambda n: (n.count("."), n))
    kept: list[str] = []
    for n in ordered:
        if not any(n.startswith(k + ".") for k in kept):
            kept.append(n)
    return sorted(kept)


def saved_module_names(structure: ModelStructure, cfg: ResolvedConfig) -> list[str]:
    """Modules wrapped by ModulesToSaveWrapper in the loading scope (outermost matches)."""
    entries = list(cfg.lora.modules_to_save) if cfg.lora else []
    if not entries:
        return []
    loaded_modules = {
        a for t in structure.loaded_tensors(cfg.loading_scope) for a in module_ancestors(t.module)
    }
    matched = [n for n in loaded_modules if matching.modules_to_save_match(n, entries)]
    if structure.output_embedding and matching.modules_to_save_match(
        structure.output_embedding, entries
    ):
        matched.append(structure.output_embedding)  # tied lm_head has no own tensor
    nested = [n for n in matched if any(n != m and n.startswith(m + ".") for m in matched)]
    if nested:
        raise config_error(
            "modules_to_save 항목이 서로 포함 관계인 모듈을 동시에 선택합니다.",
            modules=sorted(nested)[:10],
        )
    return _outermost(matched)


def build_trainability(structure: ModelStructure, cfg: ResolvedConfig) -> Trainability:
    if cfg.strategy is Strategy.FULL:
        return Trainability(full=True)
    lora = cfg.lora
    if lora is None:
        raise config_error("LoRA/QLoRA 전략인데 해석된 LoRA 설정이 없습니다.")
    saved = saved_module_names(structure, cfg)
    ranks: dict[str, int] = {}
    unknown: list[str] = []
    for name in lora.target_modules:
        module = structure.linear_by_name.get(name)
        if module is None:
            unknown.append(name)
            continue
        if not structure.in_scope(module.component, cfg.loading_scope):
            continue  # not loaded in this scope (e.g. vision Linear under text_only)
        if matching.inside_modules_to_save(name, lora.modules_to_save):
            continue  # PEFT never wraps modules under a modules_to_save module
        ranks[name] = matching.rank_for(name, lora.rank_pattern, lora.r)
    if unknown:
        raise config_error(
            "해석된 LoRA 대상 모듈이 model inventory에 없습니다.", modules=sorted(unknown)[:10]
        )
    return Trainability(
        full=False, lora_ranks=ranks, saved_modules=frozenset(saved), dora=lora.use_dora
    )


# ---------------------------------------------------------------- trainable groups


def _shape(dims: Iterable[int]) -> str:
    return "x".join(str(d) for d in dims)


@dataclass
class _Group:
    kind: str
    dtype: str
    executed: bool
    component: str
    note: str
    numel: int = 0
    count: int = 0


class _GroupBuilder:
    """Aggregates tensors into groups keyed by (name, dtype, executed)."""

    def __init__(self) -> None:
        self._groups: dict[tuple[str, str, bool], _Group] = {}

    def add(
        self,
        *,
        name: str,
        kind: str,
        dtype: str,
        numel: int,
        executed: bool,
        component: str,
        note: str = "",
    ) -> None:
        group = self._groups.setdefault(
            (name, dtype, executed), _Group(kind, dtype, executed, component, note)
        )
        group.numel += numel
        group.count += 1

    def groups(self) -> list[TrainableGroup]:
        out: list[TrainableGroup] = []
        for (name, _, _), g in self._groups.items():
            note = g.note if g.executed else (g.note + " " + NO_GRAD_NOTE).strip()
            out.append(
                ArchTrainableGroup(
                    name=name,
                    kind=g.kind,  # type: ignore[arg-type]
                    numel=g.numel,
                    dtype=g.dtype,
                    tensor_count=g.count,
                    note=note,
                    receives_grad=g.executed,
                    component=g.component,
                )
            )
        return out


def _label(component: ModelComponent) -> str:
    return "text" if component in EXECUTED_COMPONENTS else component.value


def _trainable_extra_dtype(tensor: TensorInfo, cfg: ResolvedConfig) -> str:
    """dtype of trainable non-adapter params (bias, modules_to_save copies): TRL casts every
    trainable param of a quantized model to bf16; otherwise the original dtype (§5.2, §Q6.1)."""
    return "bfloat16" if cfg.quantization.enabled else resident_dtype(tensor, cfg)


def trainable_group_list(structure: ModelStructure, cfg: ResolvedConfig) -> list[TrainableGroup]:
    builder = _GroupBuilder()
    if cfg.strategy is Strategy.FULL:
        if cfg.quantization.enabled:
            raise config_error("4-bit로 로드한 모델은 adapter 없이 전체 학습할 수 없습니다.")
        for t in structure.loaded_tensors(cfg.loading_scope):
            label = _label(t.component)
            role = "embedding:" if t.role is TensorRole.EMBEDDING else ""
            builder.add(
                name=f"full:{label}:{role}{_shape(t.shape)}",
                kind="full",
                dtype=resident_dtype(t, cfg),
                numel=t.numel,
                executed=t.component in EXECUTED_COMPONENTS,
                component=label,
            )
        return builder.groups()

    trainability = build_trainability(structure, cfg)
    lora = cfg.lora
    assert lora is not None  # build_trainability raised otherwise
    adapter_dtype = canonical_dtype(cfg.effective_dtypes.adapter)
    for name, rank in trainability.lora_ranks.items():
        m = structure.linear_by_name[name]
        label = _label(m.component)
        executed = m.component in EXECUTED_COMPONENTS
        builder.add(
            name=f"lora:A:{label}:{rank}x{m.in_features}",
            kind="lora",
            dtype=adapter_dtype,
            numel=rank * m.in_features,
            executed=executed,
            component=label,
        )
        builder.add(
            name=f"lora:B:{label}:{m.out_features}x{rank}",
            kind="lora",
            dtype=adapter_dtype,
            numel=m.out_features * rank,
            executed=executed,
            component=label,
        )
        if lora.use_dora:
            builder.add(
                name=f"dora_magnitude:{label}:{m.out_features}",
                kind="lora",
                dtype=adapter_dtype,
                numel=m.out_features,
                executed=executed,
                component=label,
                note="DoRA magnitude vector (out_features,)",
            )

    loaded = structure.loaded_tensors(cfg.loading_scope)
    if lora.bias in ("all", "lora_only"):
        lora_mods = set(trainability.lora_ranks)
        for t in loaded:
            if not (t.name == "bias" or t.name.endswith(".bias")):
                continue
            if lora.bias == "lora_only" and t.module not in lora_mods:
                continue
            if trainability.module_trainable(t.module):
                continue  # counted with its modules_to_save copy
            label = _label(t.component)
            builder.add(
                name=f"bias:{label}:{_shape(t.shape)}",
                kind="bias",
                dtype=_trainable_extra_dtype(t, cfg),
                numel=t.numel,
                executed=t.component in EXECUTED_COMPONENTS,
                component=label,
                note=f"bias='{lora.bias}': 기본 가중치의 bias가 학습됩니다 (가중치는 base에 포함).",
            )

    qmods = quantized_modules(structure, cfg)
    for module in sorted(trainability.saved_modules):
        tensors = structure.tensors_under(module, cfg.loading_scope)
        if not tensors and module == structure.output_embedding:
            # A tied lm_head has no tensor of its own; its copy has the embedding's shape.
            tensors = [
                t
                for t in loaded
                if t.role is TensorRole.EMBEDDING and t.component in EXECUTED_COMPONENTS
            ][:1]
        if any(t.module in qmods for t in tensors):
            raise config_error(
                "modules_to_save가 4-bit로 양자화되는 모듈을 가리킵니다. Linear4bit 복사본 학습은 "
                "검증되지 않아 지원하지 않습니다.",
                module=module,
            )
        shapes = {tuple(t.shape) for t in tensors}
        for t in tensors:
            label = _label(t.component)
            suffix = f":{_shape(t.shape)}" if len(shapes) > 1 else ""
            builder.add(
                name=f"modules_to_save:{module}{suffix}",
                kind="modules_to_save",
                dtype=_trainable_extra_dtype(t, cfg),
                numel=t.numel,
                executed=t.component in EXECUTED_COMPONENTS,
                component=label,
                note="원본은 frozen으로 남고 학습 가능한 복사본이 추가됩니다.",
            )
    return builder.groups()
