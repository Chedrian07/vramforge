"""Structural view of a `ModelInventory` shared by the decoder adapters.

Reads only the inventory contract (ArchitectureFacts, TensorInfo, LinearModule) and answers which
text decoder layers exist (layer type, Linear modules, norms), the dimensions the formulas need,
and which tensors a loading scope keeps. Structure is validated from module shapes, never from
model-name substrings (plan.md §6.3, §11.4).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    ArchitectureFacts,
    ErrorCode,
    Issue,
    LinearModule,
    ModelComponent,
    ModelInventory,
    Severity,
    Stage,
    TensorInfo,
    TensorRole,
)

Family = Literal["dense", "qwen3_5"]
Scope = Literal["full_checkpoint", "text_only"]

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
LINEAR_ATTENTION = "linear_attention"

ATTENTION_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
GDN_PROJ = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

# Components executed on text-only data. Vision/audio towers and projectors are resident when
# loaded but never run (docs/research/architecture-memory.md §9).
EXECUTED_COMPONENTS = frozenset({ModelComponent.TEXT, ModelComponent.OTHER})
TOWER_COMPONENTS = frozenset(
    {ModelComponent.VISION, ModelComponent.PROJECTOR, ModelComponent.AUDIO}
)

# Config keys whose positive value marks a mixture-of-experts MLP (not supported here).
MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "n_shared_experts",
    "moe_layer_freq",
    "decoder_sparse_step",
)
# Qwen3_5TextConfig class default (docs/research/architecture-memory.md §1.1, VERIFIED).
QWEN35_DEFAULT_PARTIAL_ROTARY = 0.25
SILU_ACTS = frozenset({"silu", "swish"})

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def arch_issue(
    code: ErrorCode, message: str, stage: Stage, component: str, details: dict[str, object]
) -> EstimatorError:
    """An error with a Korean, display-safe message and structured details."""
    return EstimatorError(
        Issue(
            code=code,
            severity=Severity.ERROR,
            stage=stage,
            user_message=message,
            affected_component=component,
            details=details,
        )
    )


def structure_error(message: str, **details: object) -> EstimatorError:
    """UNSUPPORTED_ARCHITECTURE for a structure the adapters do not support."""
    return arch_issue(
        ErrorCode.UNSUPPORTED_ARCHITECTURE, message, Stage.INSPECTING, "architecture", details
    )


def is_moe(facts: ArchitectureFacts) -> bool:
    for key in MOE_KEYS:
        value = facts.extra.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, int | float) and value > 0:
            return True
    return False


def text_layer_types(facts: ArchitectureFacts) -> list[str]:
    """Per-layer types; without `layer_types` infer them like transformers 5.18
    `cache_utils.get_layer_types_and_kwargs` (sliding_window -> sliding, else full)."""
    if facts.layer_types:
        return list(facts.layer_types)
    kind = SLIDING_ATTENTION if facts.sliding_window is not None else FULL_ATTENTION
    return [kind] * facts.num_hidden_layers


def partial_rotary_factor(facts: ArchitectureFacts, default: float) -> float:
    for source in (facts.rope, facts.extra):
        value = source.get("partial_rotary_factor")
        if isinstance(value, int | float) and 0 < value <= 1:
            return float(value)
    return default


@dataclass(frozen=True)
class Dims:
    """Text decoder dimensions (symbols of docs/research/architecture-memory.md §0)."""

    hidden: int  # H
    intermediate: int  # I
    vocab: int  # V (config)
    heads: int  # nq
    kv_heads: int  # nkv
    head_dim: int  # d
    rotary_dim: int  # r
    sliding_window: int | None = None
    lin_key_heads: int = 0  # Hk
    lin_value_heads: int = 0  # Hv
    lin_key_dim: int = 0  # dk
    lin_value_dim: int = 0  # dv
    conv_kernel: int = 0  # K

    @property
    def lin_key_width(self) -> int:  # Kd = Hk*dk
        return self.lin_key_heads * self.lin_key_dim

    @property
    def lin_value_width(self) -> int:  # Vd = Hv*dv
        return self.lin_value_heads * self.lin_value_dim

    @property
    def conv_dim(self) -> int:  # C = 2*Kd + Vd
        return 2 * self.lin_key_width + self.lin_value_width


@dataclass(frozen=True)
class Layer:
    index: int
    layer_type: str
    linears: Mapping[str, LinearModule]  # kind -> module
    norms: Mapping[str, TensorInfo]  # norm module leaf -> weight tensor
    tensors: tuple[TensorInfo, ...]


def module_ancestors(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _leaf(module: str) -> str:
    return module.rpartition(".")[2]


class ModelStructure:
    """Parsed, validated text-decoder structure of one inventory."""

    def __init__(self, inventory: ModelInventory, family: Family) -> None:
        self.inventory = inventory
        self.facts = inventory.facts
        self.family = family
        self.layer_types = text_layer_types(self.facts)
        if len(self.layer_types) != self.facts.num_hidden_layers:
            raise structure_error(
                "config의 layer_types 수가 num_hidden_layers와 다릅니다.",
                layer_types=len(self.layer_types),
                num_hidden_layers=self.facts.num_hidden_layers,
            )
        self.tensors: tuple[TensorInfo, ...] = tuple(inventory.tensors)
        self.linear_modules: tuple[LinearModule, ...] = self._all_linears()
        self.linear_by_name: dict[str, LinearModule] = {m.name: m for m in self.linear_modules}
        self.output_embedding: str | None = self._output_embedding()
        self.tied_skip: frozenset[str] = self._tied_skip()
        self.module_names: frozenset[str] = frozenset(
            a for t in self.tensors for a in module_ancestors(t.module)
        )
        self.layers: tuple[Layer, ...] = self._layers()
        self.dims = self._dims()
        self._validate_rmsnorm()
        if family == "qwen3_5":
            self._validate_qwen35()
        else:
            self._validate_dense()

    # ------------------------------------------------------------------ parsing

    def _all_linears(self) -> tuple[LinearModule, ...]:
        linears = list(self.inventory.linear_modules)
        known = {m.name for m in linears}
        for t in self.tensors:
            if t.role is TensorRole.LM_HEAD and t.module not in known and len(t.shape) == 2:
                linears.append(
                    LinearModule(
                        name=t.module,
                        kind=_leaf(t.module),
                        in_features=t.shape[1],
                        out_features=t.shape[0],
                        has_bias=any(x.name == f"{t.module}.bias" for x in self.tensors),
                        component=t.component,
                        layer_index=None,
                        layer_type=None,
                        dtype=t.dtype,
                    )
                )
                known.add(t.module)
        return tuple(linears)

    def _output_embedding(self) -> str | None:
        """Module name of the LM head. A tied head is often not serialized; the CausalLM classes
        still own an `lm_head` module that shares the embedding (PEFT can wrap it)."""
        for t in self.tensors:
            if t.role is TensorRole.LM_HEAD:
                return t.module
        for m in self.linear_modules:
            if m.kind == "lm_head":
                return m.name
        if self.facts.tie_word_embeddings:
            present = {t.name for t in self.tensors}
            for group in self.inventory.tied_groups:
                for name in group:
                    if name not in present and name.removesuffix(".weight").endswith("lm_head"):
                        return name.removesuffix(".weight")
            return "lm_head"
        return None

    def _tied_skip(self) -> frozenset[str]:
        """Tensor names that alias an earlier member of a tied group (counted once)."""
        present = {t.name for t in self.tensors}
        skip: set[str] = set()
        groups = [list(g) for g in self.inventory.tied_groups]
        if not groups and self.facts.tie_word_embeddings:
            emb = [t.name for t in self.tensors if t.role is TensorRole.EMBEDDING]
            head = [t.name for t in self.tensors if t.role is TensorRole.LM_HEAD]
            text_emb = [n for n in emb if not n.endswith("pos_embed.weight")]
            if text_emb and head:
                groups.append([text_emb[0], *head])
        roles = {t.name: t.role for t in self.tensors}
        for group in groups:
            # keep the embedding (the real nn.Embedding parameter), drop the aliases
            members = sorted(
                (n for n in group if n in present),
                key=lambda n: roles.get(n) is not TensorRole.EMBEDDING,
            )
            skip.update(members[1:])
        return frozenset(skip)

    def _text_layer_of(self, component: ModelComponent, index: int | None, name: str) -> int | None:
        if component is not ModelComponent.TEXT:
            return None
        return index if index is not None else _layer_index(name)

    def _layers(self) -> tuple[Layer, ...]:
        n = self.facts.num_hidden_layers
        linears: list[dict[str, LinearModule]] = [{} for _ in range(n)]
        norms: list[dict[str, TensorInfo]] = [{} for _ in range(n)]
        tensors: list[list[TensorInfo]] = [[] for _ in range(n)]
        for m in self.linear_modules:
            li = self._text_layer_of(m.component, m.layer_index, m.name)
            if li is None:
                continue
            if not 0 <= li < n:
                raise structure_error("decoder layer 번호가 범위를 벗어났습니다.", module=m.name)
            if m.kind in linears[li]:
                raise structure_error(
                    "한 decoder layer에 같은 이름의 Linear가 둘 이상 있습니다.",
                    layer=li,
                    kind=m.kind,
                )
            linears[li][m.kind] = m
        for t in self.tensors:
            li = self._text_layer_of(t.component, t.layer_index, t.name)
            if li is None or not 0 <= li < n:
                continue
            tensors[li].append(t)
            if t.role is TensorRole.NORM and t.name.endswith(".weight"):
                norms[li][_leaf(t.module)] = t
        return tuple(
            Layer(i, self.layer_types[i], linears[i], norms[i], tuple(tensors[i])) for i in range(n)
        )

    def _dims(self) -> Dims:
        f = self.facts
        heads = f.num_attention_heads
        if not heads:
            raise structure_error("num_attention_heads가 없어 attention 구조를 해석할 수 없습니다.")
        kv_heads = f.num_key_value_heads or heads
        head_dim = f.head_dim
        if head_dim is None:
            o_proj = next(
                (ly.linears.get("o_proj") for ly in self.layers if "o_proj" in ly.linears), None
            )
            head_dim = o_proj.in_features // heads if o_proj else f.hidden_size // heads
        intermediate = f.intermediate_size
        if intermediate is None:
            gate = next(
                (ly.linears.get("gate_proj") for ly in self.layers if "gate_proj" in ly.linears),
                None,
            )
            if gate is None:
                raise structure_error("intermediate_size를 결정할 수 없습니다.")
            intermediate = gate.out_features
        default_prf = QWEN35_DEFAULT_PARTIAL_ROTARY if self.family == "qwen3_5" else 1.0
        rotary = int(head_dim * partial_rotary_factor(f, default_prf))
        lin = f.linear_attention
        return Dims(
            hidden=f.hidden_size,
            intermediate=intermediate,
            vocab=f.vocab_size,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary,
            sliding_window=f.sliding_window,
            lin_key_heads=int(lin.get("num_key_heads", 0)),
            lin_value_heads=int(lin.get("num_value_heads", 0)),
            lin_key_dim=int(lin.get("key_head_dim", 0)),
            lin_value_dim=int(lin.get("value_head_dim", 0)),
            conv_kernel=int(lin.get("conv_kernel_dim", 0)),
        )

    # ------------------------------------------------------------------ validation

    def _validate_rmsnorm(self) -> None:
        """The verified saved sets are RMSNorm ones (architecture-memory.md §5). A norm with a
        bias is a LayerNorm (e.g. StableLM with Llama module names): different saved tensors."""
        biased = sorted(
            {
                t.module
                for t in self.tensors
                if t.role is TensorRole.NORM
                and t.component in EXECUTED_COMPONENTS
                and t.name.endswith(".bias")
            }
        )
        if biased:
            raise structure_error(
                "bias가 있는 norm(LayerNorm)은 검증된 RMSNorm 식과 저장 tensor가 달라 지원하지 "
                "않습니다.",
                norms=[_leaf(m) for m in biased[:10]],
            )

    def _expect(self, layer: Layer, kind: str, in_f: int, out_f: int) -> None:
        m = layer.linears.get(kind)
        if m is None:
            raise structure_error(
                "decoder layer에 필요한 Linear 모듈이 없습니다.", layer=layer.index, kind=kind
            )
        if (m.in_features, m.out_features) != (in_f, out_f):
            raise structure_error(
                "Linear 모듈의 shape이 config 차원과 맞지 않습니다.",
                layer=layer.index,
                kind=kind,
                expected=[out_f, in_f],
                actual=[m.out_features, m.in_features],
            )

    def _validate_common(self, layer: Layer, allowed: tuple[str, ...]) -> None:
        extra = sorted(set(layer.linears) - set(allowed))
        if extra:
            raise structure_error(
                "등록된 구조에 없는 Linear 모듈이 decoder layer에 있습니다 (MoE 등).",
                layer=layer.index,
                kinds=extra,
            )
        d = self.dims
        self._expect(layer, "gate_proj", d.hidden, d.intermediate)
        self._expect(layer, "up_proj", d.hidden, d.intermediate)
        self._expect(layer, "down_proj", d.intermediate, d.hidden)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            if norm not in layer.norms:
                raise structure_error(
                    "pre-norm decoder layer의 RMSNorm이 없습니다.", layer=layer.index, norm=norm
                )

    def _validate_qwen35(self) -> None:
        d = self.dims
        if not (d.lin_key_heads and d.lin_value_heads and d.lin_key_dim and d.lin_value_dim):
            raise structure_error("Gated DeltaNet 차원(linear_*)이 config에 없습니다.")
        if d.conv_kernel <= 0:
            raise structure_error("linear_conv_kernel_dim이 config에 없습니다.")
        for layer in self.layers:
            if layer.layer_type == FULL_ATTENTION:
                self._validate_common(layer, ATTENTION_PROJ + MLP_PROJ)
                # Qwen3.5 q_proj emits query and output gate together (§1.2).
                self._expect(layer, "q_proj", d.hidden, 2 * d.heads * d.head_dim)
                self._expect(layer, "k_proj", d.hidden, d.kv_heads * d.head_dim)
                self._expect(layer, "v_proj", d.hidden, d.kv_heads * d.head_dim)
                self._expect(layer, "o_proj", d.heads * d.head_dim, d.hidden)
                for norm in ("q_norm", "k_norm"):
                    t = layer.norms.get(norm)
                    if t is None or t.shape != [d.head_dim]:
                        raise structure_error(
                            "full-attention layer의 head 단위 q_norm/k_norm이 없습니다.",
                            layer=layer.index,
                        )
            elif layer.layer_type == LINEAR_ATTENTION:
                self._validate_common(layer, GDN_PROJ + MLP_PROJ)
                self._expect(layer, "in_proj_qkv", d.hidden, d.conv_dim)
                self._expect(layer, "in_proj_z", d.hidden, d.lin_value_width)
                self._expect(layer, "in_proj_a", d.hidden, d.lin_value_heads)
                self._expect(layer, "in_proj_b", d.hidden, d.lin_value_heads)
                self._expect(layer, "out_proj", d.lin_value_width, d.hidden)
                if "norm" not in layer.norms:
                    raise structure_error(
                        "Gated DeltaNet의 gated RMSNorm이 없습니다.", layer=layer.index
                    )
            else:
                raise structure_error(
                    "Qwen3.5 hybrid adapter가 지원하지 않는 layer type입니다.",
                    layer=layer.index,
                    layer_type=layer.layer_type,
                )

    def _validate_dense(self) -> None:
        d = self.dims
        if d.lin_value_heads or LINEAR_ATTENTION in self.layer_types:
            raise structure_error("linear-attention layer가 있어 dense adapter를 쓸 수 없습니다.")
        for layer in self.layers:
            if layer.layer_type not in (FULL_ATTENTION, SLIDING_ATTENTION):
                raise structure_error(
                    "dense adapter가 지원하지 않는 layer type입니다.",
                    layer=layer.index,
                    layer_type=layer.layer_type,
                )
            if layer.layer_type == SLIDING_ATTENTION and d.sliding_window is None:
                raise structure_error("sliding_attention layer인데 sliding_window가 없습니다.")
            self._validate_common(layer, ATTENTION_PROJ + MLP_PROJ)
            self._expect(layer, "q_proj", d.hidden, d.heads * d.head_dim)
            self._expect(layer, "k_proj", d.hidden, d.kv_heads * d.head_dim)
            self._expect(layer, "v_proj", d.hidden, d.kv_heads * d.head_dim)
            self._expect(layer, "o_proj", d.heads * d.head_dim, d.hidden)
            unknown = sorted(
                set(layer.norms)
                - {"input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"}
            )
            if unknown:
                raise structure_error(
                    "등록된 dense 구조에 없는 norm이 있습니다.", layer=layer.index, norms=unknown
                )

    # ------------------------------------------------------------------ queries

    def count(self, layer_type: str) -> int:
        return sum(1 for t in self.layer_types if t == layer_type)

    @property
    def final_norm(self) -> TensorInfo | None:
        """The text model's final RMSNorm weight (text norm outside the decoder layers)."""
        return next(
            (
                t
                for t in self.tensors
                if t.role is TensorRole.NORM
                and t.component in EXECUTED_COMPONENTS
                and t.name.endswith(".weight")
                and t.shape == [self.dims.hidden]
                and self._text_layer_of(t.component, t.layer_index, t.name) is None
            ),
            None,
        )

    def qk_norm_shape(self, layer: Layer) -> tuple[int, int] | None:
        """(rows per token, width) of the q_norm; None when the layer has no q/k norm."""
        t = layer.norms.get("q_norm")
        if t is None:
            return None
        width = t.shape[0]
        return (self.dims.heads * self.dims.head_dim // width, width)

    @staticmethod
    def in_scope(component: ModelComponent, scope: Scope) -> bool:
        if component is ModelComponent.MTP:
            return False  # never loaded: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`
        if scope == "text_only":
            return component not in TOWER_COMPONENTS
        return True

    def loaded_tensors(self, scope: Scope) -> list[TensorInfo]:
        """Tensors resident after loading in `scope`, tied aliases counted once."""
        return [
            t
            for t in self.tensors
            if t.name not in self.tied_skip and self.in_scope(t.component, scope)
        ]

    def linear_in_scope(self, module: LinearModule, scope: Scope) -> bool:
        return self.in_scope(module.component, scope)

    def tensors_under(self, module: str, scope: Scope) -> list[TensorInfo]:
        prefix = module + "."
        return [
            t
            for t in self.loaded_tensors(scope)
            if t.module == module or t.module.startswith(prefix)
        ]

    def text_linears(self) -> list[LinearModule]:
        return [m for ly in self.layers for m in ly.linears.values()]
