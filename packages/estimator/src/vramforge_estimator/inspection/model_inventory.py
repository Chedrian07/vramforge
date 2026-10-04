"""Tensor inventory from safetensors headers + config (plan.md §6.2–6.3).

Every serialized tensor becomes a `TensorInfo` (component, role, module, layer index). 2-D
``*.weight`` tensors of Linear-like modules become `LinearModule`s (shape ``[out, in]``) — the
unit for quantization and LoRA accounting; embeddings and ``lm_head`` are not Linear modules
here. Tied embeddings are counted once in ``params_total``. Pre-quantized checkpoints
(quantization-only dtypes such as E8M0 scales or FP4/FP6 payloads, quantizer tensor names, FP8
tensors, integer weight matrices) are refused instead of being inventoried as dense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.keys import stable_hash
from vramforge_estimator.schemas import (
    ArchitectureFacts,
    ComponentParams,
    ErrorCode,
    LinearModule,
    ModelComponent,
    ModelInventory,
    Stage,
    TensorInfo,
    TensorRole,
)
from vramforge_estimator.sources.issues import blocking_error

from .model_config import CheckpointEvidence, ParsedConfig, build_facts
from .safetensors_header import HeaderTensor

INVENTORY_VERSION = "vramforge.model-inventory.v1"

_MTP_SEGMENTS = frozenset({"mtp", "mtp_layers"})
_VISION_SEGMENTS = frozenset(
    {
        "visual",
        "vision",
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "image_encoder",
        "vision_embed_tokens",
    }
)
_AUDIO_SEGMENTS = frozenset({"audio", "audio_tower", "audio_model", "audio_encoder"})
_PROJECTOR_SEGMENTS = frozenset(
    {"multi_modal_projector", "mm_projector", "projector", "connector", "merger"}
)
_EMBED_MODULES = frozenset(
    {"embed_tokens", "wte", "word_embeddings", "tok_embeddings", "embed_in", "shared"}
)
_POS_EMBED_MODULES = frozenset(
    {
        "pos_embed",
        "wpe",
        "position_embeddings",
        "embed_positions",
        "position_embedding",
        "pos_embedding",
    }
)
_LM_HEAD_MODULES = frozenset({"lm_head", "embed_out"})
_MLP_SEGMENTS = frozenset(
    {"mlp", "feed_forward", "ffn", "block_sparse_moe", "moe", "experts", "shared_expert"}
)
_MLP_LEAVES = frozenset(
    {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "w1",
        "w2",
        "w3",
        "fc1",
        "fc2",
        "c_fc",
        "dense_h_to_4h",
        "dense_4h_to_h",
    }
)
_LAYER_CONTAINERS = frozenset({"layers", "blocks", "layer", "block", "h"})
_NORM_LEAF = re.compile(r"(norm|^ln(_\w+|\d+)?$)")
# HF Conv1D modules store weights as [in, out] (GPT-2 family).
_CONV1D_TRANSPOSED = {"gpt2": frozenset({"c_attn", "c_proj", "c_fc", "q_attn"})}
# Tensor names of pre-quantized checkpoints (GPTQ/AWQ/bitsandbytes/fp8 scales).
_QUANT_SUFFIXES = {
    ".qweight": "gptq/awq",
    ".qzeros": "gptq/awq",
    ".g_idx": "gptq",
    ".weight_scale": "fp8",
    ".weight_scale_inv": "fp8",
    ".absmax": "bitsandbytes",
    ".quant_map": "bitsandbytes",
    ".nested_absmax": "bitsandbytes",
    ".SCB": "bitsandbytes",
}
# Dtypes that only quantized storage uses: OCP Microscaling (MX) E8M0 block scales and FP4/FP6
# element payloads. Their presence alone marks a pre-quantized checkpoint.
_MX_SCALE_DTYPE = "float8_e8m0fnu"
_FP4_DTYPES = frozenset({"float4_e2m1"})
_FP6_DTYPES = frozenset({"float6_e2m3", "float6_e3m2"})
_QUANT_ONLY_DTYPES = frozenset({_MX_SCALE_DTYPE}) | _FP4_DTYPES | _FP6_DTYPES
_FP8_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2"})
# A weight matrix (Linear, embedding, lm_head) stored as integers is a packed quantized payload
# (GPTQ/AWQ/MLX/bitsandbytes...). Integer or complex tensors in other roles (buffers, indices,
# position ids) are ordinary dense data and are inventoried as such.
_INT_DTYPES = frozenset({"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"})
_MATRIX_ROLES = frozenset({TensorRole.LINEAR_WEIGHT, TensorRole.EMBEDDING, TensorRole.LM_HEAD})


@dataclass(frozen=True)
class ShardTensors:
    shard: str
    tensors: list[HeaderTensor]


def _error(code: ErrorCode, message: str, **details: object) -> EstimatorError:
    return blocking_error(code, message, stage=Stage.INSPECTING, component="model", details=details)


def component_of(name: str) -> ModelComponent:
    segments = name.split(".")
    if any(s in _MTP_SEGMENTS for s in segments):
        return ModelComponent.MTP
    if any(s in _VISION_SEGMENTS for s in segments):
        return ModelComponent.VISION
    if any(s in _AUDIO_SEGMENTS for s in segments):
        return ModelComponent.AUDIO
    if any(s in _PROJECTOR_SEGMENTS for s in segments):
        return ModelComponent.PROJECTOR
    return ModelComponent.TEXT


def layer_index_of(name: str) -> int | None:
    segments = name.split(".")
    for container, value in pairwise(segments):
        if container in _LAYER_CONTAINERS and value.isdigit():
            return int(value)
    return None


def _split(name: str) -> tuple[str, str, str]:
    """(module path, module leaf, parameter leaf)."""
    module, _, param = name.rpartition(".")
    return module, module.rpartition(".")[2], param


def role_of(name: str, ndim: int, conv_modules: set[str], matrix_modules: set[str]) -> TensorRole:
    """`matrix_modules`: modules with a 2-D ``weight``; `conv_modules`: weight with ndim >= 3."""
    module, leaf, param = _split(name)
    if leaf in _POS_EMBED_MODULES:
        return TensorRole.PARAMETER
    if leaf in _EMBED_MODULES and param == "weight" and ndim == 2:
        return TensorRole.EMBEDDING
    if (leaf in _LM_HEAD_MODULES or module == "output") and param in ("weight", "bias"):
        return TensorRole.LM_HEAD
    if param not in ("weight", "bias"):
        return TensorRole.OTHER if param in ("inv_freq", "position_ids") else TensorRole.PARAMETER
    if _NORM_LEAF.search(leaf):
        return TensorRole.NORM
    if module in conv_modules:
        return TensorRole.CONV
    if param == "weight" and ndim == 2:
        return TensorRole.LINEAR_WEIGHT
    if param == "bias" and ndim == 1 and module in matrix_modules:
        return TensorRole.LINEAR_BIAS
    return TensorRole.OTHER


@dataclass(frozen=True)
class _QuantizedEvidence:
    format: str  # e.g. "gptq/awq", "fp8", "mxfp4"
    reason: str  # "quantized_dtype" (from dtypes) | "prequantized_checkpoint" (tensor names)
    dtypes: tuple[str, ...] = ()  # the quantization-style dtypes found (dtype evidence only)


def _low_bit_format(dtypes: set[str]) -> str:
    """Format label: an "mx" prefix when E8M0 block scales are present, then the element type."""
    prefix = "mx" if _MX_SCALE_DTYPE in dtypes else ""
    if dtypes & _FP4_DTYPES:
        return f"{prefix}fp4"
    if dtypes & _FP6_DTYPES:
        return f"{prefix}fp6"
    if dtypes & _FP8_DTYPES:
        return f"{prefix}fp8"
    return prefix  # E8M0 scales next to integer-packed elements: microscaling, element unknown


def _quantized_format(tensors: list[TensorInfo]) -> _QuantizedEvidence | None:
    """Most decisive evidence first.

    Quantization-only dtypes (E8M0 scales, FP4/FP6 payloads) whatever the tensors are called,
    then quantizer tensor names, then FP8 tensors, then weight matrices stored as integers.
    """
    dtypes = {t.dtype for t in tensors}
    if dtypes & _QUANT_ONLY_DTYPES:
        found = dtypes & (_QUANT_ONLY_DTYPES | _FP8_DTYPES)
        return _QuantizedEvidence(_low_bit_format(dtypes), "quantized_dtype", tuple(sorted(found)))
    for tensor in tensors:
        for suffix, fmt in _QUANT_SUFFIXES.items():
            if tensor.name.endswith(suffix):
                return _QuantizedEvidence(fmt, "prequantized_checkpoint")
    if fp8 := dtypes & _FP8_DTYPES:
        return _QuantizedEvidence("fp8", "quantized_dtype", tuple(sorted(fp8)))
    packed = {t.dtype for t in tensors if t.role in _MATRIX_ROLES and t.dtype in _INT_DTYPES}
    if packed:
        return _QuantizedEvidence("integer-packed", "quantized_dtype", tuple(sorted(packed)))
    return None


def _layer_type(
    tensor: TensorInfo, leaf: str, segments: list[str], layer_types: list[str]
) -> str | None:
    if tensor.component is ModelComponent.VISION:
        return "vision"
    if tensor.component is not ModelComponent.TEXT:
        return tensor.component.value
    if leaf in _MLP_LEAVES or any(s in _MLP_SEGMENTS for s in segments):
        return "mlp"
    index = tensor.layer_index
    if index is not None and 0 <= index < len(layer_types):
        return layer_types[index]
    return None


def build_inventory(
    config: ParsedConfig,
    shards: list[ShardTensors],
    *,
    index_total_size: int | None,
    extra_facts: dict[str, Any] | None = None,
) -> ModelInventory:
    seen: set[str] = set()
    headers: list[tuple[str, HeaderTensor]] = []
    for shard in shards:
        for tensor in shard.tensors:
            if tensor.name in seen:
                raise _error(
                    ErrorCode.MODEL_METADATA_UNAVAILABLE,
                    "같은 이름의 tensor가 여러 shard에 있습니다.",
                    reason="duplicate_tensor",
                    tensor=tensor.name,
                )
            seen.add(tensor.name)
            headers.append((shard.shard, tensor))
    headers.sort(key=lambda item: item[1].name)

    weights = [t for _, t in headers if t.name.endswith(".weight")]
    conv_modules = {t.name.rpartition(".")[0] for t in weights if len(t.shape) >= 3}
    matrix_modules = {t.name.rpartition(".")[0] for t in weights if len(t.shape) == 2}
    tensors = [
        TensorInfo(
            name=t.name,
            dtype=t.dtype,
            shape=list(t.shape),
            numel=t.numel,
            nbytes=t.nbytes,
            shard=shard,
            component=component_of(t.name),
            role=role_of(t.name, len(t.shape), conv_modules, matrix_modules),
            module=t.name.rpartition(".")[0],
            layer_index=layer_index_of(t.name),
        )
        for shard, t in headers
    ]

    quantized = _quantized_format(tensors)
    if quantized is not None:
        evidence: dict[str, object] = {}
        cause = ""
        if quantized.dtypes:
            evidence["quantized_dtypes"] = list(quantized.dtypes)
            cause = f"(양자화 형식 dtype: {', '.join(quantized.dtypes)})"
        raise _error(
            ErrorCode.UNSUPPORTED_MODEL_FORMAT,
            f"이미 양자화된 checkpoint({quantized.format})는 지원하지 않습니다{cause}. "
            "원본 정밀도(bf16/fp16/fp32) safetensors 가중치가 필요합니다.",
            reason=quantized.reason,
            quantized_checkpoint_format=quantized.format,
            **evidence,
        )

    by_name = {t.name: t for t in tensors}
    text_embeddings = [
        t for t in tensors if t.role is TensorRole.EMBEDDING and t.component is ModelComponent.TEXT
    ]
    lm_heads = [t for t in tensors if t.role is TensorRole.LM_HEAD and t.name.endswith("weight")]
    dtype_params: dict[str, int] = {}
    for t in tensors:
        dtype_params[t.dtype] = dtype_params.get(t.dtype, 0) + t.numel
    evidence = CheckpointEvidence(
        has_vision_tensors=any(t.component is ModelComponent.VISION for t in tensors),
        has_mtp_tensors=any(t.component is ModelComponent.MTP for t in tensors),
        has_lm_head_tensor=bool(lm_heads),
        has_embedding_tensor=bool(text_embeddings),
        dtype_params=dtype_params,
    )
    facts = build_facts(config, evidence)
    facts = _with_tie_notes(facts, text_embeddings, lm_heads)
    if extra_facts:
        facts = facts.model_copy(update={"extra": {**facts.extra, **extra_facts}})

    tied_groups: list[list[str]] = []
    tied_duplicates: set[str] = set()
    if facts.tie_word_embeddings and len(text_embeddings) == 1 and len(lm_heads) <= 1:
        embedding = text_embeddings[0]
        if not lm_heads:
            tied_groups.append([embedding.name, "lm_head.weight"])
        elif lm_heads[0].shape == embedding.shape:
            # transformers re-ties the serialized head to the embedding: one resident storage.
            tied_groups.append([embedding.name, lm_heads[0].name])
            tied_duplicates.add(lm_heads[0].name)

    linear_modules = _linear_modules(tensors, by_name, facts, config)

    params_total = sum(t.numel for t in tensors if t.name not in tied_duplicates)
    bytes_total = sum(t.nbytes for t in tensors)
    components: dict[ModelComponent, list[int]] = {}
    for t in tensors:
        acc = components.setdefault(t.component, [0, 0])
        if t.name not in tied_duplicates:
            acc[0] += t.numel
        acc[1] += t.nbytes
    by_component = [
        ComponentParams(component=c, params=v[0], bytes_serialized=v[1])
        for c, v in sorted(components.items(), key=lambda kv: list(ModelComponent).index(kv[0]))
    ]

    inventory = ModelInventory(
        facts=facts,
        tensors=tensors,
        linear_modules=linear_modules,
        params_total=params_total,
        bytes_serialized_total=bytes_total,
        by_component=by_component,
        tied_groups=tied_groups,
        quantized_checkpoint_format=None,
        index_total_size=index_total_size,
        inventory_hash="pending",
    )
    payload = inventory.model_dump(mode="json", exclude={"inventory_hash"})
    return inventory.model_copy(
        update={"inventory_hash": stable_hash(INVENTORY_VERSION, payload, prefix="inv_")}
    )


def _with_tie_notes(
    facts: ArchitectureFacts, embeddings: list[TensorInfo], heads: list[TensorInfo]
) -> ArchitectureFacts:
    extra = dict(facts.extra)
    extra["lm_head_serialized"] = bool(heads)
    if not facts.tie_word_embeddings and embeddings and not heads:
        # Untied config but no head in the checkpoint: the loader would initialise it randomly.
        extra["lm_head_missing"] = True
    return facts.model_copy(update={"extra": extra})


def _linear_modules(
    tensors: list[TensorInfo],
    by_name: dict[str, TensorInfo],
    facts: ArchitectureFacts,
    config: ParsedConfig,
) -> list[LinearModule]:
    transposed = _CONV1D_TRANSPOSED.get(config.model_type or "", frozenset())
    modules: list[LinearModule] = []
    for tensor in tensors:
        if tensor.role is not TensorRole.LINEAR_WEIGHT:
            continue
        segments = tensor.module.split(".")
        leaf = segments[-1]
        out_features, in_features = tensor.shape
        if leaf in transposed:
            in_features, out_features = tensor.shape
        modules.append(
            LinearModule(
                name=tensor.module,
                kind=leaf,
                in_features=in_features,
                out_features=out_features,
                has_bias=f"{tensor.module}.bias" in by_name,
                component=tensor.component,
                layer_index=tensor.layer_index,
                layer_type=_layer_type(tensor, leaf, segments, facts.layer_types),
                dtype=tensor.dtype,
            )
        )
    return modules
