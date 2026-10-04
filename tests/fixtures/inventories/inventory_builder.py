"""Build `ModelInventory` fixtures from config + tensor headers (test-only, no network).

This mirrors what the model inspector produces (plan.md §6.2-6.3) closely enough for the
architecture adapters' tests, using only the public contract (`vramforge_estimator.schemas`):

- component: `model.visual.*` / `visual.*` -> vision, `mtp.*` -> mtp, everything else -> text
- role: embeddings, lm_head, norms (module leaf contains "norm"), conv weights, raw parameters
  (A_log, dt_bias), 2-D Linear weights and their biases
- `linear_modules`: every 2-D Linear weight, plus `lm_head` (kind "lm_head") unless disabled
- `tied_groups`: `[embed_tokens, lm_head]` when the config ties them

Load it from a test with `importlib` (tests/ has no packages); see
tests/unit/architectures/conftest.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vramforge_estimator.schemas import (
    ArchitectureFacts,
    ComponentParams,
    LayerTypeCount,
    LinearModule,
    ModelComponent,
    ModelInventory,
    TensorInfo,
    TensorRole,
)
from vramforge_estimator.units import canonical_dtype, tensor_bytes

HERE = Path(__file__).resolve().parent
MIMO_HEADERS = HERE / "mimo-v2.6-distill-qwen-9b.headers.json"

TensorRow = tuple[str, str, Sequence[int]]

_MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "decoder_sparse_step",
)
_EXTRA_KEYS = (
    "hidden_act",
    "attn_output_gate",
    "rms_norm_eps",
    "partial_rotary_factor",
    "full_attention_interval",
    "mtp_num_hidden_layers",
    "attention_bias",
    "attention_dropout",
    "mlp_bias",
    "is_encoder_decoder",
    *_MOE_KEYS,
)
_ATTN_KINDS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def text_config_of(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("text_config") or config)


def facts_from_config(config: dict[str, Any]) -> ArchitectureFacts:
    tc = text_config_of(config)
    n_layers = int(tc["num_hidden_layers"])
    layer_types = list(tc.get("layer_types") or [])
    counts: dict[str, int] = {}
    for lt in layer_types:
        counts[lt] = counts.get(lt, 0) + 1
    heads = tc.get("num_attention_heads")
    head_dim = tc.get("head_dim")
    source = "explicit" if head_dim else ("inferred" if heads else "unknown")
    if not head_dim and heads:
        head_dim = int(tc["hidden_size"]) // int(heads)
    linear: dict[str, int] = {}
    for key, src in (
        ("num_key_heads", "linear_num_key_heads"),
        ("num_value_heads", "linear_num_value_heads"),
        ("key_head_dim", "linear_key_head_dim"),
        ("value_head_dim", "linear_value_head_dim"),
        ("conv_kernel_dim", "linear_conv_kernel_dim"),
    ):
        if tc.get(src) is not None:
            linear[key] = int(tc[src])
    rope = dict(tc.get("rope_parameters") or tc.get("rope_scaling") or {})
    extra = {k: tc[k] for k in _EXTRA_KEYS if k in tc}
    dtype = tc.get("dtype") or tc.get("torch_dtype") or config.get("dtype")
    return ArchitectureFacts(
        architectures=list(config.get("architectures") or []),
        model_type=str(config.get("model_type", tc.get("model_type", ""))),
        text_model_type=tc.get("model_type") if "text_config" in config else None,
        num_hidden_layers=n_layers,
        hidden_size=int(tc["hidden_size"]),
        intermediate_size=tc.get("intermediate_size"),
        vocab_size=int(tc["vocab_size"]),
        num_attention_heads=heads,
        num_key_value_heads=tc.get("num_key_value_heads", heads),
        head_dim=head_dim,
        head_dim_source=source,
        layer_types=layer_types,
        layer_type_counts=[LayerTypeCount(layer_type=k, count=v) for k, v in counts.items()],
        sliding_window=tc.get("sliding_window"),
        linear_attention=linear,
        max_position_embeddings=tc.get("max_position_embeddings"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", tc.get("tie_word_embeddings"))),
        config_dtype=canonical_dtype(str(dtype)) if dtype else None,
        has_vision="vision_config" in config,
        has_mtp=int(tc.get("mtp_num_hidden_layers") or 0) > 0,
        rope=rope,
        extra=extra,
        config_sha256=hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
    )


def _component(name: str) -> ModelComponent:
    if name.startswith(("model.visual.", "visual.")):
        return ModelComponent.VISION
    if name.startswith(("mtp.", "model.mtp.")):
        return ModelComponent.MTP
    return ModelComponent.TEXT


def _split(name: str) -> tuple[str, str]:
    module, _, leaf = name.rpartition(".")
    return module, leaf


def _role(name: str, shape: Sequence[int], linear_modules: set[str]) -> TensorRole:
    module, leaf = _split(name)
    mod_leaf = module.rpartition(".")[2]
    if name.endswith(("embed_tokens.weight", "pos_embed.weight")):
        return TensorRole.EMBEDDING
    if mod_leaf == "lm_head":
        return TensorRole.LM_HEAD
    if "norm" in mod_leaf:
        return TensorRole.NORM
    if mod_leaf == "conv1d" or module.endswith("patch_embed.proj"):
        return TensorRole.CONV
    if leaf in ("A_log", "dt_bias"):
        return TensorRole.PARAMETER
    if module in linear_modules:
        return TensorRole.LINEAR_WEIGHT if leaf == "weight" else TensorRole.LINEAR_BIAS
    return TensorRole.OTHER


def _layer_index(name: str, component: ModelComponent) -> int | None:
    pattern = (
        r"(?:^|\.)blocks\.(\d+)\." if component is ModelComponent.VISION else r"layers\.(\d+)\."
    )
    m = re.search(pattern, name)
    return int(m.group(1)) if m else None


def inventory_from_tensors(
    config: dict[str, Any],
    rows: Iterable[TensorRow],
    *,
    include_lm_head_linear: bool = True,
) -> ModelInventory:
    facts = facts_from_config(config)
    rows = [(n, canonical_dtype(str(dt)), [int(x) for x in sh]) for n, dt, sh in rows]
    names = {n for n, _, _ in rows}
    linear_names = {
        _split(n)[0]
        for n, _, sh in rows
        if n.endswith(".weight")
        and len(sh) == 2
        and not n.endswith(("embed_tokens.weight", "pos_embed.weight"))
        and "norm" not in _split(n)[0].rpartition(".")[2]
        and _split(n)[0].rpartition(".")[2] != "lm_head"
    }
    tensors: list[TensorInfo] = []
    linears: list[LinearModule] = []
    for name, dtype, shape in sorted(rows):
        numel = 1
        for s in shape:
            numel *= s
        component = _component(name)
        role = _role(name, shape, linear_names)
        module = _split(name)[0]
        index = _layer_index(name, component)
        tensors.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                numel=numel,
                nbytes=tensor_bytes(numel, dtype),
                component=component,
                role=role,
                module=module,
                layer_index=index,
            )
        )
        is_head = role is TensorRole.LM_HEAD and include_lm_head_linear
        if (role is TensorRole.LINEAR_WEIGHT or is_head) and name.endswith(".weight"):
            kind = module.rpartition(".")[2]
            if component is ModelComponent.VISION:
                layer_type: str | None = "vision"
            elif index is not None and kind in _ATTN_KINDS and facts.layer_types:
                layer_type = facts.layer_types[index]
            elif index is not None:
                layer_type = "mlp"
            else:
                layer_type = None
            linears.append(
                LinearModule(
                    name=module,
                    kind=kind,
                    in_features=shape[1],
                    out_features=shape[0],
                    has_bias=f"{module}.bias" in names,
                    component=component,
                    layer_index=index,
                    layer_type=layer_type,
                    dtype=dtype,
                )
            )
    tied: list[list[str]] = []
    if facts.tie_word_embeddings:
        group = [t.name for t in tensors if t.role in (TensorRole.EMBEDDING, TensorRole.LM_HEAD)]
        group = [n for n in group if not n.endswith("pos_embed.weight")]
        if group:
            tied.append(group)
    skip = {n for g in tied for n in g[1:]}
    by_comp: dict[ModelComponent, list[int]] = {}
    for t in tensors:
        if t.name in skip:
            continue
        acc = by_comp.setdefault(t.component, [0, 0])
        acc[0] += t.numel
        acc[1] += t.nbytes
    digest = hashlib.sha256(
        _canonical_json([[t.name, t.dtype, t.shape] for t in tensors]).encode()
    ).hexdigest()
    return ModelInventory(
        facts=facts,
        tensors=tensors,
        linear_modules=linears,
        params_total=sum(v[0] for v in by_comp.values()),
        bytes_serialized_total=sum(v[1] for v in by_comp.values()),
        by_component=[
            ComponentParams(component=c, params=v[0], bytes_serialized=v[1])
            for c, v in sorted(by_comp.items(), key=lambda kv: kv[0].value)
        ],
        tied_groups=tied,
        inventory_hash=f"inv_{digest}",
    )


def load_headers(path: Path = MIMO_HEADERS) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mimo_inventory(*, include_lm_head_linear: bool = True) -> ModelInventory:
    doc = load_headers()
    rows = [(n, dt, sh) for n, dt, sh, _shard in doc["tensors"]]
    return inventory_from_tensors(
        doc["config"], rows, include_lm_head_linear=include_lm_head_linear
    )


# ---------------------------------------------------------------- synthetic tensor lists


def qwen35_text_rows(
    tc: dict[str, Any], *, prefix: str = "model.language_model."
) -> list[TensorRow]:
    """Tensor rows of a Qwen3.5 text decoder (transformers 5.18 module layout)."""
    h, i, v = tc["hidden_size"], tc["intermediate_size"], tc["vocab_size"]
    nq, nkv, d = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    hk, hv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv, k = tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["linear_conv_kernel_dim"]
    conv = 2 * hk * dk + hv * dv
    dt = str(tc.get("dtype", "bfloat16"))
    rows: list[TensorRow] = [
        (f"{prefix}embed_tokens.weight", dt, [v, h]),
        (f"{prefix}norm.weight", dt, [h]),
    ]
    for li, lt in enumerate(tc["layer_types"]):
        p = f"{prefix}layers.{li}."
        rows += [
            (p + "input_layernorm.weight", dt, [h]),
            (p + "post_attention_layernorm.weight", dt, [h]),
        ]
        rows += [(p + "mlp.gate_proj.weight", dt, [i, h]), (p + "mlp.up_proj.weight", dt, [i, h])]
        rows += [(p + "mlp.down_proj.weight", dt, [h, i])]
        if lt == "full_attention":
            a = p + "self_attn."
            rows += [
                (a + "q_proj.weight", dt, [2 * nq * d, h]),
                (a + "k_proj.weight", dt, [nkv * d, h]),
            ]
            rows += [
                (a + "v_proj.weight", dt, [nkv * d, h]),
                (a + "o_proj.weight", dt, [h, nq * d]),
            ]
            rows += [(a + "q_norm.weight", dt, [d]), (a + "k_norm.weight", dt, [d])]
        else:
            a = p + "linear_attn."
            rows += [
                (a + "in_proj_qkv.weight", dt, [conv, h]),
                (a + "in_proj_z.weight", dt, [hv * dv, h]),
            ]
            rows += [(a + "in_proj_b.weight", dt, [hv, h]), (a + "in_proj_a.weight", dt, [hv, h])]
            rows += [
                (a + "out_proj.weight", dt, [h, hv * dv]),
                (a + "conv1d.weight", dt, [conv, 1, k]),
            ]
            rows += [
                (a + "A_log", dt, [hv]),
                (a + "dt_bias", dt, [hv]),
                (a + "norm.weight", dt, [dv]),
            ]
    if not tc.get("tie_word_embeddings", False):
        rows.append(("lm_head.weight", dt, [v, h]))
    return rows


def dense_rows(
    cfg: dict[str, Any], *, qk_norm: bool = False, prefix: str = "model."
) -> list[TensorRow]:
    """Tensor rows of a Llama/Qwen3-style dense decoder."""
    h, i, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    d = cfg.get("head_dim") or h // nq
    dt = str(cfg.get("dtype", "bfloat16"))
    bias = bool(cfg.get("attention_bias", False))
    rows: list[TensorRow] = [
        (f"{prefix}embed_tokens.weight", dt, [v, h]),
        (f"{prefix}norm.weight", dt, [h]),
    ]
    for li in range(cfg["num_hidden_layers"]):
        p = f"{prefix}layers.{li}."
        rows += [
            (p + "input_layernorm.weight", dt, [h]),
            (p + "post_attention_layernorm.weight", dt, [h]),
        ]
        a = p + "self_attn."
        rows += [(a + "q_proj.weight", dt, [nq * d, h]), (a + "k_proj.weight", dt, [nkv * d, h])]
        rows += [(a + "v_proj.weight", dt, [nkv * d, h]), (a + "o_proj.weight", dt, [h, nq * d])]
        if bias:
            rows += [(a + "q_proj.bias", dt, [nq * d]), (a + "k_proj.bias", dt, [nkv * d])]
            rows += [(a + "v_proj.bias", dt, [nkv * d])]
        if qk_norm:
            rows += [(a + "q_norm.weight", dt, [d]), (a + "k_norm.weight", dt, [d])]
        rows += [(p + "mlp.gate_proj.weight", dt, [i, h]), (p + "mlp.up_proj.weight", dt, [i, h])]
        rows += [(p + "mlp.down_proj.weight", dt, [h, i])]
    if not cfg.get("tie_word_embeddings", False):
        rows.append(("lm_head.weight", dt, [v, h]))
    return rows


# Tiny configs of docs/research/architecture-memory.md §4.1 / §F (golden values).
TINY_Q35_TEXT: dict[str, Any] = {
    "model_type": "qwen3_5_text",
    "architectures": ["Qwen3_5ForCausalLM"],
    "vocab_size": 1000,
    "hidden_size": 96,
    "intermediate_size": 176,
    "num_hidden_layers": 2,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 40,
    "linear_num_key_heads": 3,
    "linear_num_value_heads": 9,
    "linear_key_head_dim": 24,
    "linear_value_head_dim": 28,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention", "full_attention"],
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 2, 1],
        "mrope_interleaved": True,
    },
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "dtype": "bfloat16",
}

TINY_DENSE: dict[str, Any] = {
    "vocab_size": 1000,
    "hidden_size": 96,
    "intermediate_size": 176,
    "num_hidden_layers": 2,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 40,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "dtype": "bfloat16",
}


def tiny_q35_inventory(**overrides: Any) -> ModelInventory:
    tc = {**TINY_Q35_TEXT, **overrides}
    return inventory_from_tensors(tc, qwen35_text_rows(tc, prefix="model."))


def tiny_dense_inventory(
    model_type: str = "llama", *, qk_norm: bool = False, **overrides: Any
) -> ModelInventory:
    arch = {"llama": "LlamaForCausalLM", "qwen3": "Qwen3ForCausalLM"}.get(model_type, "X")
    cfg = {**TINY_DENSE, "model_type": model_type, "architectures": [arch], **overrides}
    return inventory_from_tensors(cfg, dense_rows(cfg, qk_norm=qk_norm))
