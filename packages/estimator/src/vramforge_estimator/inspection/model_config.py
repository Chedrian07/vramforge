"""config.json → `ArchitectureFacts` (plan.md §6.2–6.4).

Values written in config.json are taken as is ("explicit"). Missing values come from the pinned
transformers config class for the same ``model_type`` (what the trainer will actually see) and are
marked as such in ``facts.extra["field_sources"]``; only when that is impossible a documented
inference is used, otherwise the field stays unknown. transformers is imported lazily (analysis
extra) and never runs repository code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ArchitectureFacts, ErrorCode, LayerTypeCount, Stage
from vramforge_estimator.units import canonical_dtype

TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config")

_ALIASES: dict[str, tuple[str, ...]] = {
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers", "n_layers"),
    "hidden_size": ("hidden_size", "n_embd", "d_model"),
    "vocab_size": ("vocab_size",),
    "intermediate_size": ("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size"),
    "num_attention_heads": ("num_attention_heads", "n_head", "num_heads", "n_heads"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
    "max_position_embeddings": ("max_position_embeddings", "n_positions", "max_sequence_length"),
}

# Gated DeltaNet / linear-attention dimensions (docs/research/architecture-memory.md §1.3).
LINEAR_ATTENTION_KEYS = {
    "linear_num_key_heads": "num_key_heads",
    "linear_num_value_heads": "num_value_heads",
    "linear_key_head_dim": "key_head_dim",
    "linear_value_head_dim": "value_head_dim",
    "linear_conv_kernel_dim": "conv_kernel_dim",
}

# Config fields that adapters may need; copied to facts.extra when present.
TEXT_EXTRA_KEYS = (
    "attn_output_gate",
    "partial_rotary_factor",
    "rms_norm_eps",
    "layer_norm_eps",
    "layer_norm_epsilon",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
    "mlp_bias",
    "full_attention_interval",
    "mlp_only_layers",
    "mamba_ssm_dtype",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "use_sliding_window",
    "max_window_layers",
    "sliding_window_pattern",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "decoder_sparse_step",
    "n_routed_experts",
    "n_shared_experts",
    "first_k_dense_replace",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "final_logit_softcapping",
    "attn_logit_softcapping",
    "query_pre_attn_scalar",
    "use_cache",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
)
TOP_EXTRA_KEYS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "transformers_version",
)
VISION_SUMMARY_KEYS = (
    "model_type",
    "depth",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "num_attention_heads",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "num_position_embeddings",
    "in_channels",
    "hidden_act",
    "deepstack_visual_indexes",
)


@dataclass(frozen=True)
class ParsedConfig:
    raw: dict[str, Any]
    text: dict[str, Any]  # nested text config, or `raw` itself for plain decoders
    nested_key: str | None
    sha256: str  # of the exact config.json bytes

    @property
    def model_type(self) -> str | None:
        value = self.raw.get("model_type")
        return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class CheckpointEvidence:
    """What the safetensors headers say, used where config.json is silent."""

    has_vision_tensors: bool
    has_mtp_tensors: bool
    has_lm_head_tensor: bool
    has_embedding_tensor: bool
    dtype_params: dict[str, int]


def _error(code: ErrorCode, message: str, **details: object) -> EstimatorError:
    return EstimatorError(
        make_issue(code, message, stage=Stage.INSPECTING, component="model", **details)
    )


def load_config(data: bytes) -> ParsedConfig:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raw = None
    if not isinstance(raw, dict):
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "config.json을 JSON 객체로 읽을 수 없습니다.",
            reason="invalid_config_json",
        )
    nested_key = next((k for k in TEXT_CONFIG_KEYS if isinstance(raw.get(k), dict)), None)
    text = raw[nested_key] if nested_key else raw
    return ParsedConfig(
        raw=raw, text=text, nested_key=nested_key, sha256=hashlib.sha256(data).hexdigest()
    )


def native_model_types() -> frozenset[str]:
    """``model_type`` values with a built-in config class in the pinned transformers."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return frozenset(CONFIG_MAPPING_NAMES)


def native_class_names(names: list[str]) -> dict[str, bool]:
    import transformers

    available = set(dir(transformers))  # lazy module: lists names without importing them
    return {name: name in available for name in names}


def requires_remote_code(config: ParsedConfig) -> bool:
    """``auto_map`` without a native transformers class for ``model_type`` (plan §18)."""
    if not config.raw.get("auto_map"):
        return False
    return config.model_type is None or config.model_type not in native_model_types()


def quantization_method(config: ParsedConfig) -> str | None:
    """Pre-quantized checkpoints declare ``quantization_config`` (plan §6.3)."""
    for scope in (config.raw, config.text):
        quant = scope.get("quantization_config")
        if isinstance(quant, dict) and quant:
            method = quant.get("quant_method")
            return str(method) if method else "unknown"
        if quant:
            return "unknown"
    return None


def _first_int(scope: dict[str, Any], field: str) -> int | None:
    for key in _ALIASES.get(field, (field,)):
        value = scope.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _transformers_defaults(config: ParsedConfig) -> tuple[dict[str, Any], str | None]:
    """Selected attributes of the transformers config object built from config.json."""
    model_type = config.model_type
    if model_type is None:
        return {}, None
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if model_type not in CONFIG_MAPPING:
            return {}, None
        cfg = CONFIG_MAPPING[model_type].from_dict(copy.deepcopy(config.raw))
        text = cfg.get_text_config(decoder=True)
    except Exception as exc:  # invalid config for the native class; recorded, not fatal
        return {}, type(exc).__name__
    out: dict[str, Any] = {}
    for name in ("num_key_value_heads", "head_dim"):
        value = getattr(text, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    layer_types = getattr(text, "layer_types", None)
    if isinstance(layer_types, list | tuple) and all(isinstance(t, str) for t in layer_types):
        out["layer_types"] = list(layer_types)
    for name, obj in (("tie_top", cfg), ("tie_text", text)):
        value = getattr(obj, "tie_word_embeddings", None)
        if isinstance(value, bool):
            out[name] = value
    return out, None


def _canonical_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return canonical_dtype(value)
    except ValueError:
        return None


def _sliding_or_linear_evidence(text: dict[str, Any]) -> bool:
    sliding = text.get("sliding_window") is not None and text.get("use_sliding_window") is not False
    return (
        sliding
        or any(key in text for key in LINEAR_ATTENTION_KEYS)
        or any(
            key in text
            for key in ("full_attention_interval", "attention_types", "sliding_window_pattern")
        )
    )


def build_facts(config: ParsedConfig, evidence: CheckpointEvidence) -> ArchitectureFacts:
    raw, text = config.raw, config.text
    model_type = config.model_type
    if model_type is None:
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "config.json에 model_type이 없어 모델 구조를 확인할 수 없습니다.",
            reason="missing_config_field",
            field="model_type",
        )
    required: dict[str, int] = {}
    for field in ("num_hidden_layers", "hidden_size", "vocab_size"):
        value = _first_int(text, field)
        if value is None:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                f"config.json에서 {field} 값을 찾을 수 없습니다.",
                reason="missing_config_field",
                field=field,
            )
        required[field] = value
    num_layers = required["num_hidden_layers"]

    defaults, defaults_error = _transformers_defaults(config)
    sources: dict[str, str] = {}

    heads = _first_int(text, "num_attention_heads")
    kv_heads = _first_int(text, "num_key_value_heads")
    sources["num_key_value_heads"] = "explicit"
    if kv_heads is None:
        kv_heads = defaults.get("num_key_value_heads")
        sources["num_key_value_heads"] = "transformers_default" if kv_heads else "unknown"

    head_dim = text.get("head_dim")
    head_dim_source = "explicit"
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        head_dim = defaults.get("head_dim")
        head_dim_source = "inferred"
        if head_dim is None and heads and required["hidden_size"] % heads == 0:
            head_dim = required["hidden_size"] // heads
        if head_dim is None:
            head_dim_source = "unknown"

    explicit_types = text.get("layer_types")
    if isinstance(explicit_types, list) and all(isinstance(t, str) for t in explicit_types):
        layer_types = list(explicit_types)
        sources["layer_types"] = "explicit"
    elif "layer_types" in defaults:
        layer_types = defaults["layer_types"]
        sources["layer_types"] = "transformers_default"
    elif not _sliding_or_linear_evidence(text):
        layer_types = ["full_attention"] * num_layers
        sources["layer_types"] = "inferred"
    else:
        layer_types = []
        sources["layer_types"] = "unknown"
    if layer_types and len(layer_types) != num_layers:
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "config.json의 layer_types 길이가 num_hidden_layers와 다릅니다.",
            reason="layer_types_length_mismatch",
            layer_types=len(layer_types),
            num_hidden_layers=num_layers,
        )
    counts: dict[str, int] = {}
    for layer_type in layer_types:
        counts[layer_type] = counts.get(layer_type, 0) + 1

    # Tie flag of the default loader (config.architectures[0]: the top-level config).
    tie_top_raw = raw.get("tie_word_embeddings")
    tie_text_raw = text.get("tie_word_embeddings")
    if isinstance(tie_top_raw, bool):
        tie, sources["tie_word_embeddings"] = tie_top_raw, "explicit"
    elif "tie_top" in defaults:
        tie, sources["tie_word_embeddings"] = defaults["tie_top"], "transformers_default"
    elif isinstance(tie_text_raw, bool):
        tie, sources["tie_word_embeddings"] = tie_text_raw, "explicit_text_config"
    else:
        # The checkpoint itself: an embedding without any serialized lm_head must be tied.
        tie = evidence.has_embedding_tensor and not evidence.has_lm_head_tensor
        sources["tie_word_embeddings"] = "checkpoint_evidence"

    dtype_top = raw.get("dtype", raw.get("torch_dtype"))
    dtype_text = text.get("dtype", text.get("torch_dtype")) if config.nested_key else None
    config_dtype = _canonical_or_none(dtype_top) or _canonical_or_none(dtype_text)
    dtype_source = (
        "top_level"
        if _canonical_or_none(dtype_top)
        else ("text_config" if config_dtype else "unknown")
    )

    rope: dict[str, Any] = {}
    if isinstance(text.get("rope_parameters"), dict):
        rope = copy.deepcopy(text["rope_parameters"])
    else:
        rope = {k: text[k] for k in ("rope_theta", "rope_scaling") if text.get(k) is not None}

    linear = {
        name: value
        for key, name in LINEAR_ATTENTION_KEYS.items()
        if isinstance(value := text.get(key), int) and not isinstance(value, bool)
    }
    sliding = text.get("sliding_window")

    extra: dict[str, Any] = {k: copy.deepcopy(text[k]) for k in TEXT_EXTRA_KEYS if k in text}
    extra.update({k: copy.deepcopy(raw[k]) for k in TOP_EXTRA_KEYS if k in raw})
    vision = raw.get("vision_config")
    if isinstance(vision, dict):
        extra["vision_config"] = {k: vision[k] for k in VISION_SUMMARY_KEYS if k in vision}
    extra["tie_word_embeddings_top"] = tie_top_raw if isinstance(tie_top_raw, bool) else None
    extra["tie_word_embeddings_text"] = tie_text_raw if isinstance(tie_text_raw, bool) else None
    if "tie_text" in defaults:
        extra["tie_word_embeddings_text_effective"] = defaults["tie_text"]
    extra["config_dtype_source"] = dtype_source
    extra["dtype_text_config"] = _canonical_or_none(dtype_text)
    extra["checkpoint_dtype_params"] = dict(sorted(evidence.dtype_params.items()))
    extra["field_sources"] = sources
    extra["transformers_native_model_type"] = model_type in native_model_types()
    architectures = [a for a in raw.get("architectures") or [] if isinstance(a, str)]
    extra["architecture_classes_native"] = native_class_names(architectures)
    extra["auto_map_present"] = bool(raw.get("auto_map"))
    if defaults_error:
        extra["transformers_config_error"] = defaults_error

    text_model_type = text.get("model_type") if config.nested_key else None
    return ArchitectureFacts(
        architectures=architectures,
        model_type=model_type,
        text_model_type=text_model_type if isinstance(text_model_type, str) else None,
        num_hidden_layers=num_layers,
        hidden_size=required["hidden_size"],
        intermediate_size=_first_int(text, "intermediate_size"),
        vocab_size=required["vocab_size"],
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        head_dim_source=head_dim_source,  # type: ignore[arg-type]
        layer_types=layer_types,
        layer_type_counts=[
            LayerTypeCount(layer_type=k, count=v) for k, v in sorted(counts.items())
        ],
        sliding_window=sliding
        if isinstance(sliding, int) and not isinstance(sliding, bool)
        else None,
        linear_attention=linear,
        max_position_embeddings=_first_int(text, "max_position_embeddings"),
        tie_word_embeddings=tie,
        config_dtype=config_dtype,
        has_vision=isinstance(vision, dict) or evidence.has_vision_tensors,
        has_mtp=evidence.has_mtp_tensors,
        rope=rope,
        extra=extra,
        config_sha256=config.sha256,
    )
