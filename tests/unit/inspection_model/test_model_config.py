"""config.json → ArchitectureFacts, and the tiny dense-decoder inventory (plan.md §6.3)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import inspect_model
from vramforge_estimator.inspection.model_config import (
    CheckpointEvidence,
    build_facts,
    load_config,
    quantization_method,
    requires_remote_code,
)
from vramforge_estimator.schemas import ErrorCode, ModelComponent, ModelSourceRef, TensorRole
from vramforge_estimator.sources import SourceAccess, resolve_model

TINY = Path(__file__).parents[2] / "fixtures" / "models" / "tiny-dense-decoder"


def _evidence(**overrides: Any) -> CheckpointEvidence:
    values: dict[str, Any] = {
        "has_vision_tensors": False,
        "has_mtp_tensors": False,
        "has_lm_head_tensor": True,
        "has_embedding_tensor": True,
        "dtype_params": {"bfloat16": 10},
    }
    values.update(overrides)
    return CheckpointEvidence(**values)


def _facts(config: dict[str, Any], **evidence: Any):  # type: ignore[no-untyped-def]
    return build_facts(load_config(json.dumps(config).encode()), _evidence(**evidence))


LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 256,
}


def test_dense_defaults_come_from_the_transformers_config() -> None:
    facts = _facts(LLAMA)
    assert (facts.head_dim, facts.head_dim_source) == (16, "inferred")
    assert facts.num_key_value_heads == 4  # LlamaConfig: defaults to num_attention_heads
    assert facts.extra["field_sources"]["num_key_value_heads"] == "transformers_default"
    assert facts.layer_types == ["full_attention", "full_attention"]
    assert facts.extra["field_sources"]["layer_types"] == "inferred"
    assert facts.tie_word_embeddings is False
    assert facts.extra["field_sources"]["tie_word_embeddings"] == "transformers_default"
    assert facts.config_dtype is None
    assert facts.has_vision is False and facts.has_mtp is False


def test_explicit_values_win() -> None:
    facts = _facts({**LLAMA, "head_dim": 32, "num_key_value_heads": 2, "tie_word_embeddings": True})
    assert (facts.head_dim, facts.head_dim_source) == (32, "explicit")
    assert facts.num_key_value_heads == 2
    assert facts.tie_word_embeddings is True
    assert facts.extra["field_sources"]["tie_word_embeddings"] == "explicit"


def test_qwen3_layer_types_from_transformers_config() -> None:
    facts = _facts(
        {
            **LLAMA,
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "sliding_window": None,
            "use_sliding_window": False,
        }
    )
    assert facts.layer_types == ["full_attention", "full_attention"]
    assert facts.extra["field_sources"]["layer_types"] == "transformers_default"


def test_unknown_model_type_uses_raw_values_only() -> None:
    config = {**LLAMA, "model_type": "vramforge_unknown_arch", "sliding_window": 128}
    facts = _facts(config, has_lm_head_tensor=False)
    assert facts.extra["transformers_native_model_type"] is False
    assert facts.extra["architecture_classes_native"] == {"LlamaForCausalLM": True}
    assert facts.layer_types == []  # sliding window evidence: not inferred
    assert facts.extra["field_sources"]["layer_types"] == "unknown"
    assert facts.sliding_window == 128
    assert facts.num_key_value_heads is None
    # no tie flag anywhere: an embedding without a serialized head must be tied
    assert facts.tie_word_embeddings is True
    assert facts.extra["field_sources"]["tie_word_embeddings"] == "checkpoint_evidence"


def test_nested_text_config_and_tie_flags_at_both_levels() -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 256,
            "tie_word_embeddings": True,
            "dtype": "bfloat16",
            "full_attention_interval": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
        },
        "vision_config": {"depth": 2, "hidden_size": 32, "model_type": "qwen3_5_vision"},
    }
    facts = _facts(config)
    assert facts.text_model_type == "qwen3_5_text"
    assert facts.tie_word_embeddings is False  # the top-level config drives the default loader
    assert facts.extra["tie_word_embeddings_top"] is False
    assert facts.extra["tie_word_embeddings_text"] is True
    assert facts.extra["tie_word_embeddings_text_effective"] is True
    # layer_types derived by Qwen3_5TextConfig from full_attention_interval
    assert facts.layer_types == ["linear_attention", "full_attention"] * 2
    assert facts.extra["field_sources"]["layer_types"] == "transformers_default"
    assert facts.linear_attention["num_value_heads"] == 4
    assert facts.config_dtype == "bfloat16"
    assert facts.extra["config_dtype_source"] == "text_config"
    assert facts.has_vision is True
    assert facts.extra["vision_config"] == {
        "depth": 2,
        "hidden_size": 32,
        "model_type": "qwen3_5_vision",
    }


@pytest.mark.parametrize(
    ("dtype_fields", "expected", "source"),
    [
        ({"dtype": "bfloat16"}, "bfloat16", "top_level"),
        ({"torch_dtype": "float16"}, "float16", "top_level"),
        ({"torch_dtype": "auto"}, None, "unknown"),
        ({}, None, "unknown"),
    ],
)
def test_config_dtype(dtype_fields: dict[str, str], expected: str | None, source: str) -> None:
    facts = _facts({**LLAMA, **dtype_fields})
    assert facts.config_dtype == expected
    assert facts.extra["config_dtype_source"] == source


def test_missing_required_fields() -> None:
    for field in ("hidden_size", "num_hidden_layers", "vocab_size", "model_type"):
        config = {k: v for k, v in LLAMA.items() if k != field}
        with pytest.raises(EstimatorError) as exc:
            _facts(config)
        assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE
        assert exc.value.issue.details == {"reason": "missing_config_field", "field": field}


def test_layer_types_must_match_layer_count() -> None:
    with pytest.raises(EstimatorError) as exc:
        _facts({**LLAMA, "layer_types": ["full_attention"]})
    assert exc.value.issue.details["reason"] == "layer_types_length_mismatch"


def test_invalid_config_json() -> None:
    for raw in (b"[1]", b"{", b"\xff"):
        with pytest.raises(EstimatorError) as exc:
            load_config(raw)
        assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE


def test_remote_code_and_quantization_detection() -> None:
    def cfg(**fields: Any):  # type: ignore[no-untyped-def]
        return load_config(json.dumps({**LLAMA, **fields}).encode())

    assert requires_remote_code(cfg(model_type="my_custom_arch", auto_map={"AutoModel": "m.M"}))
    assert requires_remote_code(cfg(model_type=None, auto_map={"AutoModel": "m.M"}))
    assert not requires_remote_code(cfg(auto_map={"AutoModel": "m.M"}))  # native llama
    assert not requires_remote_code(cfg(model_type="my_custom_arch"))
    assert (
        quantization_method(cfg(quantization_config={"quant_method": "gptq", "bits": 4})) == "gptq"
    )
    assert quantization_method(cfg(quantization_config={"bits": 4})) == "unknown"
    assert quantization_method(cfg()) is None


def test_tiny_dense_decoder_inventory(tmp_path: Path, st_writer: Any) -> None:
    root = tmp_path / "models"
    target = root / "tiny"
    target.mkdir(parents=True)
    (target / "config.json").write_bytes((TINY / "config.json").read_bytes())
    header = json.loads((TINY / "model.safetensors.header.json").read_text(encoding="utf-8"))
    st_writer.write_header(target / "model.safetensors", header)
    access = SourceAccess(local_roots={"models": root})
    inv = inspect_model(
        resolve_model(ModelSourceRef(reference="local:models/tiny"), access), access
    )

    facts = inv.facts
    assert (facts.model_type, facts.num_hidden_layers, facts.hidden_size) == ("llama", 2, 64)
    assert (facts.head_dim, facts.head_dim_source) == (16, "inferred")
    assert facts.num_key_value_heads == 2
    assert facts.layer_types == ["full_attention", "full_attention"]
    assert facts.tie_word_embeddings is True
    assert facts.config_dtype == "float16"
    assert facts.max_position_embeddings == 512
    assert facts.rope == {"rope_theta": 10000.0}
    assert facts.extra["rms_norm_eps"] == 1e-05

    assert len(inv.tensors) == 20
    assert inv.params_total == 90_432
    assert inv.bytes_serialized_total == 180_864
    assert inv.index_total_size is None
    assert inv.tied_groups == [["model.embed_tokens.weight", "lm_head.weight"]]
    assert [(c.component, c.params) for c in inv.by_component] == [(ModelComponent.TEXT, 90_432)]
    assert Counter(m.layer_type for m in inv.linear_modules) == {"full_attention": 8, "mlp": 6}
    k_proj = next(m for m in inv.linear_modules if m.name == "model.layers.1.self_attn.k_proj")
    assert (k_proj.in_features, k_proj.out_features, k_proj.layer_index) == (64, 32, 1)
    assert Counter(t.role for t in inv.tensors) == {
        TensorRole.EMBEDDING: 1,
        TensorRole.LINEAR_WEIGHT: 14,
        TensorRole.NORM: 5,
    }


def test_tied_head_serialized_twice_is_counted_once(tmp_path: Path, st_writer: Any) -> None:
    root = tmp_path / "models"
    target = root / "tied"
    target.mkdir(parents=True)
    (target / "config.json").write_bytes((TINY / "config.json").read_bytes())
    st_writer.write_safetensors(
        target / "model.safetensors",
        {
            "model.embed_tokens.weight": ("F16", [256, 64]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [64, 64]),
            "lm_head.weight": ("F16", [256, 64]),
        },
    )
    access = SourceAccess(local_roots={"models": root})
    inv = inspect_model(
        resolve_model(ModelSourceRef(reference="local:models/tied"), access), access
    )
    assert inv.tied_groups == [["model.embed_tokens.weight", "lm_head.weight"]]
    assert inv.params_total == 256 * 64 + 64 * 64  # lm_head shares the embedding storage
    assert inv.bytes_serialized_total == 2 * (2 * 256 * 64 + 64 * 64)
    assert inv.facts.extra["lm_head_serialized"] is True
