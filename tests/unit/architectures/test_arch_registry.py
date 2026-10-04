"""Structural adapter matching (plan §6.2, §11.4): never by model-name substrings."""

from __future__ import annotations

from types import ModuleType

import pytest
from arch_helpers import make_cfg

from vramforge_estimator.architectures import ArchitectureAdapter, get_adapter, match_adapter
from vramforge_estimator.architectures.registry import adapter_ids
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, ModelInventory


def test_example_model_matches_the_hybrid_adapter(mimo: ModelInventory) -> None:
    assert match_adapter(mimo.facts) == "qwen3_5_hybrid"


def test_dense_models_match_the_dense_adapter(ib: ModuleType) -> None:
    assert match_adapter(ib.tiny_dense_inventory("llama").facts) == "dense_decoder"
    assert match_adapter(ib.tiny_dense_inventory("qwen3", qk_norm=True).facts) == "dense_decoder"
    assert match_adapter(ib.tiny_dense_inventory("mistral", sliding_window=64).facts) == (
        "dense_decoder"
    )
    assert match_adapter(ib.tiny_q35_inventory().facts) == "qwen3_5_hybrid"


def test_matching_ignores_names(mimo: ModelInventory, ib: ModuleType) -> None:
    renamed = mimo.facts.model_copy(
        update={"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    )
    assert match_adapter(renamed) == "qwen3_5_hybrid"
    dense = ib.tiny_dense_inventory("llama").facts.model_copy(
        update={"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
    )
    assert match_adapter(dense) == "dense_decoder"


@pytest.mark.parametrize(
    "extra",
    [
        {"num_experts": 512},
        {"num_local_experts": 8},
        {"hidden_act": "gelu_pytorch_tanh"},
        {"is_encoder_decoder": True},
    ],
)
def test_unsupported_structures_do_not_match(
    mimo: ModelInventory, ib: ModuleType, extra: dict
) -> None:
    for facts in (mimo.facts, ib.tiny_dense_inventory("llama").facts):
        assert match_adapter(facts.model_copy(update={"extra": {**facts.extra, **extra}})) is None


def test_linear_attention_without_gated_deltanet_dims_does_not_match(mimo: ModelInventory) -> None:
    assert match_adapter(mimo.facts.model_copy(update={"linear_attention": {}})) is None


def test_unknown_layer_types_do_not_match(ib: ModuleType) -> None:
    facts = ib.tiny_dense_inventory("llama").facts
    chunked = facts.model_copy(update={"layer_types": ["chunked_attention"] * 2})
    assert match_adapter(chunked) is None


def test_get_adapter_and_protocol(mimo: ModelInventory) -> None:
    assert adapter_ids() == ["qwen3_5_hybrid", "dense_decoder"]
    adapter: ArchitectureAdapter = get_adapter("qwen3_5_hybrid")
    assert adapter.adapter_id == "qwen3_5_hybrid" and adapter.supports(mimo.facts)
    assert adapter.lm_head_dims(mimo) == (4096, 248_320)
    with pytest.raises(EstimatorError) as err:
        get_adapter("generic_multiplier")
    assert err.value.issue.code is ErrorCode.UNSUPPORTED_ARCHITECTURE


def test_wrong_adapter_refuses_the_inventory(mimo: ModelInventory) -> None:
    with pytest.raises(EstimatorError) as err:
        get_adapter("dense_decoder").resident_weights(mimo, make_cfg(), ["t"])
    assert err.value.issue.code is ErrorCode.UNSUPPORTED_ARCHITECTURE
    assert err.value.issue.user_message  # Korean, display-safe


def test_tied_dense_lm_head_dims_fall_back_to_the_embedding(ib: ModuleType) -> None:
    inv = ib.tiny_dense_inventory("llama", tie_word_embeddings=True)
    inv = inv.model_copy(
        update={
            "tensors": [t for t in inv.tensors if t.name != "lm_head.weight"],
            "linear_modules": [m for m in inv.linear_modules if m.kind != "lm_head"],
        }
    )
    assert get_adapter("dense_decoder").lm_head_dims(inv) == (96, 1000)
