"""Structural adapter matching (plan §6.2, §11.4): never by model-name substrings."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType

import pytest
from arch_helpers import make_cfg

from vramforge_estimator.architectures import ArchitectureAdapter, get_adapter, match_adapter
from vramforge_estimator.architectures.registry import adapter_ids
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, ModelComponent, ModelInventory, Strategy

PROTOCOL_METHODS = sorted(
    name
    for name, member in vars(ArchitectureAdapter).items()
    if inspect.isfunction(member) and not name.startswith("_")
)


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


def test_layernorm_configs_do_not_match(mimo: ModelInventory, ib: ModuleType) -> None:
    dense = ib.tiny_dense_inventory("llama").facts
    layernorm = dense.model_copy(update={"extra": {**dense.extra, "layer_norm_eps": 1e-5}})
    assert match_adapter(layernorm) is None  # StableLM / GPT-NeoX style nn.LayerNorm
    both = {**mimo.facts.extra, "layer_norm_eps": 1e-5}
    assert "rms_norm_eps" in mimo.facts.extra
    assert match_adapter(mimo.facts.model_copy(update={"extra": both})) == "qwen3_5_hybrid"


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


def _signature(fn: Callable[..., object]) -> list[tuple[str, object, object, object]]:
    return [
        (p.name, p.kind, p.default, p.annotation) for p in inspect.signature(fn).parameters.values()
    ]


@pytest.mark.parametrize("adapter_id", ["qwen3_5_hybrid", "dense_decoder"])
def test_adapters_implement_every_protocol_method_with_its_signature(adapter_id: str) -> None:
    assert "loading_budget_bytes" in PROTOCOL_METHODS and len(PROTOCOL_METHODS) == 10
    cls = type(get_adapter(adapter_id))
    for name in PROTOCOL_METHODS:
        theirs, ours = getattr(ArchitectureAdapter, name), getattr(cls, name)
        assert _signature(ours) == _signature(theirs), name
        assert inspect.signature(ours).return_annotation == (
            inspect.signature(theirs).return_annotation
        ), name


def test_loading_budget_is_the_device_map_sum(mimo: ModelInventory) -> None:
    adapter = get_adapter("qwen3_5_hybrid")
    # 4-bit Linear weights at 0.5 B/param + everything else in the load dtype (LQ §4.4-4.5)
    assert adapter.loading_budget_bytes(mimo, make_cfg()) == 7_765_103_072
    assert adapter.loading_budget_bytes(mimo, make_cfg(load="float32")) == 11_845_364_672
    text = adapter.loading_budget_bytes(mimo, make_cfg(scope="text_only"))
    assert text < 7_765_103_072  # the vision tower is not loaded
    unquantized = adapter.loading_budget_bytes(mimo, make_cfg(strategy=Strategy.LORA))
    loaded = sum(t.numel for t in mimo.tensors if t.component is not ModelComponent.MTP)
    assert unquantized == 2 * loaded  # bf16 everywhere; MTP is never loaded, nothing is tied
