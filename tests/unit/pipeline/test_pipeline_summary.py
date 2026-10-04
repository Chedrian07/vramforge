"""The inventory summary embedded in results keeps counts and identity of the full inventory."""

from vramforge_estimator.pipeline import inventory_summary
from vramforge_estimator.schemas import (
    ArchitectureFacts,
    ComponentParams,
    LinearModule,
    ModelComponent,
    ModelInventory,
    TensorInfo,
    TensorRole,
)


def test_inventory_summary_counts() -> None:
    facts = ArchitectureFacts(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=1,
        hidden_size=8,
        vocab_size=16,
        config_sha256="c" * 64,
    )
    tensor = TensorInfo(
        name="model.layers.0.mlp.up_proj.weight",
        dtype="bfloat16",
        shape=[8, 8],
        numel=64,
        nbytes=128,
        component=ModelComponent.TEXT,
        role=TensorRole.LINEAR_WEIGHT,
        module="model.layers.0.mlp.up_proj",
        layer_index=0,
    )
    linear = LinearModule(
        name="model.layers.0.mlp.up_proj",
        kind="up_proj",
        in_features=8,
        out_features=8,
        has_bias=False,
        component=ModelComponent.TEXT,
        layer_index=0,
        dtype="bfloat16",
    )
    inv = ModelInventory(
        facts=facts,
        tensors=[tensor],
        linear_modules=[linear],
        params_total=64,
        bytes_serialized_total=128,
        by_component=[
            ComponentParams(component=ModelComponent.TEXT, params=64, bytes_serialized=128)
        ],
        inventory_hash="h",
    )
    summary = inventory_summary(inv)
    assert summary.tensor_count == 1
    assert summary.linear_module_count == 1
    assert summary.params_total == 64
    assert summary.inventory_hash == "h"
    assert summary.facts == facts
