"""Inventory structure parsing/validation and the library name-matching rules."""

from __future__ import annotations

from types import ModuleType

import pytest

from vramforge_estimator.architectures import matching
from vramforge_estimator.architectures.structure import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    SLIDING_ATTENTION,
    ModelStructure,
    is_moe,
    text_layer_types,
)
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, ModelInventory


def test_mimo_structure_layer_types_and_dims(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    assert st.count(LINEAR_ATTENTION) == 24 and st.count(FULL_ATTENTION) == 8
    assert st.layer_types[:4] == [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]
    d = st.dims
    assert (d.hidden, d.intermediate, d.heads, d.kv_heads, d.head_dim) == (4096, 12288, 16, 4, 256)
    assert d.rotary_dim == 64  # partial_rotary_factor 0.25
    assert (d.lin_key_heads, d.lin_value_heads, d.lin_key_dim, d.lin_value_dim) == (
        16,
        32,
        128,
        128,
    )
    assert d.conv_kernel == 4 and d.conv_dim == 8192 and d.lin_value_width == 4096
    assert st.output_embedding == "lm_head"
    assert st.final_norm is not None and st.final_norm.module == "model.language_model.norm"
    assert not st.tied_skip
    full = st.layers[3]
    assert set(full.linears) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert full.linears["q_proj"].out_features == 2 * 16 * 256  # query + output gate
    assert set(st.layers[0].norms) == {"input_layernorm", "post_attention_layernorm", "norm"}


def test_loading_scope_keeps_or_drops_the_vision_tower(mimo: ModelInventory) -> None:
    st = ModelStructure(mimo, "qwen3_5")
    full = st.loaded_tensors("full_checkpoint")
    text = st.loaded_tensors("text_only")
    assert len(full) == 760 and len(text) == 427
    assert sum(t.numel for t in full) == 9_409_813_744
    assert sum(t.numel for t in text) == 8_953_803_264  # 333 `model.visual.*` tensors dropped


def test_mtp_tensors_are_never_loaded(ib: ModuleType) -> None:
    tc = dict(ib.TINY_Q35_TEXT, mtp_num_hidden_layers=1)
    rows = [*ib.qwen35_text_rows(tc, prefix="model."), ("mtp.fc.weight", "bfloat16", [96, 192])]
    st = ModelStructure(ib.inventory_from_tensors(tc, rows), "qwen3_5")
    assert all(not t.name.startswith("mtp.") for t in st.loaded_tensors("full_checkpoint"))
    assert "mtp.fc" not in {m.name for m in st.text_linears()}


def test_lm_head_is_found_without_a_linear_module_entry(ib: ModuleType) -> None:
    inv = ib.tiny_q35_inventory()
    no_head = inv.model_copy(
        update={"linear_modules": [m for m in inv.linear_modules if m.kind != "lm_head"]}
    )
    st = ModelStructure(no_head, "qwen3_5")
    assert st.output_embedding == "lm_head"
    assert st.linear_by_name["lm_head"].out_features == 1000


def test_tied_embeddings_are_counted_once(ib: ModuleType) -> None:
    cfg = dict(ib.TINY_DENSE, model_type="llama", architectures=["LlamaForCausalLM"])
    cfg["tie_word_embeddings"] = True
    rows = [*ib.dense_rows(cfg), ("lm_head.weight", "bfloat16", [1000, 96])]
    st = ModelStructure(ib.inventory_from_tensors(cfg, rows), "dense")
    assert st.tied_skip == {"lm_head.weight"}
    names = {t.name for t in st.loaded_tensors("full_checkpoint")}
    assert "model.embed_tokens.weight" in names and "lm_head.weight" not in names


def test_dense_family_rejects_linear_attention(mimo: ModelInventory) -> None:
    with pytest.raises(EstimatorError) as err:
        ModelStructure(mimo, "dense")
    assert err.value.issue.code is ErrorCode.UNSUPPORTED_ARCHITECTURE


def test_unexpected_linear_module_is_unsupported(ib: ModuleType) -> None:
    tc = dict(ib.TINY_Q35_TEXT)
    rows = ib.qwen35_text_rows(tc, prefix="model.")
    rows.append(("model.layers.0.mlp.shared_expert_gate.weight", "bfloat16", [1, 96]))
    with pytest.raises(EstimatorError) as err:
        ModelStructure(ib.inventory_from_tensors(tc, rows), "qwen3_5")
    assert err.value.issue.details["kinds"] == ["shared_expert_gate"]


def test_ungated_q_proj_is_not_qwen35(ib: ModuleType) -> None:
    tc = dict(ib.TINY_Q35_TEXT)
    rows = [
        (n, dt, [240, 96] if n.endswith("q_proj.weight") else sh)
        for n, dt, sh in ib.qwen35_text_rows(tc, prefix="model.")
    ]
    with pytest.raises(EstimatorError) as err:
        ModelStructure(ib.inventory_from_tensors(tc, rows), "qwen3_5")
    assert err.value.issue.details["kind"] == "q_proj"


def test_missing_pre_norm_is_unsupported(ib: ModuleType) -> None:
    cfg = dict(ib.TINY_DENSE, model_type="olmo2", architectures=["X"])
    rows = [r for r in ib.dense_rows(cfg) if "input_layernorm" not in r[0]]
    with pytest.raises(EstimatorError):
        ModelStructure(ib.inventory_from_tensors(cfg, rows), "dense")


def test_layernorm_decoder_with_llama_names_is_unsupported(ib: ModuleType) -> None:
    # StableLM: q/k/v/o + gate/up/down + input/post_attention_layernorm, but nn.LayerNorm (bias)
    cfg = dict(ib.TINY_DENSE, model_type="stablelm", architectures=["StableLmForCausalLM"])
    rows = ib.dense_rows(cfg)
    rows += [(n.removesuffix("weight") + "bias", dt, sh) for n, dt, sh in rows if "norm" in n]
    with pytest.raises(EstimatorError) as err:
        ModelStructure(ib.inventory_from_tensors(cfg, rows), "dense")
    assert err.value.issue.code is ErrorCode.UNSUPPORTED_ARCHITECTURE
    assert "input_layernorm" in err.value.issue.details["norms"]


def test_vision_layernorm_biases_do_not_block_the_text_decoder(mimo: ModelInventory) -> None:
    # the vision tower's LayerNorms have biases but never run on text-only data
    assert any(t.name.endswith("norm1.bias") for t in mimo.tensors)
    assert ModelStructure(mimo, "qwen3_5").count(LINEAR_ATTENTION) == 24


def test_layer_types_are_inferred_like_transformers(ib: ModuleType) -> None:
    dense = ib.tiny_dense_inventory("llama").facts
    assert text_layer_types(dense) == [FULL_ATTENTION, FULL_ATTENTION]
    sliding = ib.tiny_dense_inventory("mistral", sliding_window=64).facts
    assert text_layer_types(sliding) == [SLIDING_ATTENTION, SLIDING_ATTENTION]


def test_moe_markers(ib: ModuleType) -> None:
    facts = ib.tiny_dense_inventory("llama").facts
    assert not is_moe(facts)
    assert is_moe(facts.model_copy(update={"extra": {"num_experts": 64}}))
    assert not is_moe(facts.model_copy(update={"extra": {"num_experts": 0}}))


# ---------------------------------------------------------------- name matching rules


def test_peft_list_semantics_are_exact_or_dot_suffix() -> None:
    assert matching.peft_name_match("model.layers.0.self_attn.q_proj", ["q_proj"])
    assert matching.peft_name_match("lm_head", ["lm_head"])
    assert not matching.peft_name_match("model.layers.0.self_attn.xq_proj", ["q_proj"])


def test_peft_regex_is_a_full_match() -> None:
    name = "model.language_model.layers.3.self_attn.q_proj"
    assert matching.peft_regex_match(name, r".*language_model.*\.(q|k)_proj")
    assert not matching.peft_regex_match(name, r"language_model")


def test_modules_to_save_selection_has_no_dot_boundary() -> None:
    # `"norm"` also wraps input_layernorm and q_norm (peft utils/other.py:1087-1088)
    assert matching.modules_to_save_match("model.layers.0.input_layernorm", ["norm"])
    assert matching.modules_to_save_match("model.layers.0.self_attn.q_norm", ["norm"])
    assert not matching.modules_to_save_match("model.visual.blocks.0.norm1", ["norm"])
    # ...but adapter exclusion uses a dot boundary
    assert matching.inside_modules_to_save("model.layers.0.mlp.down_proj", ["mlp"])
    assert not matching.inside_modules_to_save("model.layers.0.mlp.down_proj", ["ml"])


def test_rank_pattern_first_match_wins() -> None:
    pattern = {"down_proj": 8, r"layers\.0\..*": 4}
    assert matching.rank_for("model.layers.0.mlp.down_proj", pattern, 16) == 8
    assert matching.rank_for("model.layers.0.mlp.up_proj", pattern, 16) == 4
    assert matching.rank_for("model.layers.1.mlp.up_proj", pattern, 16) == 16


def test_bnb_skip_semantics() -> None:
    assert matching.bnb_skip_match("lm_head", ["lm_head"])
    assert matching.bnb_skip_match("model.visual.blocks.0.attn.qkv", ["model.visual"])
    assert matching.bnb_skip_match("model.layers.3.mlp.fc1", ["fc1"])
    assert not matching.bnb_skip_match("model.layers.3.mlp.gate_proj", ["lm_head"])


@pytest.mark.parametrize("pattern", [r"(a+)+$", r"(.*)*x", r"(\w+\.)+q_proj", "x" * 600, "("])
def test_unsafe_or_invalid_patterns_are_rejected(pattern: str) -> None:
    with pytest.raises(EstimatorError) as err:
        matching.peft_regex_match("model.layers.0.self_attn.q_proj", pattern)
    assert err.value.issue.code is ErrorCode.INVALID_REQUEST
