"""MiMo-V2.6-Distill-Qwen-9B inventory from the committed fixture (plan.md §6.3–6.4, §21).

Expected numbers: docs/research/architecture-memory.md §9 and
docs/research/loading-quantization-peft.md §2.4 / V1 (header-only inspection of the same revision).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vramforge_estimator.inspection import inspect_model
from vramforge_estimator.schemas import ModelComponent, ModelInventory, TensorRole
from vramforge_estimator.sources import ResolvedSource, SourceAccess

PARAMS_TOTAL = 9_409_813_744
PARAMS_VISION = 456_010_480
PARAMS_TEXT = 8_953_803_264


def _check_mimo(inv: ModelInventory) -> None:
    facts = inv.facts
    assert facts.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert (facts.model_type, facts.text_model_type) == ("qwen3_5", "qwen3_5_text")
    assert facts.num_hidden_layers == 32
    assert (facts.hidden_size, facts.intermediate_size, facts.vocab_size) == (4096, 12288, 248320)
    assert (facts.num_attention_heads, facts.num_key_value_heads) == (16, 4)
    assert (facts.head_dim, facts.head_dim_source) == (256, "explicit")
    assert facts.layer_types[:4] == ["linear_attention"] * 3 + ["full_attention"]
    assert {c.layer_type: c.count for c in facts.layer_type_counts} == {
        "full_attention": 8,
        "linear_attention": 24,
    }
    assert facts.linear_attention == {
        "num_key_heads": 16,
        "num_value_heads": 32,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel_dim": 4,
    }
    assert facts.sliding_window is None
    assert facts.max_position_embeddings == 262144
    assert facts.tie_word_embeddings is False
    assert facts.extra["tie_word_embeddings_top"] is False
    assert facts.extra["tie_word_embeddings_text"] is False
    assert facts.config_dtype == "bfloat16"
    assert facts.extra["config_dtype_source"] == "text_config"
    assert facts.has_vision is True
    assert facts.has_mtp is False  # config declares mtp_num_hidden_layers=1, checkpoint has none
    assert facts.extra["mtp_num_hidden_layers"] == 1
    assert facts.extra["attn_output_gate"] is True
    assert facts.extra["partial_rotary_factor"] == 0.25
    assert facts.extra["rms_norm_eps"] == 1e-06
    assert facts.extra["vision_config"]["depth"] == 27
    assert facts.extra["checkpoint_dtype_params"] == {"bfloat16": PARAMS_TOTAL}
    assert facts.extra["field_sources"]["layer_types"] == "explicit"
    assert facts.extra["transformers_native_model_type"] is True
    assert facts.extra["architecture_classes_native"] == {"Qwen3_5ForConditionalGeneration": True}
    assert facts.rope["rope_theta"] == 10000000
    assert facts.rope["mrope_section"] == [11, 11, 10]
    assert facts.config_sha256 == "407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633"

    assert len(inv.tensors) == 760
    assert {t.dtype for t in inv.tensors} == {"bfloat16"}
    assert inv.params_total == PARAMS_TOTAL
    assert inv.bytes_serialized_total == 2 * PARAMS_TOTAL
    assert inv.index_total_size == 18_819_627_488
    assert {(c.component, c.params, c.bytes_serialized) for c in inv.by_component} == {
        (ModelComponent.TEXT, PARAMS_TEXT, 2 * PARAMS_TEXT),
        (ModelComponent.VISION, PARAMS_VISION, 2 * PARAMS_VISION),
    }
    assert inv.tied_groups == []
    assert inv.quantized_checkpoint_format is None
    assert inv.inventory_hash.startswith("inv_")

    roles = Counter(t.role for t in inv.tensors)
    assert roles == {
        TensorRole.LINEAR_WEIGHT: 358,
        TensorRole.LINEAR_BIAS: 110,
        TensorRole.NORM: 215,
        TensorRole.CONV: 26,
        TensorRole.PARAMETER: 49,  # 24 A_log + 24 dt_bias + vision pos_embed
        TensorRole.EMBEDDING: 1,
        TensorRole.LM_HEAD: 1,
    }
    by_name = {t.name: t for t in inv.tensors}
    q = by_name["model.language_model.layers.3.self_attn.q_proj.weight"]
    assert (q.module, q.layer_index, q.component, q.shard) == (
        "model.language_model.layers.3.self_attn.q_proj",
        3,
        ModelComponent.TEXT,
        "model-00003-of-00004.safetensors",  # as listed in the index weight_map
    )
    assert by_name["model.visual.pos_embed.weight"].role is TensorRole.PARAMETER
    assert by_name["model.visual.merger.linear_fc1.weight"].component is ModelComponent.VISION
    assert by_name["model.language_model.layers.0.linear_attn.A_log"].role is TensorRole.PARAMETER
    assert (
        by_name["model.language_model.layers.0.linear_attn.conv1d.weight"].role is TensorRole.CONV
    )
    assert by_name["model.language_model.layers.0.linear_attn.norm.weight"].role is TensorRole.NORM
    assert by_name["lm_head.weight"].nbytes == 2 * 248320 * 4096

    linear = inv.linear_modules
    by_component = Counter(m.component for m in linear)
    assert by_component == {ModelComponent.TEXT: 248, ModelComponent.VISION: 110}
    assert Counter(m.layer_type for m in linear if m.component is ModelComponent.TEXT) == {
        "mlp": 96,
        "linear_attention": 120,
        "full_attention": 32,
    }
    assert {m.layer_type for m in linear if m.component is ModelComponent.VISION} == {"vision"}
    names = {m.name: m for m in linear}
    assert "lm_head" not in names and "model.language_model.embed_tokens" not in names
    q_proj = names["model.language_model.layers.3.self_attn.q_proj"]
    assert (q_proj.kind, q_proj.in_features, q_proj.out_features, q_proj.has_bias) == (
        "q_proj",
        4096,
        8192,  # query and output gate (docs/research/architecture-memory.md §1.2)
        False,
    )
    a_proj = names["model.language_model.layers.0.linear_attn.in_proj_a"]
    assert (a_proj.out_features, a_proj.layer_type, a_proj.dtype) == (
        32,
        "linear_attention",
        "bfloat16",
    )
    qkv = names["model.visual.blocks.0.attn.qkv"]
    assert (qkv.in_features, qkv.out_features, qkv.has_bias, qkv.layer_index) == (
        1152,
        3456,
        True,
        0,
    )
    gate = names["model.language_model.layers.31.mlp.gate_proj"]
    assert (gate.in_features, gate.out_features, gate.layer_type) == (4096, 12288, "mlp")
    total_linear = sum(m.in_features * m.out_features for m in linear)
    assert total_linear == 7_369_682_944  # quantizable elements (loading-quantization-peft §2.4)


def test_mimo_inventory_from_local_directory(
    mimo_local: tuple[ResolvedSource, SourceAccess], mimo: SimpleNamespace
) -> None:
    source, access = mimo_local
    _check_mimo(inspect_model(source, access))


def test_rebuilt_shards_match_published_frames(
    tmp_path: Path, st_writer: Any, mimo: SimpleNamespace
) -> None:
    # PROVENANCE.md frame digests: the fixture reproduces the published header bytes exactly
    expected = {
        "model-00001-of-00004.safetensors": "b9a4eb7e269fed6f83f046413693444f83581583ef7d10fcdd2123174295340b",
        "model-00002-of-00004.safetensors": "4446e06139df99a4b1456b52cafc72ccac28035d82fe2d6baef695d580a6d6b1",
        "model-00003-of-00004.safetensors": "0bdfb937253ee825744db5e76e3925bd013f0d6af2c3bbfda69d16d4f69afad1",
        "model-00004-of-00004.safetensors": "7d2c7f4b2eaa619eb6b539a018adb78c3daa826a34e68548a2738625c5a0041e",
    }
    target = mimo.write_dir(tmp_path / "m", st_writer)
    for shard, digest in expected.items():
        assert mimo.frame_sha256(target / shard) == digest
        assert (target / shard).stat().st_size == mimo.shards[shard][0]


def test_mimo_inventory_from_fake_hub_matches_local(
    mimo_hub: tuple[ResolvedSource, SourceAccess, Any],
    mimo_local: tuple[ResolvedSource, SourceAccess],
) -> None:
    source, access, client = mimo_hub
    remote = inspect_model(source, access)
    _check_mimo(remote)
    local = inspect_model(*mimo_local)
    assert remote.inventory_hash == local.inventory_hash
    assert remote == local
    # only metadata files are downloaded; weights are read through header requests only
    downloads = {name for call, name in client.calls if call == "download_file"}
    assert downloads == {"config.json", "model.safetensors.index.json"}
    headers = sorted(name for call, name in client.calls if call == "read_safetensors_header")
    assert headers == sorted(f"model-0000{i}-of-00004.safetensors" for i in range(1, 5))


def test_inventory_hash_is_deterministic(mimo_local: tuple[ResolvedSource, SourceAccess]) -> None:
    first = inspect_model(*mimo_local)
    second = inspect_model(*mimo_local)
    assert first.inventory_hash == second.inventory_hash
    assert [t.name for t in first.tensors] == sorted(t.name for t in first.tensors)


def test_inventory_hash_changes_with_the_checkpoint(
    tmp_path: Path,
    st_writer: Any,
    mimo: SimpleNamespace,
    mimo_local: tuple[ResolvedSource, SourceAccess],
) -> None:
    import json

    from vramforge_estimator.schemas import ModelSourceRef
    from vramforge_estimator.sources import resolve_model

    base = inspect_model(*mimo_local)
    root = tmp_path / "other"
    target = mimo.write_dir(root / "mimo", st_writer)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["rms_norm_eps"] = 1e-05
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    access = SourceAccess(local_roots={"m": root})
    changed = inspect_model(resolve_model(ModelSourceRef(reference="local:m/mimo"), access), access)
    assert changed.inventory_hash != base.inventory_hash
    assert changed.params_total == base.params_total


@pytest.mark.parametrize("cap", [1024])
def test_metadata_cap_is_enforced(
    mimo_hub: tuple[ResolvedSource, SourceAccess, Any], cap: int
) -> None:
    from dataclasses import replace

    from vramforge_estimator.errors import EstimatorError
    from vramforge_estimator.schemas import ErrorCode

    source, access, _ = mimo_hub
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, replace(access, max_metadata_bytes=cap))
    assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE
    assert exc.value.issue.details["reason"] == "file_too_large"
