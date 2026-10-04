"""LoRA target resolution and trainable parameter groups (plan §9.4, §19.1, §19.3).

MiMo expectations: docs/research/loading-quantization-peft.md §5.3 (verified with real PEFT on a
meta model, V4): auto/text-only 248 modules / 43,278,336 params, all-linear 358 / 51,265,024.
"""

from __future__ import annotations

from types import ModuleType

import pytest
from arch_helpers import make_cfg

from vramforge_estimator.architectures.structure import (
    ATTENTION_PROJ,
    GDN_PROJ,
    MLP_PROJ,
    ModelStructure,
)
from vramforge_estimator.architectures.trainable import (
    ArchTrainableGroup,
    build_trainability,
    peft_target_spec,
    resolve_lora_targets,
    trainable_group_list,
)
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, ModelInventory, Strategy

VERIFIED = frozenset(ATTENTION_PROJ + GDN_PROJ + MLP_PROJ)
TEXT_NAMES = [*ATTENTION_PROJ, *GDN_PROJ, *MLP_PROJ]


@pytest.fixture(scope="module")
def st(mimo: ModelInventory) -> ModelStructure:
    return ModelStructure(mimo, "qwen3_5")


def _targets(
    st: ModelStructure, target: str | list[str], exclude: list[str] | None = None
) -> list[str]:
    return [m.name for m in resolve_lora_targets(st, target, exclude or [], VERIFIED)]


def _numel(groups: list, *, grad: bool | None = None, kind: str | None = None) -> int:
    return sum(
        g.numel
        for g in groups
        if (grad is None or g.receives_grad is grad) and (kind is None or g.kind == kind)
    )


def test_plan_lora_row_4096x4096_r16_is_131072(st: ModelStructure) -> None:
    name = "model.language_model.layers.3.self_attn.o_proj"  # in = out = 4096
    groups = trainable_group_list(st, make_cfg(strategy=Strategy.LORA, targets=[name], r=16))
    assert _numel(groups) == 131_072
    assert {g.name for g in groups} == {"lora:A:text:16x4096", "lora:B:text:4096x16"}


def test_alpha_does_not_change_adapter_size(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    a16 = trainable_group_list(st, make_cfg(targets=names, alpha=16))
    a64 = trainable_group_list(st, make_cfg(targets=names, alpha=64))
    assert [(g.name, g.numel) for g in a16] == [(g.name, g.numel) for g in a64]


def test_mimo_auto_verified_targets(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    assert len(names) == 248
    assert not any("visual" in n or n == "lm_head" for n in names)
    groups = trainable_group_list(st, make_cfg(targets=names))
    assert _numel(groups) == 43_278_336
    assert all(g.dtype == "bfloat16" for g in groups)  # TRL casts QLoRA adapters to bf16


def test_mimo_all_linear_includes_vision_without_gradients(st: ModelStructure) -> None:
    names = _targets(st, "all-linear")
    assert len(names) == 358 and "lm_head" not in names
    groups = trainable_group_list(st, make_cfg(targets=names))
    assert _numel(groups) == 51_265_024
    assert _numel(groups, grad=True) == 43_278_336
    vision = [g for g in groups if not g.receives_grad]
    assert _numel(vision) == 7_986_688 and all(g.component == "vision" for g in vision)
    assert all(isinstance(g, ArchTrainableGroup) for g in groups)
    assert all("gradient" in g.note for g in vision)


def test_text_only_scope_has_no_vision_adapters(st: ModelStructure) -> None:
    names = _targets(st, "all-linear")
    groups = trainable_group_list(st, make_cfg(scope="text_only", targets=names))
    assert _numel(groups) == 43_278_336 and all(g.receives_grad for g in groups)


def test_explicit_names_regex_and_exclude(st: ModelStructure) -> None:
    assert len(_targets(st, TEXT_NAMES)) == 248  # PEFT suffix semantics
    assert len(_targets(st, r".*language_model.*\.(q|k|v|o)_proj")) == 32  # one regex
    assert len(_targets(st, "all-linear", ["qkv", "proj", "linear_fc1", "linear_fc2"])) == 248
    assert len(_targets(st, "auto_verified", ["in_proj_a", "in_proj_b"])) == 248 - 48


def test_single_regex_in_a_list_is_a_regex(st: ModelStructure) -> None:
    # request schema: `target_modules` list = explicit suffixes or a single regex string
    assert len(_targets(st, [r".*language_model.*\.(q|k|v|o)_proj"])) == 32
    assert len(_targets(st, ["self_attn.q_proj"])) == 8  # dots alone keep suffix semantics
    assert len(_targets(st, ["all-linear"])) == 358


@pytest.mark.parametrize(
    ("target", "spec"),
    [
        ("all-linear", "all-linear"),
        (["all-linear"], "all-linear"),  # PEFT expands only the plain string
        ([r".*\.(q|v)_proj"], r".*\.(q|v)_proj"),  # PEFT fullmatches only a plain string
        (["q_proj", "v_proj"], ["q_proj", "v_proj"]),
        (["self_attn.q_proj"], ["self_attn.q_proj"]),
        ("auto_verified", "auto_verified"),
    ],
)
def test_peft_target_spec(target: str | list[str], spec: str | list[str]) -> None:
    assert peft_target_spec(target) == spec


@pytest.mark.parametrize("target", [["conv1d"], ["embed_tokens"], r".*\.mlp"])
def test_non_linear_targets_are_rejected(st: ModelStructure, target: str | list[str]) -> None:
    with pytest.raises(EstimatorError) as err:
        _targets(st, target)
    assert err.value.issue.code is ErrorCode.CONFLICTING_OPTIONS


def test_no_matching_target_is_an_error(st: ModelStructure) -> None:
    with pytest.raises(EstimatorError):
        _targets(st, ["does_not_exist"])


def test_rank_pattern_uses_real_module_dims(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    groups = trainable_group_list(st, make_cfg(targets=names, rank_pattern={"down_proj": 8}))
    # 32 down_proj (12288 -> 4096) drop from r=16 to r=8
    assert _numel(groups) == 43_278_336 - 32 * 8 * (12288 + 4096)
    assert any(g.name == "lora:A:text:8x12288" for g in groups)


def test_dora_adds_one_magnitude_per_target(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    groups = trainable_group_list(st, make_cfg(targets=names, dora=True))
    magnitude = sum(g.numel for g in groups if g.name.startswith("dora_magnitude"))
    assert magnitude == 8 * 43_008 + 24 * 45_120  # Σ out_features of the 248 targets
    assert build_trainability(st, make_cfg(targets=names, dora=True)).dora


def test_lora_adapter_dtype_follows_resolution(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    lora = trainable_group_list(st, make_cfg(strategy=Strategy.LORA, targets=names))
    assert {g.dtype for g in lora} == {"float32"}  # PEFT autocast_adapter_dtype


def test_modules_to_save_adds_a_trainable_copy(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    q = trainable_group_list(st, make_cfg(targets=names, modules_to_save=["lm_head"]))
    copy = [g for g in q if g.kind == "modules_to_save"]
    assert [(g.name, g.numel, g.dtype) for g in copy] == [
        ("modules_to_save:lm_head", 1_017_118_720, "bfloat16")  # TRL bf16 cast under QLoRA
    ]
    lora = trainable_group_list(
        st, make_cfg(strategy=Strategy.LORA, targets=names, modules_to_save=["lm_head"])
    )
    assert [g.dtype for g in lora if g.kind == "modules_to_save"] == ["bfloat16"]  # load dtype


def test_modules_to_save_excludes_inner_lora_targets(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    cfg = make_cfg(
        strategy=Strategy.LORA, scope="text_only", targets=names, modules_to_save=["mlp"]
    )
    groups = trainable_group_list(st, cfg)
    assert _numel(groups, kind="lora") == 43_278_336 - 32 * 3 * 16 * (4096 + 12288)
    assert _numel(groups, kind="modules_to_save") == 32 * 3 * 4096 * 12288
    tr = build_trainability(st, cfg)
    assert tr.module_trainable("model.language_model.layers.0.mlp.down_proj")
    assert tr.lora_rank("model.language_model.layers.0.mlp.down_proj") is None


def test_modules_to_save_of_quantized_modules_is_unsupported(st: ModelStructure) -> None:
    names = _targets(st, "auto_verified")
    with pytest.raises(EstimatorError) as err:
        trainable_group_list(st, make_cfg(targets=names, modules_to_save=["mlp"]))
    assert err.value.issue.code is ErrorCode.CONFLICTING_OPTIONS


def test_nested_modules_to_save_are_rejected(st: ModelStructure) -> None:
    cfg = make_cfg(strategy=Strategy.LORA, targets=[], modules_to_save=["mlp", "down_proj"])
    with pytest.raises(EstimatorError):
        trainable_group_list(st, cfg)


def test_bias_options(st: ModelStructure) -> None:
    names = _targets(st, "all-linear")
    all_bias = trainable_group_list(st, make_cfg(targets=names, bias="all"))
    bias = [g for g in all_bias if g.kind == "bias"]
    # MiMo has biases only in the vision tower: Linear 280,432 + LayerNorm 63,360 + patch 1,152
    assert _numel(bias) == 280_432 + 63_360 + 1_152
    assert not any(g.receives_grad for g in bias) and {g.dtype for g in bias} == {"bfloat16"}
    lora_only = trainable_group_list(st, make_cfg(targets=names, bias="lora_only"))
    assert _numel([g for g in lora_only if g.kind == "bias"]) == 280_432


def test_full_finetune_groups(st: ModelStructure) -> None:
    groups = trainable_group_list(st, make_cfg(strategy=Strategy.FULL))
    assert _numel(groups) == 9_409_813_744
    assert _numel(groups, grad=True) == 8_953_803_264  # vision tower gets no gradient
    emb = [g for g in groups if ":embedding:" in g.name and g.receives_grad]
    assert [(g.numel, g.tensor_count) for g in emb] == [(1_017_118_720, 1)]
    assert {g.kind for g in groups} == {"full"}


def test_full_finetune_patterns(st: ModelStructure) -> None:
    every = trainable_group_list(
        st, make_cfg(strategy=Strategy.FULL, trainable_full_patterns=[".*"])
    )
    assert sum(g.numel for g in every) == 9_409_813_744
    mlp_only = make_cfg(strategy=Strategy.FULL, trainable_full_patterns=[r".*\.mlp\..*"])
    groups = trainable_group_list(st, mlp_only)
    assert sum(g.numel for g in groups if g.receives_grad) == 32 * 3 * 4096 * 12288
    tr = build_trainability(st, mlp_only)
    assert tr.module_trainable("model.language_model.layers.0.mlp.up_proj")
    assert not tr.module_trainable("model.language_model.layers.0.input_layernorm")


def test_full_finetune_with_4bit_is_rejected(st: ModelStructure) -> None:
    with pytest.raises(EstimatorError):
        trainable_group_list(st, make_cfg(strategy=Strategy.FULL, quant=True))


def test_never_loaded_mtp_linears_are_not_targets(ib: ModuleType) -> None:
    tc = dict(ib.TINY_Q35_TEXT, mtp_num_hidden_layers=1)
    rows = [*ib.qwen35_text_rows(tc, prefix="model."), ("mtp.fc.weight", "bfloat16", [96, 192])]
    st = ModelStructure(ib.inventory_from_tensors(tc, rows), "qwen3_5")
    every = {m.name for m in resolve_lora_targets(st, "all-linear", [], VERIFIED)}
    assert "mtp.fc" not in every and len(every) == 8 + 7  # text decoder only, no lm_head
    for only_mtp in (["fc"], r".*fc"):  # only an unloaded module matches: nothing to adapt
        with pytest.raises(EstimatorError):
            resolve_lora_targets(st, only_mtp, [], VERIFIED)
    mixed = {m.name for m in resolve_lora_targets(st, ["fc", "down_proj"], [], VERIFIED)}
    assert mixed == {"model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"}


def test_tied_lm_head_copy_uses_the_embedding_shape(ib: ModuleType) -> None:
    cfg_dict = dict(ib.TINY_DENSE, model_type="llama", architectures=["LlamaForCausalLM"])
    cfg_dict["tie_word_embeddings"] = True
    inv = ib.inventory_from_tensors(cfg_dict, ib.dense_rows(cfg_dict))  # no lm_head tensor
    st = ModelStructure(inv, "dense")
    assert st.output_embedding == "lm_head"  # tied head module exists without its own tensor
    base = ["model.layers.0.mlp.down_proj"]
    for entry in ("lm_head", "embed_tokens"):
        cfg = make_cfg(strategy=Strategy.LORA, targets=base, modules_to_save=[entry])
        assert _numel(trainable_group_list(st, cfg), kind="modules_to_save") == 1000 * 96


def test_unknown_resolved_target_is_an_error(st: ModelStructure) -> None:
    with pytest.raises(EstimatorError):
        trainable_group_list(st, make_cfg(targets=["model.language_model.layers.0.nope"]))
