"""Parity with the real stack on CPU (torch 2.14.1, transformers 5.18.0, peft 0.21.2).

Tiny random-init models only (no downloads). Three checks:
1. PEFT trainable parameter counts == `trainable_groups` (all-linear incl. vision, explicit names,
   one regex, exclude, rank_pattern, DoRA, modules_to_save, bias) and full fine-tune counts.
2. Saved-tensor bytes of one decoder layer, measured by walking the autograd graph (storage
   dedup, parameters/buffers and cos/sin excluded, 0-dim wrapped scalars not counted), == the
   adapter's per-layer formula for the same inventory/trainability.
3. Bytes held by a real `generate()` DynamicCache == `generation_ledger` at the decode timepoint.
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from arch_helpers import make_cfg  # noqa: E402

from vramforge_estimator.architectures import GenerationTimepoints, get_adapter  # noqa: E402
from vramforge_estimator.architectures import activations as act  # noqa: E402
from vramforge_estimator.architectures.ledger import (  # noqa: E402
    act_dims,
    act_mode,
    layer_train,
)
from vramforge_estimator.architectures.structure import (  # noqa: E402
    LINEAR_ATTENTION,
    ModelStructure,
)
from vramforge_estimator.architectures.trainable import build_trainability  # noqa: E402
from vramforge_estimator.schemas import ModelInventory, ResolvedConfig, Strategy  # noqa: E402

pytestmark = pytest.mark.parity
transformers.logging.set_verbosity_error()

ROPE = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_section": [2, 2, 1],
    "mrope_interleaved": True,
}
TEXT = {
    "vocab_size": 1000,
    "hidden_size": 96,
    "intermediate_size": 176,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 40,
    "linear_num_key_heads": 3,
    "linear_num_value_heads": 9,
    "linear_key_head_dim": 24,
    "linear_value_head_dim": 28,
    "linear_conv_kernel_dim": 4,
    "tie_word_embeddings": False,
    "rope_parameters": ROPE,
}
VISION = {
    "depth": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "out_hidden_size": 96,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 64,
}
DENSE = {
    "vocab_size": 1000,
    "hidden_size": 96,
    "intermediate_size": 176,
    "num_hidden_layers": 2,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 40,
    "tie_word_embeddings": False,
}


def _init(model: Any) -> Any:
    torch.manual_seed(0)
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "A_log" in name:
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif "dt_bias" in name:
                p.fill_(1.0)
            elif p.dim() >= 2:
                p.normal_(0, 0.02)
    return model


def condgen(layer_types: list[str]) -> Any:
    text = {**TEXT, "num_hidden_layers": len(layer_types), "layer_types": layer_types}
    cfg = transformers.Qwen3_5Config(
        text_config=text, vision_config=VISION, tie_word_embeddings=False
    )
    return _init(transformers.Qwen3_5ForConditionalGeneration(cfg))


def causal_q35(layer_types: list[str]) -> Any:
    cfg = transformers.Qwen3_5TextConfig(
        **{**TEXT, "num_hidden_layers": len(layer_types), "layer_types": layer_types}
    )
    cfg._attn_implementation = "sdpa"
    return _init(transformers.Qwen3_5ForCausalLM(cfg))


def dense(kind: str) -> Any:
    if kind == "llama":
        cfg = transformers.LlamaConfig(**DENSE)
        return _init(transformers.LlamaForCausalLM(cfg))
    cfg = transformers.Qwen3Config(**DENSE)
    return _init(transformers.Qwen3ForCausalLM(cfg))


def inventory(ib: ModuleType, model: Any, arch: str) -> ModelInventory:
    config = model.config.to_dict()
    config["architectures"] = [arch]
    rows = [(n, str(t.dtype), list(t.shape)) for n, t in model.state_dict().items()]
    return ib.inventory_from_tensors(config, rows)


def trainable_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 1. trainable parameters


LORA_CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("all-linear", {"target_modules": "all-linear"}, {"target": "all-linear"}),
    (
        "names",
        {"target_modules": ["q_proj", "v_proj", "in_proj_qkv", "down_proj", "linear_fc1"]},
        {"target": ["q_proj", "v_proj", "in_proj_qkv", "down_proj", "linear_fc1"]},
    ),
    (
        "regex",
        {"target_modules": r".*language_model.*\.(q|k|v|o)_proj"},
        {"target": r".*language_model.*\.(q|k|v|o)_proj"},
    ),
    (
        "exclude",
        {"target_modules": "all-linear", "exclude_modules": ["qkv", "proj"]},
        {"target": "all-linear", "exclude": ["qkv", "proj"]},
    ),
    (
        "rank_pattern",
        {"target_modules": "all-linear", "rank_pattern": {"down_proj": 4, "in_proj_qkv": 2}},
        {"target": "all-linear", "rank_pattern": {"down_proj": 4, "in_proj_qkv": 2}},
    ),
    (
        "dora",
        {"target_modules": "all-linear", "use_dora": True},
        {"target": "all-linear", "dora": True},
    ),
    (
        "modules_to_save",
        {"target_modules": "all-linear", "modules_to_save": ["lm_head", "mlp"]},
        {"target": "all-linear", "modules_to_save": ["lm_head", "mlp"]},
    ),
    (
        "bias_all",
        {"target_modules": "all-linear", "bias": "all"},
        {"target": "all-linear", "bias": "all"},
    ),
    (
        "bias_lora_only",
        {"target_modules": "all-linear", "bias": "lora_only"},
        {"target": "all-linear", "bias": "lora_only"},
    ),
]


@pytest.mark.parametrize(
    ("label", "peft_kwargs", "ours"), LORA_CASES, ids=[c[0] for c in LORA_CASES]
)
def test_peft_trainable_counts_match(
    ib: ModuleType, label: str, peft_kwargs: dict[str, Any], ours: dict[str, Any]
) -> None:
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    model = condgen(layer_types)
    inv = inventory(ib, model, "Qwen3_5ForConditionalGeneration")
    adapter = get_adapter("qwen3_5_hybrid")
    targets = adapter.lora_target_modules(inv, ours["target"], ours.get("exclude", []))
    cfg = make_cfg(
        strategy=Strategy.LORA,
        targets=[m.name for m in targets],
        r=8,
        rank_pattern=ours.get("rank_pattern"),
        dora=ours.get("dora", False),
        modules_to_save=ours.get("modules_to_save"),
        bias=ours.get("bias", "none"),
    )
    groups = adapter.trainable_groups(inv, cfg)
    pm = peft.get_peft_model(model, peft.LoraConfig(r=8, lora_alpha=16, **peft_kwargs))
    assert sum(g.numel for g in groups) == trainable_count(pm), label
    wrapped = {n for n, mod in pm.base_model.model.named_modules() if hasattr(mod, "lora_A")}
    lora_targets = build_trainability(ModelStructure(inv, "qwen3_5"), cfg).lora_ranks
    assert wrapped == set(lora_targets), label
    adapter_dtypes = {
        str(p.dtype).removeprefix("torch.")
        for n, p in pm.named_parameters()
        if p.requires_grad and ("lora_" in n)
    }
    assert adapter_dtypes <= {g.dtype for g in groups if g.kind == "lora"}


def test_auto_verified_equals_text_linear_names_in_peft(ib: ModuleType) -> None:
    model = condgen(["linear_attention", "full_attention"])
    inv = inventory(ib, model, "Qwen3_5ForConditionalGeneration")
    adapter = get_adapter("qwen3_5_hybrid")
    names = [m.name for m in adapter.lora_target_modules(inv, "auto_verified", [])]
    assert len(names) == 8 + 7 and not any("visual" in n for n in names)
    groups = adapter.trainable_groups(inv, make_cfg(strategy=Strategy.LORA, targets=names, r=8))
    pm = peft.get_peft_model(model, peft.LoraConfig(r=8, target_modules=names))
    assert sum(g.numel for g in groups) == trainable_count(pm)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_dense_peft_and_full_counts(ib: ModuleType, kind: str) -> None:
    model = dense(kind)
    inv = inventory(ib, model, f"{kind}ForCausalLM")
    adapter = get_adapter("dense_decoder")
    full = adapter.trainable_groups(inv, make_cfg(strategy=Strategy.FULL))
    assert sum(g.numel for g in full) == sum(p.numel() for p in model.parameters())
    names = [m.name for m in adapter.lora_target_modules(inv, "auto_verified", [])]
    groups = adapter.trainable_groups(inv, make_cfg(strategy=Strategy.LORA, targets=names, r=8))
    pm = peft.get_peft_model(model, peft.LoraConfig(r=8, target_modules="all-linear"))
    assert sum(g.numel for g in groups) == trainable_count(pm)


def test_full_finetune_counts_include_the_frozen_in_data_vision_tower(ib: ModuleType) -> None:
    model = condgen(["linear_attention", "full_attention"])
    inv = inventory(ib, model, "Qwen3_5ForConditionalGeneration")
    groups = get_adapter("qwen3_5_hybrid").trainable_groups(inv, make_cfg(strategy=Strategy.FULL))
    assert sum(g.numel for g in groups) == sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for n, p in model.named_parameters() if "visual" in n)
    assert sum(g.numel for g in groups if not g.receives_grad) == vision


# ---------------------------------------------------------------- 2. saved activations


def _skey(t: Any) -> int:
    return t.untyped_storage()._cdata


def saved_bytes(roots: list[Any], exclude: set[int]) -> int:
    """Walk the autograd graph from `roots`, dedup saved tensors by storage."""
    seen: set[Any] = set()
    stack = [r.grad_fn for r in roots if r.grad_fn is not None]
    storages: dict[int, int] = {}
    while stack:
        fn = stack.pop()
        if fn in seen:
            continue
        seen.add(fn)
        tensors: list[Any] = []
        for attr in dir(fn):
            if attr.startswith("_saved_"):
                with contextlib.suppress(Exception):
                    value = getattr(fn, attr)
                    tensors += value if isinstance(value, tuple | list) else [value]
        if hasattr(fn, "saved_tensors"):
            with contextlib.suppress(Exception):
                tensors += list(fn.saved_tensors)
        for t in tensors:
            if not isinstance(t, torch.Tensor) or _skey(t) in exclude:
                continue
            if t.dim() == 0 and t.dtype == torch.float64:
                continue  # wrapped python scalar (golden convention, research §F)
            storages.setdefault(_skey(t), t.untyped_storage().nbytes())
        stack += [nf for nf, _ in fn.next_functions if nf is not None and nf not in seen]
    return sum(storages.values())


MODES = {
    "full_autocast": ({"strategy": Strategy.FULL}, None, True),
    "lora_bf16_autocast": ({"strategy": Strategy.LORA, "adapter": "bfloat16"}, False, True),
    "lora_fp32_autocast": ({"strategy": Strategy.LORA}, True, True),
    "lora_fp32_plain": ({"strategy": Strategy.LORA, "compute": "float32"}, True, False),
}


def _measure_layer(
    model: Any, peft_model: Any, index: int, batch: int, seq: int, autocast: bool, mrope: bool
) -> int:
    layer = model.model.layers[index]
    x = torch.randn(batch, seq, model.config.hidden_size, dtype=torch.bfloat16).requires_grad_(True)
    if mrope:
        pos = torch.arange(seq).view(1, 1, -1).expand(4, batch, -1)
        pe, pid = model.model.rotary_emb(x, pos[1:]), pos[0]
    else:
        pid = torch.arange(seq)[None]
        pe = model.model.rotary_emb(x, pid)
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        out = layer(
            x, position_embeddings=pe, attention_mask=None, position_ids=pid, use_cache=False
        )
    exclude = {_skey(p) for p in peft_model.parameters()} | {_skey(b) for b in peft_model.buffers()}
    exclude |= {_skey(pe[0]), _skey(pe[1])}
    return saved_bytes([out], exclude)


def _expected_layer(
    inv: ModelInventory, family: str, cfg: ResolvedConfig, index: int, batch: int, seq: int
) -> int:
    st = ModelStructure(inv, family)  # type: ignore[arg-type]
    tr = build_trainability(st, cfg)
    mode = act_mode(cfg)
    assert mode is not None
    lt = layer_train(st, st.layers[index], tr)
    flash = act.AttnPath("flash", mask=False, expand_kv=False)
    if family == "dense":
        return act.total(act.dense_layer(batch, seq, act_dims(st), lt, mode, flash))
    if st.layer_types[index] == LINEAR_ATTENTION:
        return act.total(
            act.q35_linear_attention_layer(batch, seq, act_dims(st), lt, mode, "torch")
        )
    return act.total(act.q35_full_attention_layer(batch, seq, act_dims(st), lt, mode, flash))


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(("batch", "seq"), [(1, 40), (2, 100)])
def test_qwen35_layer_saved_bytes_match_torch(
    ib: ModuleType, mode: str, batch: int, seq: int
) -> None:
    cfg_kwargs, autocast_adapter, autocast = MODES[mode]
    model = causal_q35(["linear_attention", "linear_attention", "full_attention"])
    inv = inventory(ib, model, "Qwen3_5ForCausalLM")
    adapter = get_adapter("qwen3_5_hybrid")
    targets = [m.name for m in adapter.lora_target_modules(inv, "all-linear", [])]
    cfg = make_cfg(targets=targets, r=8, **cfg_kwargs)
    top = model
    if cfg.strategy is Strategy.LORA:
        top = peft.get_peft_model(
            model,
            peft.LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"),
            autocast_adapter_dtype=autocast_adapter,
        )
    top.train()
    for index in (1, 2):  # a non-first linear layer and the full-attention layer
        measured = _measure_layer(model, top, index, batch, seq, autocast, mrope=True)
        assert measured == _expected_layer(inv, "qwen3_5", cfg, index, batch, seq), (mode, index)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("mode", ["full_autocast", "lora_bf16_autocast", "lora_fp32_autocast"])
def test_dense_layer_saved_bytes_match_torch(ib: ModuleType, kind: str, mode: str) -> None:
    cfg_kwargs, autocast_adapter, autocast = MODES[mode]
    model = dense(kind)
    model.config._attn_implementation = "sdpa"
    inv = inventory(ib, model, f"{kind}ForCausalLM")
    adapter = get_adapter("dense_decoder")
    targets = [m.name for m in adapter.lora_target_modules(inv, "all-linear", [])]
    cfg = make_cfg(targets=targets, r=8, **cfg_kwargs)
    top = model
    if cfg.strategy is Strategy.LORA:
        top = peft.get_peft_model(
            model,
            peft.LoraConfig(r=8, target_modules="all-linear"),
            autocast_adapter_dtype=autocast_adapter,
        )
    top.train()
    measured = _measure_layer(model, top, 1, 2, 100, autocast, mrope=False)
    assert measured == _expected_layer(inv, "dense", cfg, 1, 2, 100)


# ---------------------------------------------------------------- 3. generation cache


def _cache_bytes(cache: Any) -> int:
    storages: dict[int, int] = {}
    for layer in cache.layers:
        for value in vars(layer).values():
            if isinstance(value, dict):
                items = list(value.values())  # LinearAttentionLayer: {state_idx: tensor}
            else:
                items = list(value) if isinstance(value, list | tuple) else [value]
            for t in items:
                if isinstance(t, torch.Tensor) and t.numel() > 1:
                    storages.setdefault(_skey(t), t.untyped_storage().nbytes())
    return sum(storages.values())


@pytest.mark.parametrize(
    ("family", "c", "prompt", "new"), [("qwen3_5", 2, 7, 5), ("dense", 3, 9, 4)]
)
def test_generate_cache_matches_ledger(
    ib: ModuleType, family: str, c: int, prompt: int, new: int
) -> None:
    if family == "qwen3_5":
        model = causal_q35(
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        )
        inv, adapter = inventory(ib, model, "Qwen3_5ForCausalLM"), get_adapter("qwen3_5_hybrid")
    else:
        model = dense("llama")
        inv, adapter = inventory(ib, model, "LlamaForCausalLM"), get_adapter("dense_decoder")
    model.eval()
    ids = torch.randint(5, 900, (c, prompt))
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=new,
            min_new_tokens=new,
            do_sample=False,
            pad_token_id=0,
            return_dict_in_generate=True,
        )
    tps = GenerationTimepoints("G:prefill", "G:decode")
    cfg = make_cfg(strategy=Strategy.LORA, targets=[])
    ledger = adapter.generation_ledger(inv, cfg, c, prompt, new, tps, "gen")
    expected = sum(
        a.bytes_low
        for a in ledger
        if a.category.value in ("generation_cache", "recurrent_state") and "G:decode" in a.live_at
    )
    assert _cache_bytes(out.past_key_values) == expected
