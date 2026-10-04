"""LoRA targets recorded for the exported LoraConfig follow PEFT 0.21.2 semantics.

`peft_target_spec(target_module_patterns)` is what the trainer config must hand to PEFT: a plain
string for "all-linear" or one regex (a list holding them matches nothing), a list of exact names
or `.suffix` entries otherwise. Feeding that value back through the architecture adapter, which
resolves targets with PEFT's rules, must select exactly the resolved modules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from vf_fakes import make_inventory

from vramforge_estimator.architectures import get_adapter
from vramforge_estimator.architectures.trainable import peft_target_spec
from vramforge_estimator.compatibility import resolve
from vramforge_estimator.compatibility import resolver as resolver_mod
from vramforge_estimator.schemas import AnalysisRequest, LinearModule, ModelComponent

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "tests" / "fixtures" / "inventories" / "inventory_builder.py"
REQUEST = ROOT / "tests" / "fixtures" / "requests" / "plan_example_grpo.json"
QKV_REGEX = r".*\.self_attn\.(q|k)_proj"


@pytest.fixture(scope="module")
def mimo():
    if not BUILDER.exists():
        pytest.skip("MiMo inventory fixture not available")
    spec = importlib.util.spec_from_file_location("vf_inventory_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vf_inventory_builder"] = module
    spec.loader.exec_module(module)
    return module.load_mimo_inventory()


def request(lora: dict[str, Any]) -> AnalysisRequest:
    data = json.loads(REQUEST.read_text(encoding="utf-8"))
    data["training"]["lora"].update(lora)
    return AnalysisRequest.model_validate(data)


@pytest.mark.parametrize(
    ("lora", "as_string", "count"),
    [
        ({"target_modules": "auto_verified"}, False, 248),
        ({"target_modules": "all-linear"}, True, 358),  # vision Linear included, lm_head not
        ({"target_modules": ["all-linear"]}, True, 358),
        ({"target_modules": [QKV_REGEX]}, True, 16),
        ({"target_modules": ["q_proj", "v_proj"]}, False, 16),
        (
            {
                "target_modules": "auto_verified",
                "exclude_modules": ["model.language_model.layers.3.self_attn.q_proj"],
            },
            False,
            247,
        ),
    ],
)
def test_emitted_targets_round_trip_through_peft_semantics(mimo, lora, as_string, count) -> None:
    cfg, report = resolve(request(lora), mimo, None)
    assert cfg is not None and cfg.lora is not None, report.blockers
    assert len(cfg.lora.target_modules) == count
    spec = peft_target_spec(cfg.lora.target_module_patterns)
    assert isinstance(spec, str) is as_string
    arch = get_adapter("qwen3_5_hybrid")
    again = arch.lora_target_modules(mimo, spec, list(cfg.lora.exclude_modules))
    assert {m.name for m in again} == set(cfg.lora.target_modules)
    reason = next(r.reason for r in cfg.resolutions if r.field == "training.lora.target_modules")
    assert ("문자열로 전달" in reason) is as_string


def test_auto_verified_emits_leaf_kinds_unless_they_select_more(mimo) -> None:
    cfg, _ = resolve(request({"target_modules": "auto_verified"}), mimo, None)
    assert cfg is not None and cfg.lora is not None
    assert cfg.lora.target_module_patterns == [
        "down_proj",
        "gate_proj",
        "in_proj_a",
        "in_proj_b",
        "in_proj_qkv",
        "in_proj_z",
        "k_proj",
        "o_proj",
        "out_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    excluded = "model.language_model.layers.3.self_attn.q_proj"
    narrowed, _ = resolve(
        request({"target_modules": "auto_verified", "exclude_modules": [excluded]}), mimo, None
    )
    assert narrowed is not None and narrowed.lora is not None
    patterns = narrowed.lora.target_module_patterns
    assert excluded not in patterns and patterns == sorted(narrowed.lora.target_modules)


def test_never_loaded_mtp_linears_do_not_force_full_names() -> None:
    inv = make_inventory()
    text = [m for m in inv.linear_modules if m.component is ModelComponent.TEXT]
    mtp = LinearModule(
        name="mtp.layers.0.self_attn.q_proj",
        kind="q_proj",
        in_features=64,
        out_features=64,
        has_bias=False,
        component=ModelComponent.MTP,
        dtype="bfloat16",
    )
    with_mtp = inv.model_copy(update={"linear_modules": [*inv.linear_modules, mtp]})
    kinds = sorted({m.kind for m in text})
    assert resolver_mod._emit_patterns("auto_verified", text, with_mtp) == kinds
    # a loaded module of the same kind outside the targets does force full names
    loaded = mtp.model_copy(update={"component": ModelComponent.VISION, "name": "visual.q_proj"})
    with_vision = inv.model_copy(update={"linear_modules": [*inv.linear_modules, loaded]})
    names = resolver_mod._emit_patterns("auto_verified", text, with_vision)
    assert names == sorted(m.name for m in text)
