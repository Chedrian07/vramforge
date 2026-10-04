"""The exported trainer config is accepted by the real TRL 1.14.1 / PEFT / transformers classes.

Runs only with the `parity` dependency group (torch, trl, peft) installed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from exports_testkit import GRPO_BUDGET, build_result

from vramforge_estimator.exports import export_trainer_config
from vramforge_estimator.exports.trainer_config import FIELDS, FORBIDDEN_FIELDS
from vramforge_estimator.schemas import Objective

pytestmark = pytest.mark.parity


def _config_class(objective: Objective):
    import trl

    return {
        Objective.SFT: trl.SFTConfig,
        Objective.DPO: trl.DPOConfig,
        Objective.GRPO: trl.GRPOConfig,
    }[objective]


@pytest.mark.parametrize("objective", list(Objective))
def test_emitted_fields_exist_and_forbidden_ones_do_not(objective: Objective) -> None:
    names = {f.name for f in dataclasses.fields(_config_class(objective))}
    assert FIELDS[objective] <= names, sorted(FIELDS[objective] - names)
    assert not (FORBIDDEN_FIELDS - {"truncation_mode", "steps_per_generation"}) & names


@pytest.mark.parametrize("objective", ["sft", "dpo", "grpo"])
def test_trl_instantiates_the_exported_args(
    objective: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft
    import transformers

    data = yaml.safe_load(export_trainer_config(build_result(objective, tmp_path, monkeypatch)))
    obj = Objective(objective)
    args = dict(data["trl"]["args"])
    args["output_dir"] = str(tmp_path / "out")
    # Test-only: TrainingArguments rejects bf16=True without a GPU unless use_cpu is set (Linux CI
    # runners have no GPU). The exported config itself never carries use_cpu.
    assert "use_cpu" not in args
    args["use_cpu"] = True
    config = _config_class(obj)(**args)
    if obj in (Objective.SFT, Objective.DPO):
        assert config.max_length is None
    if obj is Objective.SFT:
        assert config.packing is False
    if obj is Objective.GRPO:
        assert config.max_completion_length == GRPO_BUDGET
        assert not hasattr(config, "max_prompt_length")
        assert config.steps_per_generation == 4  # derived from generation_batch_size
    assert config.model_init_kwargs["dtype"] == "bfloat16"
    transformers.BitsAndBytesConfig(**data["quantization"])
    peft.LoraConfig(**data["peft"])
