"""Profile registry: loading, validation, dependency lock digest (plan §11.2)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from vramforge_estimator.compatibility.profiles import (
    PROFILES_DIR_ENV,
    EnvironmentProfile,
    ProfileError,
    clear_registry_cache,
    default_profiles_dir,
    load_registry,
    lock_digest,
)
from vramforge_estimator.units import GiB

REPO_PROFILES = Path(__file__).resolve().parents[3] / "profiles"
ENV_ID = "cuda-trl-1.14.1"
# Pinned by hand: changing a package version must be a deliberate, reviewed profile change.
EXPECTED_DIGEST = "sha256:ccd37fedde9c120a68463ded39d0925353779d48665408c54869ea67b210cc87"
PINNED = {
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
    "datasets": "5.0.1",
    "huggingface-hub": "1.33.0",
    "peft": "0.21.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.2",
    "torch": "2.14.1",
    "transformers": "5.18.0",
    "trl": "1.14.1",
}


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(PROFILES_DIR_ENV, raising=False)
    clear_registry_cache()
    yield
    clear_registry_cache()


def test_default_dir_is_repo_profiles() -> None:
    assert default_profiles_dir().resolve() == REPO_PROFILES.resolve()


def test_lock_digest_is_stable_and_matches_profile() -> None:
    env = load_registry().environments[ENV_ID]
    assert env.packages == PINNED
    assert lock_digest(PINNED) == EXPECTED_DIGEST
    assert env.dependency_lock_digest == EXPECTED_DIGEST


def test_lock_digest_ignores_order_and_name_spelling() -> None:
    reordered = dict(reversed(list(PINNED.items())))
    respelled = {k.replace("-", "_").upper(): v for k, v in PINNED.items()}
    assert lock_digest(reordered) == EXPECTED_DIGEST
    assert lock_digest(respelled) == EXPECTED_DIGEST
    assert lock_digest({**PINNED, "trl": "1.14.2"}) != EXPECTED_DIGEST


def test_environment_rejects_stale_digest() -> None:
    env = load_registry().environments[ENV_ID]
    data = env.model_dump(mode="json")
    data["packages"]["torch"] = "2.14.2"
    with pytest.raises(ValidationError, match="dependency_lock_digest"):
        EnvironmentProfile.model_validate(data)


def test_environment_has_no_optional_kernels() -> None:
    env = load_registry().environments[ENV_ID]
    assert env.kernels.installed == []
    assert env.kernels.linear_attention == "torch_fallback"
    for pkg in ("flash-linear-attention", "causal-conv1d", "liger-kernel"):
        assert pkg in env.kernels.absent
    assert env.trainer_defaults["model_init_dtype"] == "float32"
    assert env.trainer_defaults["optim"] == "adamw_torch_fused"


def test_trainer_contracts_pin_no_truncation_fields() -> None:
    env = load_registry().environments[ENV_ID]
    sft, dpo, grpo = (env.trainers[o] for o in ("sft", "dpo", "grpo"))
    assert sft.no_truncation["max_length"] is None and sft.no_truncation["packing"] is False
    assert dpo.no_truncation["max_length"] is None
    assert "max_prompt_length" in dpo.removed_fields and "rpo_alpha" in dpo.removed_fields
    assert grpo.removed_fields == ["max_prompt_length"]
    assert set(grpo.forbidden_template_kwargs) == {"truncation", "max_length"}


def test_hardware_presets_are_nominal_binary_sizes() -> None:
    reg = load_registry()
    presets = reg.gpu_presets()
    assert len({p.id for p in presets}) == len(presets) > 0
    h100 = reg.gpu("h100-80gb")
    assert h100 is not None and h100.total_bytes == 80 * GiB
    assert all("nvidia-smi" in p.note for p in presets)


def test_profiles_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "profiles"
    shutil.copytree(REPO_PROFILES, target)
    monkeypatch.setenv(PROFILES_DIR_ENV, str(target))
    reg = load_registry()
    assert reg.root == target.resolve()
    assert ENV_ID in reg.environments


def test_missing_profiles_dir_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROFILES_DIR_ENV, str(tmp_path / "nope"))
    with pytest.raises(ProfileError):
        load_registry()


def test_invalid_profile_is_reported_with_relative_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "profiles"
    shutil.copytree(REPO_PROFILES, target)
    bad = target / "environments" / "cuda-trl-1.14.1.yaml"
    bad.write_text(bad.read_text(encoding="utf-8").replace('trl: "1.14.1"', 'trl: "9.9.9"'))
    monkeypatch.setenv(PROFILES_DIR_ENV, str(target))
    with pytest.raises(ProfileError) as info:
        load_registry()
    assert "environments/cuda-trl-1.14.1.yaml" in str(info.value)
    assert str(tmp_path) not in str(info.value)


# ---------------------------------------------------------------- analytic profiles


def test_analytic_profiles_cover_both_adapters() -> None:
    reg = load_registry()
    assert {p.architecture_adapter for p in reg.analytic.values()} == {
        "dense_decoder",
        "qwen3_5_hybrid",
    }
    for prof in reg.analytic.values():
        assert reg.environment_for(prof).id == ENV_ID
        assert prof.loading.default_load_dtype == "bfloat16"
        assert set(prof.loading.load_dtypes) == {"bfloat16", "float32"}
        q = prof.quantization
        assert (q.default_format, q.double_quant, q.compute_dtype) == ("nf4", True, "bfloat16")
        assert q.skip_modules == ["lm_head"] and q.blocksize == 64 and q.nested_blocksize == 256
        assert prof.checkpointing.granularity == "per_decoder_layer"
        assert prof.presets["sft"].accumulation == prof.presets["dpo"].accumulation == 8
        assert prof.presets["grpo"].accumulation == 4
        assert all(p.microbatch == 1 for p in prof.presets.values())
        assert prof.calibration_coverage == []


def test_profile_constants_match_the_trainer_code() -> None:
    from fractions import Fraction

    from vramforge_estimator.trainers.common import DEVICE_MAP_BUDGET
    from vramforge_estimator.trainers.registry import trainer_id_for

    for prof in load_registry().analytic.values():
        factors = prof.loading.device_map_budget_factor
        assert {k: Fraction(str(v)) for k, v in factors.items()} == DEVICE_MAP_BUDGET
        assert {o.value: t for o, t in prof.trainers.items()} == {
            o: trainer_id_for(o) for o in ("sft", "dpo", "grpo")
        }


def test_hybrid_profile_runs_linear_attention_on_the_torch_fallback() -> None:
    prof = load_registry().profile_for_adapter("qwen3_5_hybrid")
    assert prof is not None
    assert prof.attention.by_layer_type == {
        "full_attention": "sdpa",
        "linear_attention": "torch_fallback",
    }
    assert prof.loading.text_only.verified is False
    rule = next(r for r in prof.loading.scope_rules if r.architecture.endswith("Generation"))
    assert rule.scope == "full_checkpoint"


def test_workspace_assumptions_are_sourced_ranges() -> None:
    for prof in load_registry().analytic.values():
        ws = prof.workspace
        assert ws.cuda_context_bytes.low == int(0.3 * GiB)
        assert ws.cuda_context_bytes.high == 1 * GiB
        for rng in (ws.cuda_context_bytes, ws.library_workspace_bytes, ws.allocator_slack_fraction):
            assert rng.low <= rng.high and rng.source
