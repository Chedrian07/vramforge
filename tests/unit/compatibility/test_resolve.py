"""Compatibility resolution (plan §11): presets, dtypes, LoRA targets, reference, GRPO, readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from vf_fakes import FakeArch, lora_numel, make_inventory, text_targets

from vramforge_estimator.compatibility import backend_profiles, resolve, support_for
from vramforge_estimator.compatibility import resolver as resolver_mod
from vramforge_estimator.schemas import (
    AnalysisRequest,
    ErrorCode,
    EvidenceLevel,
    LayerTypeCount,
    ReferenceStrategy,
    TokenizerManifest,
    TrainingReadiness,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "requests" / "plan_example_grpo.json"
HYBRID = "qwen3_5_hybrid"


def request(**changes: Any) -> AnalysisRequest:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for path, value in changes.items():
        node = data
        *parents, leaf = path.split("__")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return AnalysisRequest.model_validate(data)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch):
    """Route structural matching to `holder["id"]` and every adapter to the FakeArch."""
    holder: dict[str, str | None] = {"id": HYBRID}
    monkeypatch.setattr(resolver_mod, "match_adapter", lambda facts: holder["id"])
    monkeypatch.setattr(resolver_mod, "get_adapter", lambda adapter_id: FakeArch(adapter_id))
    return holder


def hybrid_inventory(vision: bool = True):
    inv = make_inventory(vision=vision)
    facts = inv.facts.model_copy(
        update={
            "layer_types": ["linear_attention", "full_attention"],
            "layer_type_counts": [
                LayerTypeCount(layer_type="linear_attention", count=1),
                LayerTypeCount(layer_type="full_attention", count=1),
            ],
        }
    )
    return inv.model_copy(update={"facts": facts})


def field(cfg, name):
    return next(r for r in cfg.resolutions if r.field == name)


def test_plan_example_grpo_resolution(adapter) -> None:
    inv = hybrid_inventory()
    cfg, report = resolve(request(), inv, None)
    assert cfg is not None and report.blockers == []
    assert (cfg.profile_id, cfg.architecture_adapter) == ("qwen3_5-hybrid", HYBRID)
    assert cfg.trainer_adapter == cfg.preprocessing_adapter == "trl-1.14.1-grpo"
    assert cfg.dependency_lock_digest and cfg.dependency_lock_digest.startswith("sha256:")
    assert cfg.loading_scope == "full_checkpoint"
    assert cfg.load_dtype == "bfloat16"
    q = cfg.quantization
    assert (q.method, q.double_quant, q.compute_dtype, q.blocksize) == (
        "bnb_nf4",
        True,
        "bfloat16",
        64,
    )
    assert q.skip_module_patterns == ["lm_head"]
    d = cfg.effective_dtypes
    assert (d.adapter, d.gradient, d.optimizer_state) == ("bfloat16",) * 3
    assert (d.logits, d.kv_cache, d.recurrent_state) == ("float32", "bfloat16", "float32")
    assert d.master_weights is None
    assert (cfg.microbatch, cfg.accumulation) == (1, 4)
    g = cfg.grpo
    assert g is not None
    assert (g.generation_batch_size, g.steps_per_generation, g.live_sequences) == (4, 4, 4)
    assert (g.update_microbatch, g.accumulation) == (1, 4)
    assert g.completion_budgets == [1024, 2048, 4096, 8192] and not g.budget_explicit
    assert g.reference_needed is False  # beta = 0.0
    assert cfg.attention_path_by_layer_type == {
        "linear_attention": "torch_fallback",
        "full_attention": "sdpa",
    }
    assert cfg.loss_path == "trl_fused_logprob"
    assert cfg.checkpointing_granularity == "per_decoder_layer"
    assert report.readiness is TrainingReadiness.CONDITIONAL
    assert ErrorCode.GRPO_REWARD_UNSPECIFIED in {w.code for w in report.warnings}
    assert report.support_grade is EvidenceLevel.ANALYTIC and len(report.support) == 9


def test_lora_targets_and_trainable_count(adapter) -> None:
    inv = hybrid_inventory()
    cfg, _ = resolve(request(), inv, None)
    assert cfg is not None and cfg.lora is not None
    assert cfg.lora.target_modules == text_targets(inv)
    assert cfg.lora.target_module_patterns == sorted(
        {m.rsplit(".", 1)[-1] for m in text_targets(inv)}
    )
    assert field(cfg, "lora.trainable_params").resolved == lora_numel(inv, text_targets(inv))


def test_all_linear_on_a_vlm_warns_about_vision_adapters(adapter) -> None:
    cfg, report = resolve(
        request(training__lora={"target_modules": "all-linear"}), hybrid_inventory(), None
    )
    assert cfg is not None and cfg.lora is not None
    assert cfg.lora.target_module_patterns == ["all-linear"]
    assert any("텍스트 디코더 밖" in w.user_message for w in report.warnings)


def test_rank_pattern_changes_the_count_but_alpha_does_not(adapter) -> None:
    inv = hybrid_inventory()
    base, _ = resolve(request(training__lora={"alpha": 64}), inv, None)
    ranked, _ = resolve(request(training__lora={"rank_pattern": {"q_proj": 4}}), inv, None)
    plain, _ = resolve(request(), inv, None)
    count = lambda c: field(c, "lora.trainable_params").resolved  # noqa: E731
    assert count(base) == count(plain)
    assert count(ranked) < count(plain)


def test_sft_and_dpo_presets(adapter) -> None:
    inv = hybrid_inventory()
    unset = {"training__microbatch_per_device": None, "training__gradient_accumulation_steps": None}
    sft, _ = resolve(request(training__objective="sft", **unset), inv, None)
    assert sft is not None and (sft.microbatch, sft.accumulation) == (1, 8)
    assert sft.loss_path == "trl_chunked_nll" and sft.grpo is None
    nll, _ = resolve(
        request(training__objective="sft", training__loss_kernel="standard"), inv, None
    )
    assert nll is not None and nll.loss_path == "hf_ce"
    dpo, _ = resolve(request(training__objective="dpo", **unset), inv, None)
    assert dpo is not None and dpo.dpo is not None
    assert dpo.dpo.reference_strategy is ReferenceStrategy.FROZEN_BASE_SWITCH
    assert (dpo.microbatch, dpo.accumulation) == (1, 8)
    grpo, _ = resolve(request(**unset), inv, None)
    assert grpo is not None and (grpo.microbatch, grpo.accumulation) == (1, 4)
    assert field(grpo, "training.gradient_accumulation_steps").reason == "제품 preset"


def test_dpo_reference_auto_and_separate_checkpoint(adapter) -> None:
    full = dict(
        training__objective="dpo",
        training__strategy="full",
        training__quantization={"enabled": False},
    )
    cfg, _ = resolve(request(**full), hybrid_inventory(), None)
    assert cfg is not None and cfg.dpo.reference_strategy is ReferenceStrategy.STANDALONE_MODEL
    assert cfg.effective_dtypes.adapter == cfg.load_dtype
    separate, _ = resolve(
        request(**full, dpo={"reference_model": "org/other"}), hybrid_inventory(), None
    )
    assert separate is not None
    assert field(separate, "dpo.reference_model").resolved == "org/other"


def test_lora_without_quantization_uses_fp32_adapter(adapter) -> None:
    cfg, _ = resolve(
        request(training__strategy="lora", training__quantization={"enabled": False}),
        hybrid_inventory(),
        None,
    )
    assert cfg is not None and not cfg.quantization.enabled
    assert cfg.effective_dtypes.adapter == "float32" == cfg.effective_dtypes.optimizer_state


def test_float32_load_is_an_explicit_option(adapter) -> None:
    cfg, _ = resolve(request(training__load_dtype="float32"), hybrid_inventory(), None)
    assert cfg is not None and cfg.load_dtype == "float32"
    assert cfg.effective_dtypes.kv_cache == "float32"


def test_requested_linear_attention_kernel_is_not_effective(adapter) -> None:
    cfg, report = resolve(
        request(training__linear_attention_kernel="fla"), hybrid_inventory(), None
    )
    assert cfg is not None
    assert cfg.attention_path_by_layer_type["linear_attention"] == "torch_fallback"
    assert [r.field for r in report.not_effective] == ["training.linear_attention_kernel"]


def test_text_only_on_a_vision_checkpoint_is_unsupported(adapter) -> None:
    cfg, report = resolve(request(model__loading_scope="text_only"), hybrid_inventory(), None)
    assert cfg is None
    assert report.readiness is TrainingReadiness.UNSUPPORTED
    assert ErrorCode.UNSUPPORTED_BACKEND_COMBINATION in {b.code for b in report.blockers}
    text_model, _ = resolve(
        request(model__loading_scope="text_only"), hybrid_inventory(vision=False), None
    )
    assert text_model is not None and text_model.loading_scope == "full_checkpoint"


def test_unsupported_architecture(adapter) -> None:
    adapter["id"] = None
    cfg, report = resolve(request(), hybrid_inventory(), None)
    assert cfg is None and report.readiness is TrainingReadiness.UNSUPPORTED
    assert report.blockers[0].code is ErrorCode.UNSUPPORTED_ARCHITECTURE


def test_prequantized_checkpoint_is_blocked(adapter) -> None:
    inv = hybrid_inventory().model_copy(update={"quantized_checkpoint_format": "gptq"})
    cfg, report = resolve(request(), inv, None)
    assert cfg is None
    assert ErrorCode.UNSUPPORTED_MODEL_FORMAT in {b.code for b in report.blockers}


def test_reward_placement_drives_readiness(adapter) -> None:
    inv = hybrid_inventory()
    local = {"kind": "local_model", "model_reference": "org/rm"}
    ready = resolve(request(grpo__reward=local), inv, None)[1]
    assert ready.readiness is TrainingReadiness.READY
    elsewhere = resolve(request(grpo__reward={**local, "on_training_gpu": False}), inv, None)[1]
    assert elsewhere.readiness is TrainingReadiness.CONDITIONAL
    for kind in ("cpu_rule", "remote"):
        assert resolve(request(grpo__reward={"kind": kind}), inv, None)[1].readiness is (
            TrainingReadiness.CONDITIONAL
        )


def test_grpo_explicit_budget_and_max_live_sequences(adapter) -> None:
    cfg, report = resolve(
        request(grpo__completion_budget=2048, grpo__max_live_sequences=1), hybrid_inventory(), None
    )
    assert cfg is not None and cfg.grpo.completion_budgets == [2048] and cfg.grpo.budget_explicit
    assert "grpo.max_live_sequences" in {r.field for r in report.not_effective}


def test_template_kwarg_not_used_by_the_template(adapter) -> None:
    tok = TokenizerManifest(
        tokenizer_class="T", vocab_size=10, chat_template_present=True, fingerprint="x"
    )
    cfg, report = resolve(
        request(training__template={"enable_thinking": False}), hybrid_inventory(), tok
    )
    assert cfg is not None and cfg.template_kwargs == {}
    assert "training.template.enable_thinking" in {r.field for r in report.not_effective}
    supported = tok.model_copy(update={"template_kwargs": ["enable_thinking"]})
    cfg2, _ = resolve(
        request(training__template={"enable_thinking": False}), hybrid_inventory(), supported
    )
    assert cfg2 is not None and cfg2.template_kwargs == {"enable_thinking": False}


def test_every_knob_records_requested_and_resolved(adapter) -> None:
    cfg, _ = resolve(request(), hybrid_inventory(), None)
    fields = {r.field for r in cfg.resolutions}
    for name in (
        "model.loading_scope",
        "training.load_dtype",
        "training.quantization",
        "training.lora.target_modules",
        "training.optimizer",
        "training.microbatch_per_device",
        "training.gradient_accumulation_steps",
        "training.attention_backend",
        "training.loss_kernel",
        "grpo.generation_batch_size",
        "grpo.steps_per_generation",
        "grpo.completion_budget",
    ):
        assert name in fields
    assert all(r.reason for r in cfg.resolutions)


def test_support_for_and_backend_profiles(adapter) -> None:
    adapter_id, entries = support_for(hybrid_inventory().facts)
    assert adapter_id == HYBRID and len(entries) == 9
    adapter["id"] = None
    assert support_for(hybrid_inventory().facts) == (None, [])
    resp = backend_profiles()
    assert {p.architecture_adapter for p in resp.profiles} == {"dense_decoder", HYBRID}
    assert resp.gpu_worker_connected is False
    assert resp.environments[0].dependency_lock_digest.startswith("sha256:")
    assert resp.hardware_presets and all(p.total_bytes > 0 for p in resp.hardware_presets)


class _RaisingArch(FakeArch):
    def trainable_groups(self, inventory, cfg):
        from vramforge_estimator.errors import EstimatorError, make_issue

        raise EstimatorError(
            make_issue(ErrorCode.CONFLICTING_OPTIONS, "4-bit 모듈은 사본 학습 불가")
        )

    def lora_target_modules(self, inventory, target, exclude):
        if target == "all-linear":
            from vramforge_estimator.errors import EstimatorError, make_issue

            raise EstimatorError(make_issue(ErrorCode.CONFLICTING_OPTIONS, "대상 없음"))
        return super().lora_target_modules(inventory, target, exclude)


def test_adapter_configuration_errors_become_blockers(adapter, monkeypatch) -> None:
    monkeypatch.setattr(resolver_mod, "get_adapter", lambda adapter_id: _RaisingArch(adapter_id))
    cfg, report = resolve(request(), hybrid_inventory(), None)
    assert cfg is None and report.readiness is TrainingReadiness.UNSUPPORTED
    assert [b.user_message for b in report.blockers] == ["4-bit 모듈은 사본 학습 불가"]
    cfg2, report2 = resolve(
        request(training__lora={"target_modules": "all-linear"}), hybrid_inventory(), None
    )
    assert cfg2 is None and report2.blockers[0].user_message == "대상 없음"


def test_excluding_a_configured_evaluation_makes_the_result_conditional(adapter) -> None:
    inv = hybrid_inventory()
    sft = {"training__objective": "sft"}
    assert resolve(request(**sft), inv, None)[1].readiness is TrainingReadiness.READY
    report = resolve(request(**sft, dataset__eval_split="test"), inv, None)[1]
    assert report.readiness is TrainingReadiness.CONDITIONAL
    assert ErrorCode.PROFILE_SCOPE_INCOMPLETE in {w.code for w in report.warnings}
    included = resolve(
        request(**sft, dataset__eval_split="test", scope={"include_evaluation": True}), inv, None
    )[1]
    assert included.readiness is TrainingReadiness.READY


def test_dpo_records_the_effective_dropout_and_reference_settings(adapter) -> None:
    # plan §5.3: record beta, loss type, reference sync and the effective dropout. TRL DPO's
    # disable_dropout=True sets every nn.Dropout (LoRA dropout included) to p = 0.
    dpo = {"training__objective": "dpo", "training__lora": {"dropout": 0.05}}
    cfg, report = resolve(request(**dpo), hybrid_inventory(), None)
    assert cfg is not None and cfg.lora is not None and cfg.lora.dropout == 0.0
    dropout = next(r for r in report.not_effective if r.field == "training.lora.dropout")
    assert (dropout.requested, dropout.resolved) == (0.05, 0.0)
    assert field(cfg, "dpo.disable_dropout").resolved is True
    assert field(cfg, "dpo.loss_type").resolved == ["sigmoid"]
    assert field(cfg, "dpo.beta").resolved == 0.1
    assert field(cfg, "dpo.sync_ref_model").resolved is False
    assert "2B" in field(cfg, "dpo.batch_layout").resolved
    assert "dpo.precompute_batch_size" not in {r.field for r in cfg.resolutions}
    unused, report2 = resolve(
        request(**dpo, dpo={"precompute_batch_size": 4}), hybrid_inventory(), None
    )
    assert unused is not None
    assert "dpo.precompute_batch_size" in {r.field for r in report2.not_effective}
    sft, _ = resolve(
        request(training__objective="sft", training__lora={"dropout": 0.05}),
        hybrid_inventory(),
        None,
    )
    assert sft is not None and sft.lora is not None and sft.lora.dropout == 0.05


@pytest.mark.parametrize(
    ("changes", "mode"),
    [
        ({}, "none"),
        ({"grpo__beta": 0.04}, "frozen_base_switch"),
        (
            {
                "grpo__beta": 0.04,
                "training__strategy": "full",
                "training__quantization": {"enabled": False},
            },
            "standalone_model",
        ),
    ],
)
def test_grpo_records_reference_mode_and_rollout_precision(adapter, changes, mode) -> None:
    # plan §5.4: rollout dtype/quantization recorded apart from the policy; reference mode from
    # beta and the trainer behavior.
    cfg, _ = resolve(request(**changes), hybrid_inventory(), None)
    assert cfg is not None
    assert field(cfg, "grpo.reference_mode").resolved == mode
    rollout = field(cfg, "grpo.rollout_precision").resolved
    full = "training__strategy" in changes
    assert rollout["kv_cache"] == "bfloat16" and rollout["recurrent_state"] == "float32"
    assert rollout["conv_state"] == "bfloat16"  # full: autocast; PEFT: bf16 load dtype
    assert rollout["step_logits"] == ("float32" if full else "bfloat16")
    fp32, _ = resolve(request(**changes, training__load_dtype="float32"), hybrid_inventory(), None)
    assert fp32 is not None
    low = field(fp32, "grpo.rollout_precision").resolved
    assert low["kv_cache"] == "float32"
    assert low["conv_state"] == ("bfloat16" if full else "float32")
