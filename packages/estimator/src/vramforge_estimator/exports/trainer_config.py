"""`trainer-config.yaml` for the pinned TRL 1.14.1 environment (plan.md §12.4).

Produced only for a `ready` result. The `trl.args` section maps 1:1 to
`SFTConfig/DPOConfig/GRPOConfig(**args)` and contains ONLY fields that exist in TRL 1.14.1
(verified by the parity test against the real classes):

- SFT/DPO: `max_length: null` (TRL's default 1024 truncates and silently drops rows) and
  `packing: false` (docs/research/trl-sft-dpo.md §10, B). Removed fields such as
  `max_prompt_length` are never emitted: TRL scripts silently ignore unknown YAML keys and the
  `trl` CLI fails on them (§10.3).
- GRPO: no `max_prompt_length` (the field does not exist), an explicit finite
  `max_completion_length`, `generation_batch_size` without `steps_per_generation` (mutually
  exclusive) (docs/research/trl-grpo.md R5).
- `model_init_kwargs.dtype` is pinned because TRL loads string model ids in float32 otherwise.

The model, quantization (`transformers.BitsAndBytesConfig`), PEFT (`peft.LoraConfig`), dataset
(mapping transform, raw columns dropped) and processing-class sections describe the launcher
inputs; raw data, tokens, absolute paths and private URLs are never included. A result whose
pipeline halted or whose memory estimate is missing is never ready.
"""

from __future__ import annotations

from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.pipeline import halting_issue
from vramforge_estimator.schemas import (
    AnalysisResult,
    DataPreservation,
    DatasetFormat,
    ErrorCode,
    Objective,
    ReferenceStrategy,
    RewardKind,
    ScanCoverage,
    Stage,
    TrainingReadiness,
)

TRL_VERSION = "1.14.1"
CONFIG_CLASS = {
    Objective.SFT: "SFTConfig",
    Objective.DPO: "DPOConfig",
    Objective.GRPO: "GRPOConfig",
}
TRAINER_CLASS = {
    Objective.SFT: "SFTTrainer",
    Objective.DPO: "DPOTrainer",
    Objective.GRPO: "GRPOTrainer",
}

# Fields emitted per config class (all verified to exist in TRL 1.14.1 by the parity test).
COMMON_FIELDS = frozenset(
    {
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "bf16",
        "fp16",
        "optim",
        "dataloader_drop_last",
        "max_steps",
        "seed",
        "model_init_kwargs",
        "use_liger_kernel",
        "report_to",
        "pad_to_multiple_of",
    }
)
SFT_FIELDS = COMMON_FIELDS | {
    "max_length",
    "packing",
    "eval_packing",
    "padding_free",
    "loss_type",
    "assistant_only_loss",
}
DPO_FIELDS = COMMON_FIELDS | {
    "max_length",
    "padding_free",
    "loss_type",
    "beta",
    "precompute_ref_log_probs",
    "precompute_ref_batch_size",
    "sync_ref_model",
    "disable_dropout",
}
GRPO_FIELDS = COMMON_FIELDS | {
    "num_generations",
    "generation_batch_size",
    "max_completion_length",
    "beta",
    "num_iterations",
    "use_vllm",
    "use_transformers_continuous_batching",
    "cache_implementation",
    "shuffle_dataset",
    "chat_template_kwargs",
}
FIELDS = {Objective.SFT: SFT_FIELDS, Objective.DPO: DPO_FIELDS, Objective.GRPO: GRPO_FIELDS}
# Never emitted: removed in TRL 1.14.1 or truncation knobs (docs/research/trl-sft-dpo.md §6).
FORBIDDEN_FIELDS = frozenset(
    {
        "max_prompt_length",
        "use_logits_to_keep",
        "model_adapter_name",
        "ref_adapter_name",
        "force_use_ref_model",
        "label_pad_token_id",
        "padding_value",
        "rpo_alpha",
        "reference_free",
        "max_seq_length",
        "dataset_batch_size",
        "num_of_sequences",
        "chars_per_token",
        "truncation_mode",
        "steps_per_generation",
    }
)
OPTIMIZERS = {
    "adamw_torch": "adamw_torch",
    "adamw_torch_fused": "adamw_torch_fused",
    "adamw_8bit": "adamw_bnb_8bit",
    "adamw_bnb_8bit": "adamw_bnb_8bit",
    "paged_adamw_8bit": "paged_adamw_8bit",
}
# Resolved SFT loss path (profile naming) -> SFTConfig.loss_type.
SFT_LOSS_TYPES = {
    "trl_chunked_nll": "chunked_nll",
    "chunked_nll": "chunked_nll",
    "hf_ce": "nll",
    "nll": "nll",
    "dft": "dft",
}
OUTPUT_COLUMNS = {
    Objective.SFT: ["prompt", "completion"],
    Objective.DPO: ["prompt", "chosen", "rejected"],
    Objective.GRPO: ["prompt"],
}


def _not_ready(result: AnalysisResult, reasons: list[str], code: ErrorCode) -> EstimatorError:
    readiness = result.status.training_readiness
    state = readiness.value if readiness else "unknown"
    return EstimatorError(
        make_issue(
            code,
            "실행용 trainer 설정은 학습 준비 상태가 ready일 때만 제공합니다 "
            f"(현재: {state}). " + " ".join(reasons),
            stage=Stage.EXPORT,
            readiness=state,
            reasons=reasons,
        )
    )


def check_ready(result: AnalysisResult) -> None:
    """Raise `EstimatorError` with the reason unless the result can be executed as-is."""
    stopped = halting_issue(result)
    if stopped is not None:
        # A PARTIAL/FAILED/CANCELLED run never verified the setup it would describe.
        raise _not_ready(result, [stopped.user_message], stopped.code)
    report = result.compatibility_report
    readiness = result.status.training_readiness
    if readiness is not TrainingReadiness.READY:
        codes = [i.code for i in (report.blockers if report else [])] + [
            i.code for i in (report.warnings if report else [])
        ]
        code = codes[0] if codes else ErrorCode.UNSUPPORTED_BACKEND_COMBINATION
        reasons = [i.user_message for i in (report.blockers if report else [])][:3]
        if not reasons and report is not None:
            reasons = [i.user_message for i in report.warnings][:3]
        if not reasons:
            reasons = ["학습 준비 상태를 확인하세요(데이터 보존, 호환성, 미지정 항목)."]
        raise _not_ready(result, reasons, code)
    resolved = result.resolved_config
    scan = result.dataset_scan
    problems: list[str] = []
    if resolved is None:
        problems.append("적용 설정(resolved config)이 없습니다.")
    if scan is None or scan.coverage is not ScanCoverage.COMPLETE:
        problems.append("데이터셋 전체 분석이 완료되지 않았습니다.")
    elif scan.mapping_applied is None:
        problems.append("적용한 컬럼 매핑이 기록되지 않았습니다.")
    audit = result.preservation_audit
    if audit is None or audit.status is not DataPreservation.VERIFIED:
        problems.append("데이터 보존 검사가 검증 상태가 아닙니다.")
    if result.source_manifests.model is None or result.source_manifests.dataset is None:
        problems.append("모델·데이터셋 revision이 고정되지 않았습니다.")
    if result.memory is None:
        problems.append("메모리 산정까지 끝난 결과가 아닙니다(진행 중이거나 중단된 분석).")
    if resolved is not None and resolved.objective is Objective.GRPO:
        grpo = resolved.grpo
        if grpo is None:
            problems.append("GRPO 설정이 해석되지 않았습니다.")
        else:
            if not grpo.budget_explicit or len(grpo.completion_budgets) != 1:
                raise _not_ready(
                    result,
                    ["completion budget(max_completion_length)을 하나로 지정해야 합니다."],
                    ErrorCode.GRPO_BUDGET_UNSPECIFIED,
                )
            if grpo.reward_kind is RewardKind.UNSPECIFIED:
                raise _not_ready(
                    result,
                    ["GRPO reward가 지정되지 않았습니다."],
                    ErrorCode.GRPO_REWARD_UNSPECIFIED,
                )
    if resolved is not None and resolved.objective is Objective.DPO and resolved.dpo is None:
        problems.append("DPO 설정이 해석되지 않았습니다.")
    if (
        resolved is not None
        and resolved.objective is Objective.SFT
        and resolved.loss_path not in SFT_LOSS_TYPES
    ):
        problems.append("적용한 loss 경로를 TRL SFTConfig.loss_type으로 옮길 수 없습니다.")
    if problems:
        raise _not_ready(result, problems, ErrorCode.REQUESTED_OPTION_NOT_EFFECTIVE)


def _precision(compute: str) -> tuple[bool, bool]:
    return compute == "bfloat16", compute == "float16"


def build_trainer_config(result: AnalysisResult) -> dict[str, Any]:
    check_ready(result)
    resolved = result.resolved_config
    scan = result.dataset_scan
    model = result.source_manifests.model
    dataset = result.source_manifests.dataset
    assert resolved is not None and scan is not None and model is not None and dataset is not None
    assert scan.mapping_applied is not None
    request = result.requested_config
    objective = resolved.objective
    model_id = model.repo_id or model.reference
    bf16, fp16 = _precision(resolved.effective_dtypes.compute)
    model_init_kwargs = {
        "dtype": resolved.load_dtype,
        "revision": model.resolved_revision,
        "trust_remote_code": False,
    }

    args: dict[str, Any] = {
        "output_dir": f"outputs/vramforge-{result.analysis_id[:8]}",
        "per_device_train_batch_size": resolved.microbatch,
        "gradient_accumulation_steps": resolved.accumulation,
        "gradient_checkpointing": resolved.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
        if resolved.gradient_checkpointing
        else None,
        "bf16": bf16,
        "fp16": fp16,
        "optim": OPTIMIZERS.get(resolved.optimizer.name, resolved.optimizer.name),
        "dataloader_drop_last": False,
        "max_steps": -1,
        "seed": request.training.seed,
        "model_init_kwargs": model_init_kwargs,
        "use_liger_kernel": False,
        "report_to": "none",
        "pad_to_multiple_of": resolved.pad_to_multiple_of,
    }
    if objective is Objective.SFT:
        args.update(
            {
                "max_length": None,
                "packing": False,
                "eval_packing": False,
                "padding_free": False,
                "assistant_only_loss": False,
            }
        )
        args["loss_type"] = SFT_LOSS_TYPES[resolved.loss_path]
    elif objective is Objective.DPO:
        dpo = resolved.dpo
        assert dpo is not None
        args.update(
            {
                "max_length": None,
                "padding_free": False,
                "loss_type": [dpo.loss_type],
                "beta": dpo.beta,
                "precompute_ref_log_probs": dpo.reference_strategy
                is ReferenceStrategy.PRECOMPUTED_LOG_PROBS,
                "precompute_ref_batch_size": dpo.precompute_batch_size,
                "sync_ref_model": dpo.sync_ref_model,
                "disable_dropout": True,
            }
        )
    else:
        grpo = resolved.grpo
        assert grpo is not None
        args.update(
            {
                "per_device_train_batch_size": grpo.update_microbatch,
                "gradient_accumulation_steps": grpo.accumulation,
                "num_generations": grpo.num_generations,
                "generation_batch_size": grpo.generation_batch_size,
                "max_completion_length": grpo.completion_budgets[0],
                "beta": grpo.beta,
                "num_iterations": grpo.num_iterations,
                "use_vllm": False,
                "use_transformers_continuous_batching": False,
                "cache_implementation": None,
                "shuffle_dataset": True,
                "chat_template_kwargs": dict(resolved.template_kwargs) or None,
            }
        )
    allowed = FIELDS[objective]
    unexpected = (set(args) - allowed) | (set(args) & FORBIDDEN_FIELDS)
    if unexpected:  # pragma: no cover - guarded by tests; never export unknown keys
        raise AssertionError(f"unexpected TRL fields: {sorted(unexpected)}")

    quant = resolved.quantization
    quantization = None
    if quant.enabled:
        quantization = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4" if quant.method == "bnb_fp4" else "nf4",
            "bnb_4bit_use_double_quant": quant.double_quant,
            "bnb_4bit_compute_dtype": quant.compute_dtype or resolved.effective_dtypes.compute,
            "bnb_4bit_quant_storage": quant.quant_storage_dtype or "uint8",
            "llm_int8_skip_modules": list(quant.skip_module_patterns) or None,
        }
    lora = resolved.lora
    peft = None
    if lora is not None:
        peft = {
            "r": lora.r,
            "lora_alpha": lora.alpha,
            "lora_dropout": lora.dropout,
            "target_modules": list(lora.target_module_patterns),
            "exclude_modules": list(lora.exclude_modules) or None,
            "modules_to_save": list(lora.modules_to_save) or None,
            "bias": lora.bias,
            "rank_pattern": dict(lora.rank_pattern),
            "use_dora": lora.use_dora,
            "use_rslora": lora.use_rslora,
            "task_type": "CAUSAL_LM",
        }
    mapping = scan.mapping_applied
    roles = {
        role: getattr(mapping, role)
        for role in ("system", "prompt", "chosen", "rejected", "completion", "messages", "text")
        if getattr(mapping, role)
    }
    output_columns = OUTPUT_COLUMNS[objective]
    if mapping.format is DatasetFormat.MESSAGES or mapping.messages:
        output_columns = ["messages"]
    elif mapping.format is DatasetFormat.TEXT or mapping.text:
        output_columns = ["text"]
    dataset_section = {
        "id": dataset.repo_id or dataset.reference,
        "revision": dataset.resolved_revision,
        "config": scan.config,
        "split": scan.split,
        "mapping": roles,
        "empty_system_policy": mapping.empty_system_policy.value,
        "transform": scan.transformation_note,
        "output_columns": output_columns,
        "remove_original_columns": True,
    }
    processing_class = {
        "type": "tokenizer",
        "class": "AutoTokenizer",
        "from_pretrained": {
            "pretrained_model_name_or_path": model_id,
            "revision": model.resolved_revision,
            **({"padding_side": "left"} if objective is Objective.GRPO else {}),
        },
    }
    config: dict[str, Any] = {
        "kind": "vramforge.trainer-config",
        "schema_version": result.schema_version,
        "analysis_id": result.analysis_id,
        "analysis_fingerprint": result.analysis_fingerprint,
        "profile_id": result.profile_id,
        "dependency_lock_digest": result.dependency_lock_digest,
        "trainer": {
            "library": "trl",
            "version": TRL_VERSION,
            "class": TRAINER_CLASS[objective],
            "config_class": CONFIG_CLASS[objective],
        },
        "model": {
            "id": model_id,
            "revision": model.resolved_revision,
            "model_init_kwargs": model_init_kwargs,
        },
        "quantization": quantization,
        "peft": peft,
        "dataset": dataset_section,
        "processing_class": processing_class,
        "trl": {"config_class": CONFIG_CLASS[objective], "args": args},
    }
    if objective is Objective.GRPO:
        grpo = resolved.grpo
        assert grpo is not None
        config["reward"] = {
            "kind": grpo.reward_kind.value,
            "model": grpo.reward_model_reference,
            "note": (
                "reward_funcs는 GRPOTrainer에 직접 전달해야 합니다(이 파일에는 코드가 없습니다)."
            ),
        }
    return config


HEADER = (
    f"# VRAMForge trainer config for TRL {TRL_VERSION} (training readiness: ready)\n"
    "# trl.args는 <config_class>(**args)에 그대로 전달합니다. 이 파일에 없는 TRL 키를 추가하지\n"
    "# 마세요(TRL 스크립트는 모르는 키를 조용히 무시합니다). max_length: null은 데이터를 자르지\n"
    "# 않는다는 뜻입니다.\n"
    "# Pass trl.args to the config class as-is; quantization -> transformers.BitsAndBytesConfig,\n"
    "# peft -> peft.LoraConfig, processing_class -> AutoTokenizer.from_pretrained(...).\n"
)


__all__ = [
    "COMMON_FIELDS",
    "DPO_FIELDS",
    "FIELDS",
    "FORBIDDEN_FIELDS",
    "GRPO_FIELDS",
    "HEADER",
    "SFT_FIELDS",
    "build_trainer_config",
    "check_ready",
]
