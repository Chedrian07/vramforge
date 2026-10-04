"""Fake estimator modules for pipeline/worker/API tests.

`FakeModules.install(monkeypatch)` replaces every module function the pipeline calls with a fake
that returns small, valid contract objects (no network, no tokenizer). Individual tests override
single attributes (e.g. `fakes.full_scan = raising(...)`) before installing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.pipeline import CachedScan
from vramforge_estimator.scan import LengthTable, ScanLimits, ScanOutcome
from vramforge_estimator.schemas import (
    AnalysisRequest,
    AnalysisResult,
    ArchitectureFacts,
    BatchPlan,
    BatchShape,
    Branch,
    BranchStats,
    ColumnMapping,
    CompatibilityReport,
    ComponentParams,
    ContextValidation,
    DataPreservation,
    DatasetColumn,
    DatasetFormat,
    DatasetInspection,
    DatasetScanResult,
    DatasetSplitInfo,
    DeviceEstimate,
    EffectiveDtypes,
    ErrorCode,
    EvidenceLevel,
    FileEntry,
    HardwareFit,
    HardwareFitResult,
    Issue,
    JobProgress,
    JobStatus,
    LengthStats,
    LinearModule,
    LoraResolved,
    MemoryEstimate,
    ModelComponent,
    ModelInventory,
    Objective,
    OptimizerResolved,
    Phase,
    PhasePeak,
    PreservationAudit,
    QuantizationResolved,
    ResolvedConfig,
    SamplerPlan,
    ScanCoverage,
    ScenarioEstimate,
    SourceManifest,
    SourceType,
    Strategy,
    SupportEntry,
    TensorInfo,
    TensorRole,
    TokenizerManifest,
    TrainingReadiness,
    WorkspaceAssumptions,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

REPO = Path(__file__).resolve().parents[3]
EXAMPLE_REQUEST = REPO / "tests" / "fixtures" / "requests" / "plan_example_grpo.json"
MODEL_SHA = "2367e865d009c13ac81713a2878291d33ab28177"
DATASET_SHA = "81aeacf06cf43b16d7278a3a01f019a496a53c51"
GiB = 1024**3


def example_request(**overrides: Any) -> AnalysisRequest:
    data = json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return AnalysisRequest.model_validate(data)


def raising(issue_or_exc: Issue | BaseException) -> Callable[..., Any]:
    def fn(*args: Any, **kwargs: Any) -> Any:
        if isinstance(issue_or_exc, Issue):
            raise EstimatorError(issue_or_exc)
        raise issue_or_exc

    return fn


def issue(code: ErrorCode, message: str = "실패했습니다.", **details: Any) -> Issue:
    return make_issue(code, message, **details)


# ---------------------------------------------------------------- contract object builders


def model_manifest() -> SourceManifest:
    return SourceManifest(
        kind="model",
        source_type=SourceType.HUGGINGFACE,
        reference="hf:XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        repo_id="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        resolved_revision=MODEL_SHA,
        files=[
            FileEntry(path="config.json", size=1000),
            FileEntry(path="tokenizer.json", size=2000),
            FileEntry(path="tokenizer_config.json", size=300),
        ],
        fingerprint="src_model",
    )


def dataset_manifest() -> SourceManifest:
    return SourceManifest(
        kind="dataset",
        source_type=SourceType.HUGGINGFACE,
        reference="hf:CyberNative/Code_Vulnerability_Security_DPO",
        repo_id="CyberNative/Code_Vulnerability_Security_DPO",
        resolved_revision=DATASET_SHA,
        fingerprint="src_dataset",
    )


def facts(**overrides: Any) -> ArchitectureFacts:
    values: dict[str, Any] = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "vocab_size": 256,
        "max_position_embeddings": 4096,
        "config_sha256": "c" * 64,
    }
    values.update(overrides)
    return ArchitectureFacts(**values)


def inventory(**fact_overrides: Any) -> ModelInventory:
    tensor = TensorInfo(
        name="model.layers.0.mlp.up_proj.weight",
        dtype="bfloat16",
        shape=[64, 64],
        numel=4096,
        nbytes=8192,
        component=ModelComponent.TEXT,
        role=TensorRole.LINEAR_WEIGHT,
        module="model.layers.0.mlp.up_proj",
        layer_index=0,
    )
    linear = LinearModule(
        name="model.layers.0.mlp.up_proj",
        kind="up_proj",
        in_features=64,
        out_features=64,
        has_bias=False,
        component=ModelComponent.TEXT,
        layer_index=0,
        dtype="bfloat16",
    )
    return ModelInventory(
        facts=facts(**fact_overrides),
        tensors=[tensor],
        linear_modules=[linear],
        params_total=4096,
        bytes_serialized_total=8192,
        by_component=[
            ComponentParams(component=ModelComponent.TEXT, params=4096, bytes_serialized=8192)
        ],
        inventory_hash="inv_hash",
    )


def tokenizer_manifest(*, chat_template: bool = True) -> TokenizerManifest:
    return TokenizerManifest(
        tokenizer_class="Qwen3_5Tokenizer",
        vocab_size=250,
        bos_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        model_max_length=4096,
        chat_template_present=chat_template,
        chat_template_source="chat_template.jinja" if chat_template else "none",
        chat_template_sha256="t" * 64 if chat_template else None,
        files_sha256={"tokenizer.json": "a" * 64, "tokenizer_config.json": "b" * 64},
        fingerprint="tok_fp",
    )


def dataset_inspection(**overrides: Any) -> DatasetInspection:
    values: dict[str, Any] = {
        "manifest": dataset_manifest(),
        "configs": ["default"],
        "selected_config": "default",
        "splits": [DatasetSplitInfo(name="train", num_rows=4)],
        "selected_split": "train",
        "split_auto_selected": False,
        "columns": [
            DatasetColumn(name=n, dtype="string", kind="string")
            for n in ("system", "question", "chosen", "rejected", "lang")
        ],
        "detected_format": DatasetFormat.PREFERENCE,
        "mapping_candidates": [
            ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
        ],
        "suggested_mapping": ColumnMapping(
            system="system", prompt="question", chosen="chosen", rejected="rejected"
        ),
        "mapping_ambiguous": False,
    }
    values.update(overrides)
    return DatasetInspection(**values)


def resolved_config(objective: Objective = Objective.GRPO, **overrides: Any) -> ResolvedConfig:
    values: dict[str, Any] = {
        "profile_id": "analytic/qwen3_5_hybrid-trl-1.14.1",
        "profile_version": "1",
        "environment_id": "cuda-trl-1.14.1",
        "dependency_lock_digest": "lock_abc",
        "architecture_adapter": "qwen3_5_hybrid",
        "trainer_adapter": f"trl-1.14.1-{objective.value}",
        "preprocessing_adapter": f"trl-1.14.1-{objective.value}",
        "objective": objective,
        "strategy": Strategy.QLORA,
        "loading_scope": "full_checkpoint",
        "load_dtype": "bfloat16",
        "effective_dtypes": EffectiveDtypes(
            weights_nonquantized="bfloat16",
            compute="bfloat16",
            adapter="bfloat16",
            gradient="bfloat16",
            optimizer_state="bfloat16",
            logits="float32",
            loss="float32",
            kv_cache="bfloat16",
            recurrent_state="float32",
        ),
        "quantization": QuantizationResolved(
            enabled=True,
            method="bnb_nf4",
            double_quant=True,
            blocksize=64,
            nested_blocksize=256,
            quant_storage_dtype="uint8",
            compute_dtype="bfloat16",
            skip_module_patterns=["lm_head"],
        ),
        "lora": LoraResolved(
            r=16,
            alpha=32,
            dropout=0.0,
            target_module_patterns=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules=["model.layers.0.self_attn.q_proj"],
        ),
        "optimizer": OptimizerResolved(
            name="adamw_torch_fused", states_per_param=2, state_dtype="bfloat16", fused=True
        ),
        "microbatch": 1,
        "accumulation": 4,
        "gradient_checkpointing": True,
        "checkpointing_granularity": "per_decoder_layer",
        "loss_path": "chunked_nll" if objective is Objective.SFT else "standard",
        "workspace": WorkspaceAssumptions(
            cuda_context_bytes=(300 * 1024**2, 600 * 1024**2),
            library_workspace_bytes=(0, 256 * 1024**2),
            allocator_slack_fraction=(0.0, 0.1),
        ),
    }
    values.update(overrides)
    return ResolvedConfig(**values)


def compat_report(
    *,
    readiness: TrainingReadiness = TrainingReadiness.READY,
    grade: EvidenceLevel | None = EvidenceLevel.ANALYTIC,
    adapter: str | None = "qwen3_5_hybrid",
    blockers: list[Issue] | None = None,
    warnings: list[Issue] | None = None,
) -> CompatibilityReport:
    return CompatibilityReport(
        profile_id="analytic/qwen3_5_hybrid-trl-1.14.1",
        architecture_adapter=adapter,
        support_grade=grade,
        readiness=readiness,
        support=[
            SupportEntry(
                objective=Objective.GRPO,
                strategy=Strategy.QLORA,
                grade=grade,
                readiness=readiness,
            )
        ],
        blockers=blockers or [],
        warnings=warnings or [],
    )


def scan_result(
    objective: Objective = Objective.GRPO,
    coverage: ScanCoverage = ScanCoverage.COMPLETE,
    *,
    preprocess_key: str = "pre_x",
    rows_failed: int = 0,
) -> DatasetScanResult:
    stats = LengthStats(count=4, min=10, max=272, max_row_id="train:2", total_tokens=500)
    return DatasetScanResult(
        coverage=coverage,
        objective=objective,
        config="default",
        split="train",
        mapping_applied=ColumnMapping(
            system="system", prompt="question", chosen="chosen", rejected="rejected"
        ),
        transformation_note="GRPO: prompt만 사용합니다.",
        rows_expected=4,
        rows_seen=4,
        rows_ok=4 - rows_failed,
        rows_failed=rows_failed,
        branches=[BranchStats(branch=Branch.PROMPT, stats=stats)],
        preprocess_key=preprocess_key,
        artifact_id="lengths",
        tokenizer_fingerprint="tok_fp",
        template_fingerprint="t" * 64,
        preprocessing_adapter="trl-1.14.1-grpo",
        preprocessing_adapter_version="1",
    )


def batch_plan(objective: Objective = Objective.GRPO) -> BatchPlan:
    shape = BatchShape(
        name="worst_case",
        objective=objective,
        rows_per_microbatch=1,
        sequences_per_forward=1,
        padded_length=272,
        token_slots=272,
        logits_positions=272,
        description="가장 긴 row",
    )
    return BatchPlan(
        objective=objective,
        unit="completions" if objective is Objective.GRPO else "samples",
        microbatch=1,
        accumulation=4,
        effective_batch=4,
        sampler=SamplerPlan(kind="repeat_sampler", seed=42, covers_all_rows=True),
        worst_case=shape,
        scenarios=[shape],
        batch_key="bat_x",
    )


def scenario(
    scenario_id: str,
    *,
    high: int | None = 10 * GiB,
    fit: HardwareFit = HardwareFit.NOT_EVALUATED,
) -> ScenarioEstimate:
    device = DeviceEstimate(
        device="cuda:0",
        phases=[
            PhasePeak(phase=Phase.POLICY_FORWARD_BACKWARD, included=True, bytes_high=high),
            PhasePeak(phase=Phase.EVALUATION, included=False, excluded_reason="범위 밖"),
        ],
        timepoints=[],
        peak_phase=Phase.POLICY_FORWARD_BACKWARD,
        known_floor_bytes=7 * GiB,
        scenario_low_bytes=None if high is None else high - GiB,
        scenario_high_bytes=high,
    )
    return ScenarioEstimate(
        scenario_id=scenario_id,
        label=scenario_id,
        batch_shape=batch_plan().worst_case,
        devices=[device],
        hardware_fit=HardwareFitResult(
            status=fit,
            reason="not_evaluated" if fit is HardwareFit.NOT_EVALUATED else "fits_with_margin",
            message="용량만 계산" if fit is HardwareFit.NOT_EVALUATED else "예상 적합",
        ),
    )


def memory_estimate(
    scenarios: list[ScenarioEstimate] | None = None, primary: str | None = "budget_1024"
) -> MemoryEstimate:
    return MemoryEstimate(
        evidence_level=EvidenceLevel.ANALYTIC,
        scenarios=scenarios or [scenario("budget_1024")],
        primary_scenario_id=primary,
    )


def context_validation(status: str = "ok", limit: int = 4096) -> ContextValidation:
    return ContextValidation(
        model_declared_max=limit,
        effective_limit=limit,
        limit_source="config.max_position_embeddings",
        max_observed_length=272,
        status=status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- context


@dataclass
class FakeContext:
    artifact_dir: Path
    analysis_id: str = "a" * 32
    access: SourceAccess = field(default_factory=SourceAccess)
    limits: ScanLimits = field(default_factory=ScanLimits)
    cancel_after_stage: JobStatus | None = None
    cancel_now: bool = False
    stages: list[JobStatus] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    reports: list[JobProgress] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None

    def set_stage(self, stage: JobStatus) -> None:
        self.stages.append(stage)
        if stage is self.cancel_after_stage:
            self.cancel_now = True

    def warn(self, issue: Issue) -> None:
        self.warnings.append(issue)

    def report(self, progress: JobProgress, partial: dict[str, Any] | None = None) -> None:
        self.reports.append(progress)

    def cancelled(self) -> bool:
        return self.cancel_now

    def load_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        self.checkpoint = data


@dataclass
class CachingContext(FakeContext):
    cache: dict[str, CachedScan] = field(default_factory=dict)
    partials: list[AnalysisResult] = field(default_factory=list)

    def lookup_scan_cache(self, preprocess_key: str) -> CachedScan | None:
        return self.cache.get(preprocess_key)

    def store_scan_cache(self, preprocess_key: str, scan: CachedScan) -> None:
        self.cache[preprocess_key] = scan

    def publish_partial(self, result: AnalysisResult) -> None:
        self.partials.append(result)


# ---------------------------------------------------------------- module fakes


@dataclass
class FakeModules:
    """Callables installed in place of the estimator modules (override before `install`)."""

    calls: list[str] = field(default_factory=list)
    resolve_model: Callable[..., Any] | None = None
    resolve_dataset: Callable[..., Any] | None = None
    inspect_model: Callable[..., Any] | None = None
    load_tokenizer: Callable[..., Any] | None = None
    inspect_dataset: Callable[..., Any] | None = None
    open_rows: Callable[..., Any] | None = None
    get_adapter: Callable[..., Any] | None = None
    full_scan: Callable[..., Any] | None = None
    load_lengths: Callable[..., Any] | None = None
    validate_context: Callable[..., Any] | None = None
    audit_preservation: Callable[..., Any] | None = None
    resolve: Callable[..., Any] | None = None
    plan_batches: Callable[..., Any] | None = None
    estimate_memory: Callable[..., Any] | None = None
    estimate_host_ram: Callable[..., Any] | None = None
    estimate_disk: Callable[..., Any] | None = None
    estimate_analysis_ram: Callable[..., Any] | None = None
    scan_coverage: ScanCoverage = ScanCoverage.COMPLETE
    chat_template: bool = True
    objective: Objective = Objective.GRPO
    last_estimate_kwargs: dict[str, Any] = field(default_factory=dict)

    # defaults ------------------------------------------------------------------------
    def _resolve_model(self, ref: Any, access: Any) -> ResolvedSource:
        return ResolvedSource(
            kind="model", manifest=model_manifest(), repo_id="m/m", revision=MODEL_SHA
        )

    def _resolve_dataset(self, ref: Any, access: Any) -> ResolvedSource:
        return ResolvedSource(
            kind="dataset", manifest=dataset_manifest(), repo_id="d/d", revision=DATASET_SHA
        )

    def _inspect_model(self, source: Any, access: Any) -> ModelInventory:
        return inventory()

    def _load_tokenizer(self, source: Any, access: Any) -> TokenizerHandle:
        return TokenizerHandle(
            tokenizer=object(), manifest=tokenizer_manifest(chat_template=self.chat_template)
        )

    def _inspect_dataset(self, source: Any, ref: Any, access: Any, objective: Any = None):
        return dataset_inspection()

    def _open_rows(self, source: Any, *, config: Any, split: str, access: Any) -> object:
        return object()

    def _get_adapter(self, objective: Objective, tokenizer: Any, mapping: Any, **kwargs: Any):
        class _Adapter:
            name = f"trl-1.14.1-{objective.value}"
            version = "1"

        adapter = _Adapter()
        adapter.objective = objective  # type: ignore[attr-defined]
        return adapter

    def _full_scan(self, stream: Any, adapter: Any, ctx: Any, **kwargs: Any) -> ScanOutcome:
        lengths = ctx.artifact_dir / "lengths"
        lengths.mkdir(parents=True, exist_ok=True)
        (lengths / "manifest.json").write_text("{}", encoding="utf-8")
        (lengths / "part-00000.parquet").write_bytes(b"PAR1fakePAR1")
        return ScanOutcome(
            result=scan_result(
                kwargs["objective"], self.scan_coverage, preprocess_key=kwargs["preprocess_key"]
            ),
            artifact_path=lengths,
        )

    def _load_lengths(self, path: Path) -> LengthTable:
        return LengthTable(row_ids=["train:0"], prompt_tokens=[272])

    def _validate_context(self, scan: Any, **kwargs: Any) -> ContextValidation:
        return context_validation()

    def _audit(self, scan: DatasetScanResult, **kwargs: Any) -> PreservationAudit:
        if scan.coverage is not ScanCoverage.COMPLETE:
            return PreservationAudit(
                status=DataPreservation.UNKNOWN,
                violations=[make_issue(ErrorCode.SCAN_PARTIAL, "스캔이 끝나지 않았습니다.")],
            )
        if kwargs.get("batch_plan") is None:
            return PreservationAudit(status=DataPreservation.PENDING)
        return PreservationAudit(status=DataPreservation.VERIFIED)

    def _resolve(self, request: AnalysisRequest, inv: Any, tok: Any):
        return resolved_config(request.training.objective), compat_report()

    def _plan_batches(self, lengths: Any, resolved: ResolvedConfig, *, seed: int) -> BatchPlan:
        return batch_plan(resolved.objective)

    def _estimate_memory(self, inv: Any, cfg: Any, plan: Any, **kwargs: Any) -> MemoryEstimate:
        self.last_estimate_kwargs = {"cfg": cfg, **kwargs}
        return memory_estimate()

    def _not_implemented(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def install(self, monkeypatch: Any) -> FakeModules:
        from vramforge_estimator import (
            batching,
            compatibility,
            inspection,
            memory,
            preprocessing,
            scan,
            sources,
        )

        table = {
            (sources, "resolve_model"): self.resolve_model or self._resolve_model,
            (sources, "resolve_dataset"): self.resolve_dataset or self._resolve_dataset,
            (inspection, "inspect_model"): self.inspect_model or self._inspect_model,
            (inspection, "load_tokenizer"): self.load_tokenizer or self._load_tokenizer,
            (inspection, "inspect_dataset"): self.inspect_dataset or self._inspect_dataset,
            (inspection, "open_rows"): self.open_rows or self._open_rows,
            (preprocessing, "get_adapter"): self.get_adapter or self._get_adapter,
            (scan, "full_scan"): self.full_scan or self._full_scan,
            (scan, "load_lengths"): self.load_lengths or self._load_lengths,
            (scan, "validate_context"): self.validate_context or self._validate_context,
            (scan, "audit_preservation"): self.audit_preservation or self._audit,
            (compatibility, "resolve"): self.resolve or self._resolve,
            (batching, "plan_batches"): self.plan_batches or self._plan_batches,
            (memory, "estimate_memory"): self.estimate_memory or self._estimate_memory,
            (memory, "estimate_host_ram"): self.estimate_host_ram or self._not_implemented,
            (memory, "estimate_disk"): self.estimate_disk or self._not_implemented,
            (memory, "estimate_analysis_ram"): self.estimate_analysis_ram or self._not_implemented,
        }
        for (module, name), fn in table.items():
            monkeypatch.setattr(module, name, self._recording(name, fn))
        return self

    def _recording(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return fn(*args, **kwargs)

        return wrapper


__all__ = [
    "CachingContext",
    "FakeContext",
    "FakeModules",
    "batch_plan",
    "compat_report",
    "context_validation",
    "dataset_inspection",
    "example_request",
    "inventory",
    "issue",
    "memory_estimate",
    "raising",
    "resolved_config",
    "scan_result",
    "scenario",
    "tokenizer_manifest",
]
