// Form values <-> AnalysisRequest. The request is fully explicit and validated by the strict
// mirror before it is sent; nothing outside the contract is ever included.
import type { AnalysisRequest, GpuPreset } from "@/lib/api/types";
import { bytesToGibInput, gibInputToBytes } from "@/lib/format/bytes";

import { analysisRequestSchema, type ExplicitAnalysisRequest } from "./request-schema";
import { datasetSourceType, modelSourceType } from "./references";
import {
  DEFAULT_BUDGET_CANDIDATES,
  DEFAULT_FORM_VALUES,
  parseBudgetList,
  parseList,
  parseRankPattern,
  type FormValues,
} from "./values";

const orNull = (text: string): string | null => (text.trim() === "" ? null : text.trim());
const intOrNull = (text: string): number | null => (text.trim() === "" ? null : Number(text.trim()));

function mappingFrom(values: FormValues): ExplicitAnalysisRequest["dataset"]["mapping"] {
  if (!values.mappingEnabled) return null;
  return {
    format: values.mappingFormat,
    system: orNull(values.mapSystem),
    prompt: orNull(values.mapPrompt),
    chosen: orNull(values.mapChosen),
    rejected: orNull(values.mapRejected),
    completion: orNull(values.mapCompletion),
    messages: orNull(values.mapMessages),
    text: orNull(values.mapText),
    empty_system_policy: values.emptySystemPolicy,
  };
}

function hardwareFrom(values: FormValues, presets: readonly GpuPreset[]): ExplicitAnalysisRequest["hardware"] {
  const external = gibInputToBytes(values.externalReservedGiB) ?? 0;
  const base = { external_reserved_bytes: external, num_gpus: 1 };
  if (values.hardwareMode === "gpu_preset") {
    const preset = presets.find((p) => p.id === values.gpuPresetId);
    return {
      mode: "gpu_preset",
      gpu_preset: values.gpuPresetId || null,
      device_total_bytes: preset?.total_bytes ?? null,
      usable_bytes: gibInputToBytes(values.hardwareUsableGiB),
      ...base,
    };
  }
  if (values.hardwareMode === "custom") {
    return {
      mode: "custom",
      gpu_preset: null,
      device_total_bytes: gibInputToBytes(values.hardwareTotalGiB),
      usable_bytes: gibInputToBytes(values.hardwareUsableGiB),
      ...base,
    };
  }
  return { mode: "capacity_only", gpu_preset: null, device_total_bytes: null, usable_bytes: null, ...base };
}

/** Builds the explicit request object (not yet validated). */
export function toAnalysisRequest(
  values: FormValues,
  presets: readonly GpuPreset[] = [],
): ExplicitAnalysisRequest {
  const isGbs = values.grpoGenerationUnit === "generation_batch_size";
  const candidates = parseBudgetList(values.grpoBudgetCandidates) ??
    parseBudgetList(DEFAULT_BUDGET_CANDIDATES) ?? [1024, 2048, 4096, 8192];
  return {
    schema_version: "1.0",
    model: {
      source_type: modelSourceType(values.modelReference),
      reference: values.modelReference.trim(),
      revision: orNull(values.modelRevision),
      loading_scope: values.loadingScope,
    },
    dataset: {
      source_type: datasetSourceType(values.datasetReference, values.datasetUploadRef),
      reference: values.datasetReference.trim(),
      revision: orNull(values.datasetRevision),
      config: orNull(values.datasetConfig),
      split: orNull(values.datasetSplit),
      eval_split: orNull(values.datasetEvalSplit),
      scan_mode: "full",
      sample_rows: null,
      mapping: mappingFrom(values),
    },
    training: {
      objective: values.objective,
      strategy: values.strategy,
      quantization: {
        enabled: values.load4bit,
        format: values.quantFormat,
        double_quant: values.doubleQuant,
        compute_dtype: values.computeDtype,
      },
      lora: {
        r: Number(values.loraR),
        alpha: Number(values.loraAlpha),
        dropout: Number(values.loraDropout),
        target_modules:
          values.loraTargetMode === "custom" ? parseList(values.loraTargetModules) : values.loraTargetMode,
        exclude_modules: parseList(values.loraExcludeModules),
        modules_to_save: parseList(values.loraModulesToSave),
        bias: values.loraBias,
        rank_pattern: parseRankPattern(values.loraRankPattern) ?? {},
        use_dora: values.loraUseDora,
        use_rslora: values.loraUseRslora,
      },
      microbatch_per_device: intOrNull(values.microbatch),
      gradient_accumulation_steps: intOrNull(values.accumulation),
      gradient_checkpointing: values.gradientCheckpointing,
      precision: values.precision,
      optimizer: values.optimizer,
      load_dtype: values.loadDtype,
      attention_backend: values.attentionBackend,
      linear_attention_kernel: values.linearAttentionKernel,
      loss_kernel: values.lossKernel,
      // Strict no-truncation mode: packing, offload and compile are not offered (plan §5.2).
      packing: false,
      pad_to_multiple_of: intOrNull(values.padToMultipleOf),
      offload: { parameters: false, optimizer: false, activations: false },
      compile: false,
      num_devices: 1,
      data_policy: "strict_no_truncation",
      backend_profile: values.backendProfile.trim() || "auto",
      template: {
        enable_thinking: values.enableThinking === "default" ? null : values.enableThinking === "on",
      },
      seed: Number(values.seed),
    },
    dpo: {
      reference_strategy: values.dpoReferenceStrategy,
      reference_model:
        values.dpoReferenceStrategy === "standalone_model" ? orNull(values.dpoReferenceModel) : null,
      beta: Number(values.dpoBeta),
      loss_type: values.dpoLossType.trim(),
      precompute_batch_size:
        values.dpoReferenceStrategy === "precomputed_log_probs" ? intOrNull(values.dpoPrecomputeBatchSize) : null,
      sync_ref_model: values.dpoSyncRefModel,
    },
    grpo: {
      num_generations: Number(values.grpoNumGenerations),
      generation_batch_size: isGbs ? intOrNull(values.grpoGenerationBatchSize) : null,
      steps_per_generation: isGbs ? null : intOrNull(values.grpoStepsPerGeneration),
      num_iterations: Number(values.grpoNumIterations),
      completion_budget: values.grpoBudgetMode === "explicit" ? intOrNull(values.grpoCompletionBudget) : null,
      completion_budget_candidates: candidates,
      max_live_sequences: null,
      beta: Number(values.grpoBeta),
      reward: {
        kind: values.grpoRewardKind,
        model_reference: values.grpoRewardKind === "local_model" ? orNull(values.grpoRewardModelReference) : null,
        on_training_gpu: values.grpoRewardOnTrainingGpu,
      },
      rollout_backend: "transformers_shared_policy",
    },
    hardware: hardwareFrom(values, presets),
    scope: {
      include_evaluation: values.includeEvaluation,
      include_checkpoint_save: values.includeCheckpointSave,
    },
    profiling: { enabled: false },
    margin_policy: {
      min_bytes: gibInputToBytes(values.marginMinGiB) ?? 0,
      fraction: Number(values.marginFraction),
    },
  };
}

export type BuildResult =
  | { ok: true; request: ExplicitAnalysisRequest }
  | { ok: false; message: string };

/** Builds and validates the request against the strict contract mirror. */
export function buildRequest(values: FormValues, presets: readonly GpuPreset[] = []): BuildResult {
  const parsed = analysisRequestSchema.safeParse(toAnalysisRequest(values, presets));
  if (parsed.success) return { ok: true, request: parsed.data };
  const first = parsed.error.issues[0];
  return {
    ok: false,
    message: `요청 형식 검증 실패: ${first ? `${first.path.join(".")} — ${first.message}` : "알 수 없는 오류"}`,
  };
}

const str = (n: number | null | undefined): string => (n == null ? "" : String(n));

/** Restores form values from a stored request (e.g. `requested_config` after a reload). */
export function fromAnalysisRequest(request: AnalysisRequest): FormValues {
  const d = DEFAULT_FORM_VALUES;
  const training = request.training;
  const lora = training.lora;
  const mapping = request.dataset.mapping ?? null;
  const dpo = request.dpo;
  const grpo = request.grpo;
  const hardware = request.hardware;
  const target = lora?.target_modules ?? "auto_verified";
  const thinking = training.template?.enable_thinking;
  const usesSpg = grpo?.steps_per_generation != null && grpo.generation_batch_size == null;
  return {
    ...d,
    modelReference: request.model.reference,
    modelRevision: request.model.revision ?? "",
    loadingScope: request.model.loading_scope ?? d.loadingScope,

    datasetReference: request.dataset.reference,
    datasetUploadRef: request.dataset.source_type === "upload" ? request.dataset.reference : "",
    datasetRevision: request.dataset.revision ?? "",
    datasetConfig: request.dataset.config ?? "",
    datasetSplit: request.dataset.split ?? "",
    datasetEvalSplit: request.dataset.eval_split ?? "",
    mappingEnabled: mapping !== null,
    mappingFormat: mapping?.format ?? d.mappingFormat,
    mapSystem: mapping?.system ?? "",
    mapPrompt: mapping?.prompt ?? "",
    mapChosen: mapping?.chosen ?? "",
    mapRejected: mapping?.rejected ?? "",
    mapCompletion: mapping?.completion ?? "",
    mapMessages: mapping?.messages ?? "",
    mapText: mapping?.text ?? "",
    emptySystemPolicy: mapping?.empty_system_policy ?? d.emptySystemPolicy,
    enableThinking: thinking == null ? "default" : thinking ? "on" : "off",

    objective: training.objective,
    strategy: training.strategy,
    load4bit: training.quantization?.enabled ?? training.strategy === "qlora",
    quantFormat: training.quantization?.format ?? d.quantFormat,
    doubleQuant: training.quantization?.double_quant ?? d.doubleQuant,
    computeDtype: training.quantization?.compute_dtype ?? d.computeDtype,

    loraR: str(lora?.r ?? 16),
    loraAlpha: str(lora?.alpha ?? 32),
    loraDropout: str(lora?.dropout ?? 0),
    loraTargetMode: Array.isArray(target) ? "custom" : target,
    loraTargetModules: Array.isArray(target) ? target.join(", ") : "",
    loraExcludeModules: (lora?.exclude_modules ?? []).join(", "),
    loraModulesToSave: (lora?.modules_to_save ?? []).join(", "),
    loraBias: lora?.bias ?? d.loraBias,
    loraRankPattern: Object.entries(lora?.rank_pattern ?? {})
      .map(([k, v]) => `${k}=${v}`)
      .join("\n"),
    loraUseDora: lora?.use_dora ?? false,
    loraUseRslora: lora?.use_rslora ?? false,

    microbatch: str(training.microbatch_per_device),
    accumulation: str(training.gradient_accumulation_steps),
    padToMultipleOf: str(training.pad_to_multiple_of),
    precision: training.precision ?? d.precision,
    loadDtype: training.load_dtype ?? d.loadDtype,
    optimizer: training.optimizer ?? d.optimizer,

    gradientCheckpointing: training.gradient_checkpointing ?? true,
    attentionBackend: training.attention_backend ?? d.attentionBackend,
    linearAttentionKernel: training.linear_attention_kernel ?? d.linearAttentionKernel,
    lossKernel: training.loss_kernel ?? d.lossKernel,
    backendProfile: training.backend_profile ?? "auto",

    dpoReferenceStrategy: dpo?.reference_strategy ?? d.dpoReferenceStrategy,
    dpoReferenceModel: dpo?.reference_model ?? "",
    dpoBeta: str(dpo?.beta ?? 0.1),
    dpoLossType: dpo?.loss_type ?? d.dpoLossType,
    dpoPrecomputeBatchSize: str(dpo?.precompute_batch_size),
    dpoSyncRefModel: dpo?.sync_ref_model ?? false,

    grpoNumGenerations: str(grpo?.num_generations ?? 4),
    grpoGenerationUnit: usesSpg ? "steps_per_generation" : "generation_batch_size",
    grpoGenerationBatchSize: usesSpg ? "" : str(grpo?.generation_batch_size),
    grpoStepsPerGeneration: str(grpo?.steps_per_generation),
    grpoNumIterations: str(grpo?.num_iterations ?? 1),
    grpoBudgetMode: grpo?.completion_budget != null ? "explicit" : "candidates",
    grpoCompletionBudget: str(grpo?.completion_budget),
    grpoBudgetCandidates: (grpo?.completion_budget_candidates ?? [1024, 2048, 4096, 8192]).join(", "),
    grpoBeta: str(grpo?.beta ?? 0),
    grpoRewardKind: grpo?.reward?.kind ?? d.grpoRewardKind,
    grpoRewardModelReference: grpo?.reward?.model_reference ?? "",
    grpoRewardOnTrainingGpu: grpo?.reward?.on_training_gpu ?? true,

    hardwareMode: hardware?.mode ?? "capacity_only",
    gpuPresetId: hardware?.gpu_preset ?? "",
    hardwareTotalGiB: hardware?.mode === "custom" ? bytesToGibInput(hardware.device_total_bytes) : "",
    hardwareUsableGiB: bytesToGibInput(hardware?.usable_bytes),
    externalReservedGiB: bytesToGibInput(hardware?.external_reserved_bytes ?? 0) || "0",
    marginMinGiB: bytesToGibInput(request.margin_policy?.min_bytes ?? 2 * 1_073_741_824) || "0",
    marginFraction: str(request.margin_policy?.fraction ?? 0.15),

    includeEvaluation: request.scope?.include_evaluation ?? false,
    includeCheckpointSave: request.scope?.include_checkpoint_save ?? false,
    seed: str(training.seed ?? 42),
  };
}
