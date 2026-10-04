// UI form state. Numeric inputs are kept as text ("" = 자동/none) and converted in convert.ts,
// so an empty field can mean "resolve from the profile preset" (request `None`) instead of NaN.
import { z } from "zod";

import { isValidDatasetReference, isValidModelReference } from "./references";

const integerFmt = new Intl.NumberFormat("ko-KR");

function intText(min: number, max: number, optional = false) {
  const range = `${integerFmt.format(min)}–${integerFmt.format(max)}`;
  return z
    .string()
    .trim()
    .refine(
      (v) => (optional && v === "") || (/^\d+$/.test(v) && Number(v) >= min && Number(v) <= max),
      { message: optional ? `비워 두면 자동입니다. 값은 ${range} 사이 정수` : `${range} 사이 정수를 입력하세요` },
    );
}

function decimalText(test: (n: number) => boolean, message: string, optional = false) {
  return z
    .string()
    .trim()
    .refine((v) => (optional && v === "") || (v !== "" && Number.isFinite(Number(v)) && test(Number(v))), {
      message,
    });
}

/** "a, b\nc" -> ["a", "b", "c"] */
export function parseList(text: string): string[] {
  return text
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** "pattern=rank" per line -> record; null when a line is malformed. */
export function parseRankPattern(text: string): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const match = /^(.+?)\s*[=:]\s*(\d+)$/.exec(trimmed);
    if (!match?.[1] || !match[2]) return null;
    const rank = Number(match[2]);
    if (rank < 1 || rank > 4096) return null;
    out[match[1].trim()] = rank;
  }
  return out;
}

/** "1024, 2048" -> [1024, 2048]; null when any entry is not a valid budget. */
export function parseBudgetList(text: string): number[] | null {
  const items = parseList(text);
  if (items.length === 0) return null;
  const values = items.map((s) => (/^\d+$/.test(s) ? Number(s) : Number.NaN));
  if (values.some((n) => !Number.isInteger(n) || n < 1 || n > 1_048_576)) return null;
  return values;
}

export const formSchema = z
  .object({
    // Model
    modelReference: z
      .string()
      .trim()
      .min(1, { message: "모델을 입력하세요", abort: true })
      .max(2048, "2,048자 이하로 입력하세요")
      .refine(isValidModelReference, {
        message: "Hugging Face ID(org/name), huggingface.co URL 또는 local: 참조를 입력하세요",
      }),
    modelRevision: z.string().trim().max(256, "256자 이하로 입력하세요"),
    loadingScope: z.enum(["auto_verified", "full_checkpoint", "text_only"]),

    // Dataset
    datasetReference: z
      .string()
      .trim()
      .min(1, { message: "데이터셋을 입력하거나 파일을 업로드하세요", abort: true })
      .max(2048, "2,048자 이하로 입력하세요")
      .refine(isValidDatasetReference, {
        message: "Hugging Face 데이터셋 ID·URL, local: 참조 또는 업로드한 파일이어야 합니다",
      }),
    datasetUploadRef: z.string(),
    datasetRevision: z.string().trim().max(256, "256자 이하로 입력하세요"),
    datasetConfig: z.string().trim().max(256),
    datasetSplit: z.string().trim().max(256),
    datasetEvalSplit: z.string().trim().max(256),
    mappingEnabled: z.boolean(),
    mappingFormat: z.enum(["auto", "preference", "prompt_completion", "prompt_only", "messages", "text"]),
    mapSystem: z.string(),
    mapPrompt: z.string(),
    mapChosen: z.string(),
    mapRejected: z.string(),
    mapCompletion: z.string(),
    mapMessages: z.string(),
    mapText: z.string(),
    emptySystemPolicy: z.enum(["omit", "keep"]),
    enableThinking: z.enum(["default", "on", "off"]),

    // Method, strategy, quantization
    objective: z.enum(["sft", "dpo", "grpo"]),
    strategy: z.enum(["full", "lora", "qlora"]),
    load4bit: z.boolean(),
    quantFormat: z.enum(["nf4", "fp4"]),
    doubleQuant: z.boolean(),
    computeDtype: z.enum(["auto", "bfloat16", "float16", "float32"]),

    // Adapter
    loraR: intText(1, 4096),
    loraAlpha: decimalText((n) => n > 0, "0보다 큰 값을 입력하세요"),
    loraDropout: decimalText((n) => n >= 0 && n < 1, "0 이상 1 미만 값을 입력하세요"),
    loraTargetMode: z.enum(["auto_verified", "all-linear", "custom"]),
    loraTargetModules: z.string(),
    loraExcludeModules: z.string(),
    loraModulesToSave: z.string(),
    loraBias: z.enum(["none", "all", "lora_only"]),
    loraRankPattern: z.string(),
    loraUseDora: z.boolean(),
    loraUseRslora: z.boolean(),

    // Batch & precision
    microbatch: intText(1, 4096, true),
    accumulation: intText(1, 65536, true),
    padToMultipleOf: intText(1, 4096, true),
    precision: z.enum(["auto", "bf16", "fp16", "fp32"]),
    loadDtype: z.enum(["auto", "bfloat16", "float16", "float32"]),
    optimizer: z.enum(["adamw_torch", "adamw_torch_fused", "adamw_8bit", "paged_adamw_8bit"]),

    // Runtime
    gradientCheckpointing: z.boolean(),
    attentionBackend: z.enum(["auto", "sdpa", "eager", "flash_attention_2"]),
    linearAttentionKernel: z.enum(["auto", "torch_fallback", "fla"]),
    lossKernel: z.enum(["auto", "standard", "chunked", "liger"]),
    backendProfile: z.string().trim().min(1).max(256),

    // DPO
    dpoReferenceStrategy: z.enum(["auto", "frozen_base_switch", "standalone_model", "precomputed_log_probs"]),
    dpoReferenceModel: z.string().trim().max(2048),
    dpoBeta: decimalText((n) => n >= 0, "0 이상 값을 입력하세요"),
    dpoLossType: z.string().trim().min(1, "loss type을 입력하세요").max(64),
    dpoPrecomputeBatchSize: intText(1, 1_000_000, true),
    dpoSyncRefModel: z.boolean(),

    // GRPO
    grpoNumGenerations: intText(1, 1024),
    grpoGenerationUnit: z.enum(["generation_batch_size", "steps_per_generation"]),
    grpoGenerationBatchSize: intText(1, 1_000_000, true),
    grpoStepsPerGeneration: intText(1, 1_000_000, true),
    grpoNumIterations: intText(1, 1_000_000),
    grpoBudgetMode: z.enum(["candidates", "explicit"]),
    grpoCompletionBudget: z.string().trim(),
    grpoBudgetCandidates: z.string(),
    grpoBeta: decimalText((n) => n >= 0, "0 이상 값을 입력하세요"),
    grpoRewardKind: z.enum(["unspecified", "cpu_rule", "remote", "local_model"]),
    grpoRewardModelReference: z.string().trim().max(2048),
    grpoRewardOnTrainingGpu: z.boolean(),

    // Hardware & planning margin
    hardwareMode: z.enum(["capacity_only", "gpu_preset", "custom"]),
    gpuPresetId: z.string(),
    // Exact total of the selected preset (integer bytes as text); keeps the request independent of
    // when /backend-profiles finishes loading.
    gpuPresetTotalBytes: z.string().regex(/^\d*$/),
    hardwareTotalGiB: z.string().trim(),
    hardwareUsableGiB: z.string().trim(),
    externalReservedGiB: decimalText((n) => n >= 0, "0 이상 값을 입력하세요 (GiB)"),
    marginMinGiB: decimalText((n) => n >= 0, "0 이상 값을 입력하세요 (GiB)"),
    marginFraction: decimalText((n) => n >= 0 && n <= 10, "0–10 사이 값을 입력하세요"),

    // Scope & reproducibility
    includeEvaluation: z.boolean(),
    includeCheckpointSave: z.boolean(),
    seed: z
      .string()
      .trim()
      .refine((v) => /^-?\d+$/.test(v) && Number.isSafeInteger(Number(v)), "정수를 입력하세요"),
  })
  .superRefine((v, ctx) => {
    const issue = (path: string, message: string) => ctx.addIssue({ code: "custom", path: [path], message });
    if (v.strategy === "full" && v.load4bit) {
      issue("load4bit", "Full fine-tuning과 4-bit 로딩은 지원하지 않는 조합입니다");
    }
    if (v.strategy !== "full" && v.loraTargetMode === "custom" && parseList(v.loraTargetModules).length === 0) {
      issue("loraTargetModules", "대상 모듈을 하나 이상 입력하세요");
    }
    if (parseRankPattern(v.loraRankPattern) === null) {
      issue("loraRankPattern", "한 줄에 하나씩 `패턴=rank` 형식으로 입력하세요 (rank 1–4,096)");
    }
    if (v.objective === "grpo") {
      if (v.grpoBudgetMode === "explicit") {
        const n = v.grpoCompletionBudget;
        if (!/^\d+$/.test(n) || Number(n) < 1 || Number(n) > 1_048_576) {
          issue("grpoCompletionBudget", "1–1,048,576 사이 정수를 입력하세요");
        }
      } else if (parseBudgetList(v.grpoBudgetCandidates) === null) {
        issue("grpoBudgetCandidates", "쉼표로 구분한 정수 목록을 입력하세요 (각 1–1,048,576)");
      }
      if (v.grpoRewardKind === "local_model" && v.grpoRewardModelReference === "") {
        issue("grpoRewardModelReference", "로컬 reward 모델 참조를 입력하세요");
      }
    }
    const positiveGiB = (text: string) => text !== "" && Number.isFinite(Number(text)) && Number(text) > 0;
    if (v.hardwareMode === "custom" && !positiveGiB(v.hardwareTotalGiB)) {
      issue("hardwareTotalGiB", "GPU 전체 용량을 GiB로 입력하세요");
    }
    if (v.hardwareMode === "gpu_preset" && v.gpuPresetId === "") {
      issue("gpuPresetId", "GPU를 선택하세요");
    }
    if (v.hardwareMode !== "capacity_only" && v.hardwareUsableGiB !== "") {
      if (!positiveGiB(v.hardwareUsableGiB)) issue("hardwareUsableGiB", "0보다 큰 GiB 값을 입력하세요");
      else if (
        v.hardwareMode === "custom" &&
        positiveGiB(v.hardwareTotalGiB) &&
        Number(v.hardwareUsableGiB) > Number(v.hardwareTotalGiB)
      ) {
        issue("hardwareUsableGiB", "사용 가능 VRAM은 전체 용량 이하여야 합니다");
      }
    }
  });

export type FormValues = z.infer<typeof formSchema>;

export const DEFAULT_BUDGET_CANDIDATES = "1024, 2048, 4096, 8192";

/** Initial form (plan.md §5): request defaults, "Load in 4-bit" on -> QLoRA (plan §3.1 sketch). */
export const DEFAULT_FORM_VALUES: FormValues = {
  modelReference: "",
  modelRevision: "",
  loadingScope: "auto_verified",

  datasetReference: "",
  datasetUploadRef: "",
  datasetRevision: "",
  datasetConfig: "",
  datasetSplit: "",
  datasetEvalSplit: "",
  mappingEnabled: false,
  mappingFormat: "auto",
  mapSystem: "",
  mapPrompt: "",
  mapChosen: "",
  mapRejected: "",
  mapCompletion: "",
  mapMessages: "",
  mapText: "",
  emptySystemPolicy: "omit",
  enableThinking: "default",

  objective: "sft",
  strategy: "qlora",
  load4bit: true,
  quantFormat: "nf4",
  doubleQuant: true,
  computeDtype: "auto",

  loraR: "16",
  loraAlpha: "32",
  loraDropout: "0",
  loraTargetMode: "auto_verified",
  loraTargetModules: "",
  loraExcludeModules: "",
  loraModulesToSave: "",
  loraBias: "none",
  loraRankPattern: "",
  loraUseDora: false,
  loraUseRslora: false,

  microbatch: "",
  accumulation: "",
  padToMultipleOf: "",
  precision: "auto",
  loadDtype: "auto",
  optimizer: "adamw_torch",

  gradientCheckpointing: true,
  attentionBackend: "auto",
  linearAttentionKernel: "auto",
  lossKernel: "auto",
  backendProfile: "auto",

  dpoReferenceStrategy: "auto",
  dpoReferenceModel: "",
  dpoBeta: "0.1",
  dpoLossType: "sigmoid",
  dpoPrecomputeBatchSize: "",
  dpoSyncRefModel: false,

  grpoNumGenerations: "4",
  grpoGenerationUnit: "generation_batch_size",
  grpoGenerationBatchSize: "4",
  grpoStepsPerGeneration: "",
  grpoNumIterations: "1",
  grpoBudgetMode: "candidates",
  grpoCompletionBudget: "",
  grpoBudgetCandidates: DEFAULT_BUDGET_CANDIDATES,
  grpoBeta: "0",
  grpoRewardKind: "unspecified",
  grpoRewardModelReference: "",
  grpoRewardOnTrainingGpu: true,

  hardwareMode: "capacity_only",
  gpuPresetId: "",
  gpuPresetTotalBytes: "",
  hardwareTotalGiB: "",
  hardwareUsableGiB: "",
  externalReservedGiB: "0",
  marginMinGiB: "2",
  marginFraction: "0.15",

  includeEvaluation: false,
  includeCheckpointSave: false,
  seed: "42",
};

/**
 * The example input of plan.md §21 / §15.2 (product preset, not an analysis result):
 * GRPO + Load in 4-bit, Advanced unchanged except the §15.2 batch preset.
 */
export const EXAMPLE_FORM_VALUES: FormValues = {
  ...DEFAULT_FORM_VALUES,
  modelReference: "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
  datasetReference: "CyberNative/Code_Vulnerability_Security_DPO",
  datasetSplit: "train",
  objective: "grpo",
  strategy: "qlora",
  load4bit: true,
  microbatch: "1",
  accumulation: "4",
};
