import { describe, expect, it } from "vitest";

import { buildRequest, fromAnalysisRequest, toAnalysisRequest } from "@/lib/form/convert";
import { canonicalJson } from "@/lib/form/fingerprint";
import { datasetSourceType, isValidDatasetReference, isValidModelReference } from "@/lib/form/references";
import { analysisRequestSchema } from "@/lib/form/request-schema";
import { DEFAULT_FORM_VALUES, EXAMPLE_FORM_VALUES, formSchema, type FormValues } from "@/lib/form/values";

const GIB = 1_073_741_824;
const base: FormValues = {
  ...DEFAULT_FORM_VALUES,
  modelReference: "org/model",
  datasetReference: "org/data",
};

function request(values: FormValues) {
  const built = buildRequest(values, [{ id: "a100-80", name: "A100 80GB", total_bytes: 80 * GIB, note: "" }]);
  if (!built.ok) throw new Error(built.message);
  return built.request;
}

function fieldErrors(values: FormValues): Record<string, string> {
  const parsed = formSchema.safeParse(values);
  if (parsed.success) return {};
  const out: Record<string, string> = {};
  for (const issue of parsed.error.issues) out[String(issue.path[0])] ??= issue.message;
  return out;
}

describe("request building", () => {
  it("reproduces the plan §15.2 example request", () => {
    const r = request(EXAMPLE_FORM_VALUES);
    expect(r.model).toEqual({
      source_type: "huggingface",
      reference: "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
      revision: null,
      loading_scope: "auto_verified",
    });
    expect(r.dataset).toMatchObject({
      reference: "CyberNative/Code_Vulnerability_Security_DPO",
      revision: null,
      config: null,
      split: "train",
      scan_mode: "full",
      mapping: null,
    });
    expect(r.training).toMatchObject({
      objective: "grpo",
      strategy: "qlora",
      quantization: { enabled: true, format: "nf4", double_quant: true, compute_dtype: "auto" },
      lora: { r: 16, alpha: 32, dropout: 0, target_modules: "auto_verified", modules_to_save: [] },
      microbatch_per_device: 1,
      gradient_accumulation_steps: 4,
      gradient_checkpointing: true,
      packing: false,
      data_policy: "strict_no_truncation",
      backend_profile: "auto",
    });
    expect(r.grpo).toMatchObject({
      num_generations: 4,
      generation_batch_size: 4,
      steps_per_generation: null,
      completion_budget: null,
      completion_budget_candidates: [1024, 2048, 4096, 8192],
      beta: 0,
      reward: { kind: "unspecified", model_reference: null },
      rollout_backend: "transformers_shared_policy",
    });
    expect(r.hardware.mode).toBe("capacity_only");
    expect(r.scope).toEqual({ include_evaluation: false, include_checkpoint_save: false });
    expect(r.profiling).toEqual({ enabled: false });
    expect(r.margin_policy).toEqual({ min_bytes: 2 * GIB, fraction: 0.15 });
  });

  it("never sends unknown fields (strict mirror)", () => {
    const r = request(base);
    expect(analysisRequestSchema.safeParse({ ...r, extra: 1 }).success).toBe(false);
    expect(analysisRequestSchema.safeParse({ ...r, training: { ...r.training, foo: true } }).success).toBe(false);
  });

  it("uses empty fields as profile-preset auto (null), not zero", () => {
    const r = request(base);
    expect(r.training.microbatch_per_device).toBeNull();
    expect(r.training.gradient_accumulation_steps).toBeNull();
    expect(r.training.pad_to_multiple_of).toBeNull();
  });

  it("keeps generation_batch_size and steps_per_generation mutually exclusive", () => {
    const spg = request({ ...base, objective: "grpo", grpoGenerationUnit: "steps_per_generation", grpoStepsPerGeneration: "8" });
    expect(spg.grpo.generation_batch_size).toBeNull();
    expect(spg.grpo.steps_per_generation).toBe(8);
  });

  it("sends an explicit budget only in explicit mode", () => {
    const r = request({ ...base, objective: "grpo", grpoBudgetMode: "explicit", grpoCompletionBudget: "2048" });
    expect(r.grpo.completion_budget).toBe(2048);
  });

  it("keeps the preset total exact without waiting for the preset list", () => {
    const restored = fromAnalysisRequest(
      request({ ...base, hardwareMode: "gpu_preset", gpuPresetId: "a100-80", gpuPresetTotalBytes: String(80 * GIB) }),
    );
    expect(restored.gpuPresetTotalBytes).toBe(String(80 * GIB));
    const built = buildRequest(restored, []);
    expect(built.ok && built.request.hardware.device_total_bytes).toBe(80 * GIB);
  });

  it("converts hardware GiB inputs to integer bytes", () => {
    const preset = request({ ...base, hardwareMode: "gpu_preset", gpuPresetId: "a100-80", externalReservedGiB: "1.5" });
    expect(preset.hardware).toEqual({
      mode: "gpu_preset",
      gpu_preset: "a100-80",
      device_total_bytes: 80 * GIB,
      usable_bytes: null,
      external_reserved_bytes: 1.5 * GIB,
      num_gpus: 1,
    });
    const custom = request({ ...base, hardwareMode: "custom", hardwareTotalGiB: "24", hardwareUsableGiB: "22.5" });
    expect(custom.hardware.device_total_bytes).toBe(24 * GIB);
    expect(custom.hardware.usable_bytes).toBe(22.5 * GIB);
  });

  it("detects upload and local dataset sources", () => {
    expect(datasetSourceType("u-123", "u-123")).toBe("upload");
    expect(datasetSourceType("local:datasets/x.jsonl", "")).toBe("local");
    expect(datasetSourceType("org/data", "")).toBe("huggingface");
    const r = request({ ...base, datasetReference: "u-123", datasetUploadRef: "u-123" });
    expect(r.dataset.source_type).toBe("upload");
  });

  it("round-trips through requested_config", () => {
    const values: FormValues = {
      ...EXAMPLE_FORM_VALUES,
      mappingEnabled: true,
      mappingFormat: "preference",
      mapPrompt: "question",
      mapChosen: "chosen",
      mapRejected: "rejected",
      mapSystem: "system",
      loraTargetMode: "custom",
      loraTargetModules: "q_proj, v_proj",
      loraRankPattern: "q_proj=8",
      hardwareMode: "custom",
      hardwareTotalGiB: "24",
      enableThinking: "off",
    };
    const once = toAnalysisRequest(values);
    const twice = toAnalysisRequest(fromAnalysisRequest(once));
    expect(canonicalJson(twice)).toBe(canonicalJson(once));
  });
});

describe("form validation", () => {
  it("validates references locally with Korean messages", () => {
    expect(isValidModelReference("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")).toBe(true);
    expect(isValidModelReference("https://huggingface.co/org/m/tree/main")).toBe(true);
    expect(isValidModelReference("https://huggingface.co/datasets/org/d")).toBe(false);
    expect(isValidModelReference("local:models/qwen")).toBe(true);
    expect(isValidModelReference("http://169.254.169.254/latest")).toBe(false);
    expect(isValidDatasetReference("https://huggingface.co/datasets/org/d/viewer/default/train?row=0")).toBe(true);
    expect(fieldErrors({ ...base, modelReference: "not a model" }).modelReference).toContain("Hugging Face ID");
    expect(fieldErrors({ ...base, modelReference: "" }).modelReference).toBe("모델을 입력하세요");
  });

  it("rejects Full + 4-bit", () => {
    expect(fieldErrors({ ...base, strategy: "full", load4bit: true }).load4bit).toContain("지원하지 않는 조합");
  });

  it("checks numeric ranges and cross-field hardware rules", () => {
    expect(fieldErrors({ ...base, loraR: "0" }).loraR).toBe("1–4,096 사이 정수를 입력하세요");
    expect(fieldErrors({ ...base, microbatch: "" }).microbatch).toBeUndefined();
    expect(fieldErrors({ ...base, hardwareMode: "custom" }).hardwareTotalGiB).toBeDefined();
    expect(
      fieldErrors({ ...base, hardwareMode: "custom", hardwareTotalGiB: "24", hardwareUsableGiB: "30" }).hardwareUsableGiB,
    ).toContain("전체 용량 이하");
    expect(fieldErrors({ ...base, objective: "grpo", grpoBudgetMode: "explicit", grpoCompletionBudget: "x" }).grpoCompletionBudget).toBeDefined();
    expect(fieldErrors({ ...base, loraRankPattern: "q_proj=abc" }).loraRankPattern).toBeDefined();
  });
});
