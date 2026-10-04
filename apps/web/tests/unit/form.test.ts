import { describe, expect, it } from "vitest";

import type { DatasetInspection } from "@/lib/api/types";

import { buildRequest, fromAnalysisRequest, toAnalysisRequest } from "@/lib/form/convert";
import { canonicalJson } from "@/lib/form/fingerprint";
import { mappingSyncAfterInspection } from "@/lib/form/inspect";
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

  it("sends only the roles of the selected mapping format", () => {
    // The user first had a preference mapping, then switched the format to messages: the
    // preference roles stay in the form but are not part of the explicit mapping.
    const r = request({
      ...base,
      mappingEnabled: true,
      mappingFormat: "messages",
      mapSystem: "system",
      mapPrompt: "question",
      mapChosen: "chosen",
      mapRejected: "rejected",
      mapMessages: "conversations",
    });
    expect(r.dataset.mapping).toEqual({
      format: "messages",
      system: null,
      prompt: null,
      chosen: null,
      rejected: null,
      completion: null,
      messages: "conversations",
      text: null,
      empty_system_policy: "omit",
    });
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
    // Other forms the server's SourceResolver accepts (sources/references.py).
    expect(isValidModelReference("https://hf.co/org/m")).toBe(true);
    expect(isValidModelReference("http://huggingface.co/org/m")).toBe(true);
    expect(isValidModelReference("gpt2")).toBe(true);
    expect(isValidModelReference("hf:org/m")).toBe(true);
    expect(isValidModelReference("hf-dataset:org/d")).toBe(false);
    expect(isValidModelReference("https://hf.co/datasets/org/d")).toBe(false);
    expect(isValidDatasetReference("hf-dataset:org/d")).toBe(true);
    expect(isValidDatasetReference("hf:org/m")).toBe(false);
    expect(isValidDatasetReference("upload:vf-fixture-upload-01")).toBe(true);
    expect(isValidDatasetReference("https://example.com/data.jsonl")).toBe(false);
    expect(fieldErrors({ ...base, modelReference: "not a model" }).modelReference).toContain("Hugging Face ID");
    expect(fieldErrors({ ...base, modelReference: "" }).modelReference).toBe("모델을 입력하세요");
  });

  it("requires a complete explicit mapping (format and required roles)", () => {
    const explicit = { ...base, mappingEnabled: true };
    expect(fieldErrors({ ...explicit, mapPrompt: "question" }).mappingFormat).toContain("데이터 형식을 골라야");
    const partial = fieldErrors({ ...explicit, mappingFormat: "preference", mapPrompt: "question", mapChosen: "chosen" });
    expect(partial.mapRejected).toContain("dispreferred response (rejected)");
    expect(partial.mapChosen).toBeUndefined();
    expect(fieldErrors({ ...explicit, mappingFormat: "prompt_completion", mapPrompt: "q" }).mapCompletion).toBeDefined();
    const complete = fieldErrors({ ...explicit, mappingFormat: "preference", mapChosen: "chosen", mapRejected: "rejected" });
    expect(complete).toEqual({});
    // Auto-detection (no explicit mapping) needs no roles.
    expect(fieldErrors(base)).toEqual({});
  });

  it("rejects Full + 4-bit", () => {
    expect(fieldErrors({ ...base, strategy: "full", load4bit: true }).load4bit).toContain("지원하지 않는 조합");
  });

  it("keeps cross-field checks while a reference is still empty", () => {
    const errors = fieldErrors({ ...DEFAULT_FORM_VALUES, strategy: "full", load4bit: true, hardwareMode: "custom" });
    expect(errors.modelReference).toBe("모델을 입력하세요");
    expect(errors.datasetReference).toBe("데이터셋을 입력하거나 파일을 업로드하세요");
    expect(errors.load4bit).toContain("지원하지 않는 조합");
    expect(errors.hardwareTotalGiB).toBeDefined();
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

describe("GRPO budget candidates", () => {
  it("accepts at most 16 budgets of 1–1,048,576 tokens like the contract", () => {
    const sixteen = Array.from({ length: 16 }, (_, i) => String(1024 * (i + 1))).join(", ");
    const seventeen = `${sixteen}, 32768`;
    expect(fieldErrors({ ...base, objective: "grpo", grpoBudgetCandidates: sixteen }).grpoBudgetCandidates).toBeUndefined();
    expect(fieldErrors({ ...base, objective: "grpo", grpoBudgetCandidates: seventeen }).grpoBudgetCandidates).toMatch(/최대 16개/);
    expect(fieldErrors({ ...base, objective: "grpo", grpoBudgetCandidates: "0, 1024" }).grpoBudgetCandidates).toBeDefined();
    expect(fieldErrors({ ...base, objective: "grpo", grpoBudgetCandidates: "2097152" }).grpoBudgetCandidates).toBeDefined();
  });

  it("never sends a candidate list the server would refuse", () => {
    const r = request({ ...base, objective: "grpo" });
    const tooMany = { ...r, grpo: { ...r.grpo, completion_budget_candidates: Array.from({ length: 17 }, () => 1024) } };
    expect(analysisRequestSchema.safeParse(tooMany).success).toBe(false);
    expect(analysisRequestSchema.safeParse({ ...r, grpo: { ...r.grpo, completion_budget_candidates: [] } }).success).toBe(false);
    expect(analysisRequestSchema.safeParse({ ...r, grpo: { ...r.grpo, completion_budget_candidates: [0] } }).success).toBe(false);
  });
});

describe("mapping after an inspection", () => {
  const suggestion = { format: "preference" as const, system: "system", prompt: "question", chosen: "chosen", rejected: "rejected", completion: null, messages: null, text: null, empty_system_policy: "omit" as const };
  const columns = ["system", "question", "chosen", "rejected", "lang"].map((name) => ({ name, dtype: "string", kind: "string" }));
  const answer = (extra: Partial<DatasetInspection> = {}): DatasetInspection => ({
    columns,
    suggested_mapping: suggestion,
    mapping_ambiguous: false,
    split_auto_selected: false,
    ...extra,
  });
  const auto: FormValues = { ...DEFAULT_FORM_VALUES };
  const applied: FormValues = { ...DEFAULT_FORM_VALUES, mappingEnabled: true, mappingAutoApplied: true, mappingFormat: "preference", mapSystem: "system", mapPrompt: "question", mapChosen: "chosen", mapRejected: "rejected" };
  const chosen: FormValues = { ...applied, mappingAutoApplied: false, mapPrompt: "lang" };

  it("applies an unambiguous suggestion while the mapping is not the user's", () => {
    expect(mappingSyncAfterInspection(answer(), auto)).toEqual({ clear: false, resetPolicy: false, apply: suggestion });
    expect(mappingSyncAfterInspection(answer(), applied)).toEqual({ clear: false, resetPolicy: false, apply: suggestion });
  });

  it("drops an automatic mapping when the answer has no unambiguous suggestion", () => {
    expect(mappingSyncAfterInspection(answer({ suggested_mapping: null, mapping_ambiguous: true }), applied)).toEqual({ clear: true, resetPolicy: false, apply: null });
    expect(mappingSyncAfterInspection(answer({ mapping_ambiguous: true }), applied).apply).toBeNull();
    expect(mappingSyncAfterInspection(answer({ suggested_mapping: null }), auto)).toEqual({ clear: false, resetPolicy: false, apply: null });
  });

  it("keeps the user's mapping while its columns exist and resets it when they are gone", () => {
    expect(mappingSyncAfterInspection(answer({ suggested_mapping: null, mapping_ambiguous: true }), chosen)).toEqual({ clear: false, resetPolicy: false, apply: null });
    const other = answer({ columns: [{ name: "instruction", dtype: "string", kind: "string" }], suggested_mapping: null });
    expect(mappingSyncAfterInspection(other, chosen)).toEqual({ clear: true, resetPolicy: true, apply: null });
  });
});
