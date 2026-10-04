// Strict Zod mirror of the AnalysisRequest contract
// (packages/estimator/src/vramforge_estimator/schemas/request.py). Every outgoing request is
// parsed with it, so unknown fields are never sent and constraints match the server.
// The KeyParity checks below fail typecheck when the generated contract gains or loses a field.
import { z } from "zod";

import type { Schemas } from "@/lib/api/types";

const sourceType = z.enum(["huggingface", "local", "upload"]);

export const modelSourceRefSchema = z.strictObject({
  source_type: sourceType,
  reference: z.string().min(1).max(2048),
  revision: z.string().max(256).nullable(),
  loading_scope: z.enum(["auto_verified", "full_checkpoint", "text_only"]),
});

export const datasetFormatSchema = z.enum([
  "auto",
  "preference",
  "prompt_completion",
  "prompt_only",
  "messages",
  "text",
]);

export const columnMappingSchema = z.strictObject({
  format: datasetFormatSchema,
  system: z.string().nullable(),
  prompt: z.string().nullable(),
  chosen: z.string().nullable(),
  rejected: z.string().nullable(),
  completion: z.string().nullable(),
  messages: z.string().nullable(),
  text: z.string().nullable(),
  empty_system_policy: z.enum(["omit", "keep"]),
});

export const datasetSourceRefSchema = z.strictObject({
  source_type: sourceType,
  reference: z.string().min(1).max(2048),
  revision: z.string().max(256).nullable(),
  config: z.string().max(256).nullable(),
  split: z.string().max(256).nullable(),
  eval_split: z.string().max(256).nullable(),
  scan_mode: z.enum(["full", "sample"]),
  sample_rows: z.number().int().min(1).nullable(),
  mapping: columnMappingSchema.nullable(),
});

const dtypeChoice = z.enum(["auto", "bfloat16", "float16", "float32"]);

export const quantizationSchema = z.strictObject({
  enabled: z.boolean(),
  format: z.enum(["nf4", "fp4"]),
  double_quant: z.boolean(),
  compute_dtype: dtypeChoice,
});

export const loraSchema = z.strictObject({
  r: z.number().int().min(1).max(4096),
  alpha: z.number().gt(0),
  dropout: z.number().min(0).lt(1),
  target_modules: z.union([z.literal("auto_verified"), z.literal("all-linear"), z.array(z.string())]),
  exclude_modules: z.array(z.string()),
  modules_to_save: z.array(z.string()),
  bias: z.enum(["none", "all", "lora_only"]),
  rank_pattern: z.record(z.string(), z.number().int()),
  use_dora: z.boolean(),
  use_rslora: z.boolean(),
});

export const offloadSchema = z.strictObject({
  parameters: z.boolean(),
  optimizer: z.boolean(),
  activations: z.boolean(),
});

export const templateOptionsSchema = z.strictObject({ enable_thinking: z.boolean().nullable() });

export const trainingSchema = z.strictObject({
  objective: z.enum(["sft", "dpo", "grpo"]),
  strategy: z.enum(["full", "lora", "qlora"]),
  quantization: quantizationSchema,
  lora: loraSchema,
  microbatch_per_device: z.number().int().min(1).max(4096).nullable(),
  gradient_accumulation_steps: z.number().int().min(1).max(65536).nullable(),
  gradient_checkpointing: z.boolean(),
  precision: z.enum(["auto", "bf16", "fp16", "fp32"]),
  optimizer: z.enum(["adamw_torch", "adamw_torch_fused", "adamw_8bit", "paged_adamw_8bit"]),
  load_dtype: dtypeChoice,
  attention_backend: z.enum(["auto", "sdpa", "eager", "flash_attention_2"]),
  linear_attention_kernel: z.enum(["auto", "torch_fallback", "fla"]),
  loss_kernel: z.enum(["auto", "standard", "chunked", "liger"]),
  packing: z.boolean(),
  pad_to_multiple_of: z.number().int().min(1).max(4096).nullable(),
  offload: offloadSchema,
  compile: z.boolean(),
  num_devices: z.number().int().min(1).max(1024),
  data_policy: z.literal("strict_no_truncation"),
  backend_profile: z.string().max(256),
  template: templateOptionsSchema,
  seed: z.number().int(),
});

export const dpoSchema = z.strictObject({
  reference_strategy: z.enum(["auto", "frozen_base_switch", "standalone_model", "precomputed_log_probs"]),
  reference_model: z.string().max(2048).nullable(),
  beta: z.number().min(0),
  loss_type: z.string().max(64),
  precompute_batch_size: z.number().int().min(1).nullable(),
  sync_ref_model: z.boolean(),
});

export const rewardSchema = z.strictObject({
  kind: z.enum(["unspecified", "cpu_rule", "remote", "local_model"]),
  model_reference: z.string().max(2048).nullable(),
  on_training_gpu: z.boolean(),
});

export const grpoSchema = z.strictObject({
  num_generations: z.number().int().min(1).max(1024),
  generation_batch_size: z.number().int().min(1).nullable(),
  steps_per_generation: z.number().int().min(1).nullable(),
  num_iterations: z.number().int().min(1),
  completion_budget: z.number().int().min(1).max(1_048_576).nullable(),
  completion_budget_candidates: z.array(z.number().int().min(1).max(1_048_576)).min(1).max(16),
  max_live_sequences: z.number().int().min(1).nullable(),
  beta: z.number().min(0),
  reward: rewardSchema,
  rollout_backend: z.enum(["transformers_shared_policy", "vllm_colocate", "vllm_server"]),
});

export const hardwareSchema = z.strictObject({
  mode: z.enum(["capacity_only", "gpu_preset", "custom"]),
  gpu_preset: z.string().max(128).nullable(),
  device_total_bytes: z.number().int().min(1).nullable(),
  usable_bytes: z.number().int().min(1).nullable(),
  external_reserved_bytes: z.number().int().min(0),
  num_gpus: z.number().int().min(1).max(1024),
});

export const scopeSchema = z.strictObject({
  include_evaluation: z.boolean(),
  include_checkpoint_save: z.boolean(),
});

export const profilingSchema = z.strictObject({ enabled: z.boolean() });

export const marginPolicySchema = z.strictObject({
  min_bytes: z.number().int().min(0),
  fraction: z.number().min(0).max(10),
});

export const analysisRequestSchema = z.strictObject({
  schema_version: z.literal("1.0"),
  model: modelSourceRefSchema,
  dataset: datasetSourceRefSchema,
  training: trainingSchema,
  dpo: dpoSchema,
  grpo: grpoSchema,
  hardware: hardwareSchema,
  scope: scopeSchema,
  profiling: profilingSchema,
  margin_policy: marginPolicySchema,
});

/** A fully explicit request: every field present, as sent by this UI. */
export type ExplicitAnalysisRequest = z.infer<typeof analysisRequestSchema>;

// ---------------------------------------------------------------- contract parity (compile time)

type KeyParity<A, B> = [Exclude<keyof A, keyof B>, Exclude<keyof B, keyof A>] extends [never, never]
  ? true
  : { onlyInContract: Exclude<keyof A, keyof B>; onlyInMirror: Exclude<keyof B, keyof A> };
type Infer<T extends z.ZodType> = z.infer<T>;

export const contractParity: {
  request: KeyParity<Schemas["AnalysisRequest"], Infer<typeof analysisRequestSchema>>;
  model: KeyParity<Schemas["ModelSourceRef"], Infer<typeof modelSourceRefSchema>>;
  dataset: KeyParity<Schemas["DatasetSourceRef"], Infer<typeof datasetSourceRefSchema>>;
  mapping: KeyParity<Schemas["ColumnMapping"], Infer<typeof columnMappingSchema>>;
  training: KeyParity<Schemas["TrainingConfig"], Infer<typeof trainingSchema>>;
  quantization: KeyParity<Schemas["QuantizationConfig"], Infer<typeof quantizationSchema>>;
  lora: KeyParity<Schemas["LoraConfig"], Infer<typeof loraSchema>>;
  offload: KeyParity<Schemas["OffloadConfig"], Infer<typeof offloadSchema>>;
  template: KeyParity<Schemas["TemplateOptions"], Infer<typeof templateOptionsSchema>>;
  dpo: KeyParity<Schemas["DpoConfig"], Infer<typeof dpoSchema>>;
  grpo: KeyParity<Schemas["GrpoConfig"], Infer<typeof grpoSchema>>;
  reward: KeyParity<Schemas["RewardConfig"], Infer<typeof rewardSchema>>;
  hardware: KeyParity<Schemas["HardwareConfig"], Infer<typeof hardwareSchema>>;
  scope: KeyParity<Schemas["ScopeConfig"], Infer<typeof scopeSchema>>;
  profiling: KeyParity<Schemas["ProfilingConfig"], Infer<typeof profilingSchema>>;
  margin: KeyParity<Schemas["MarginPolicy"], Infer<typeof marginPolicySchema>>;
} = {
  request: true,
  model: true,
  dataset: true,
  mapping: true,
  training: true,
  quantization: true,
  lora: true,
  offload: true,
  template: true,
  dpo: true,
  grpo: true,
  reward: true,
  hardware: true,
  scope: true,
  profiling: true,
  margin: true,
};

// The mirror's output must also be assignable to the generated request type.
const assignable: Schemas["AnalysisRequest"] = null as unknown as ExplicitAnalysisRequest;
void assignable;
