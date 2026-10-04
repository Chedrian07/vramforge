// Shared fixture pieces. Source identities, tokenizer facts and token-length statistics come from
// docs/research/example-model-dataset.md (VERIFIED golden values); memory numbers in the analysis
// fixtures are illustrative and exist only for tests and the dev mock mode.
import { toAnalysisRequest } from "@/lib/form/convert";
import { EXAMPLE_FORM_VALUES, type FormValues } from "@/lib/form/values";
import type { Schemas } from "@/lib/api/types";

import { fixtureId } from "./builders";

export const MODEL_REF = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B";
export const DATASET_REF = "CyberNative/Code_Vulnerability_Security_DPO";
export const MODEL_REVISION = "2367e865d009c13ac81713a2878291d33ab28177";
export const DATASET_REVISION = "81aeacf06cf43b16d7278a3a01f019a496a53c51";
export const CHAT_TEMPLATE_SHA256 = "59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63";
export const VOCAB = 248_320;

export const modelManifest: Schemas["SourceManifest"] = {
  kind: "model",
  source_type: "huggingface",
  reference: `hf:${MODEL_REF}`,
  repo_id: MODEL_REF,
  requested_revision: null,
  resolved_revision: MODEL_REVISION,
  files: [
    { path: "config.json", size: 2_784, sha256: "407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633", blob_id: null },
    { path: "tokenizer.json", size: 19_989_325, sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", blob_id: null },
    { path: "chat_template.jinja", size: 3_916, sha256: CHAT_TEMPLATE_SHA256, blob_id: null },
  ],
  private: false,
  gated: false,
  last_modified: null,
  fingerprint: fixtureId("model-manifest"),
  notes: [],
};

export const datasetManifest: Schemas["SourceManifest"] = {
  kind: "dataset",
  source_type: "huggingface",
  reference: `hf:${DATASET_REF}`,
  repo_id: DATASET_REF,
  requested_revision: null,
  resolved_revision: DATASET_REVISION,
  files: [
    {
      path: "secure_programming_dpo.json",
      size: 6_867_898,
      sha256: "ad93a85feadcaee3f9cc2ff34899adcede280bb47a3ac82f680c5924754ce82c",
      blob_id: null,
    },
  ],
  private: false,
  gated: false,
  last_modified: null,
  fingerprint: fixtureId("dataset-manifest"),
  notes: ["확장자는 .json이지만 내용은 JSON Lines입니다."],
};

export const tokenizerManifest: Schemas["TokenizerManifest"] = {
  tokenizer_class: "Qwen3_5Tokenizer",
  vocab_size: 248_077,
  config_vocab_size: VOCAB,
  bos_token: null,
  eos_token: "<|im_end|>",
  pad_token: "<|endoftext|>",
  adds_bos_by_default: false,
  adds_eos_by_default: false,
  model_max_length: 262_144,
  model_max_length_is_sentinel: false,
  chat_template_present: true,
  chat_template_source: "chat_template.jinja",
  chat_template_sha256: CHAT_TEMPLATE_SHA256,
  has_generation_markers: true,
  template_kwargs: ["enable_thinking"],
  files_sha256: {
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "chat_template.jinja": CHAT_TEMPLATE_SHA256,
  },
  fingerprint: fixtureId("tokenizer"),
};

export const architectureFacts: Schemas["ArchitectureFacts"] = {
  architectures: ["Qwen3_5ForConditionalGeneration"],
  model_type: "qwen3_5",
  text_model_type: "qwen3_5_text",
  num_hidden_layers: 32,
  hidden_size: 4_096,
  intermediate_size: 12_288,
  vocab_size: VOCAB,
  num_attention_heads: 16,
  num_key_value_heads: 4,
  head_dim: 256,
  head_dim_source: "explicit",
  layer_types: [],
  layer_type_counts: [
    { layer_type: "linear_attention", count: 24 },
    { layer_type: "full_attention", count: 8 },
  ],
  sliding_window: null,
  linear_attention: {},
  max_position_embeddings: 262_144,
  tie_word_embeddings: false,
  config_dtype: "bfloat16",
  has_vision: true,
  has_mtp: true,
  rope: {},
  extra: {},
  config_sha256: "407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633",
};

// Tensor and parameter counts are the safetensors-header facts of the example checkpoint
// (docs/research/loading-quantization-peft.md V1, architecture-memory.md V19: 760 BF16 tensors,
// 9,409,813,744 params = text 8,953,803,264 + vision 456,010,480, no mtp tensors).
// linear_module_count is illustrative only (not shown by the UI).
export const modelSummary: Schemas["ModelInventorySummary"] = {
  facts: architectureFacts,
  tensor_count: 760,
  linear_module_count: 409,
  params_total: 9_409_813_744,
  bytes_serialized_total: 18_819_627_488,
  by_component: [
    { component: "text", params: 8_953_803_264, bytes_serialized: 17_907_606_528 },
    { component: "vision", params: 456_010_480, bytes_serialized: 912_020_960 },
  ],
  tied_groups: [],
  quantized_checkpoint_format: null,
  inventory_hash: fixtureId("inventory"),
};

export const preferenceMapping: Schemas["ColumnMapping"] = {
  format: "preference",
  system: "system",
  prompt: "question",
  chosen: "chosen",
  rejected: "rejected",
  completion: null,
  messages: null,
  text: null,
  empty_system_policy: "omit",
};

export function exampleRequest(overrides: Partial<FormValues> = {}): Schemas["AnalysisRequest"] {
  return toAnalysisRequest({ ...EXAMPLE_FORM_VALUES, ...overrides });
}

export const CREATED_AT = "2026-10-04T12:00:00Z";

export function issue(
  code: Schemas["ErrorCode"],
  severity: Schemas["Severity"],
  message: string,
  extra: Partial<Schemas["Issue"]> = {},
): Schemas["Issue"] {
  return {
    code,
    severity,
    stage: null,
    retryable: false,
    user_message: message,
    technical_detail_ref: null,
    affected_component: null,
    details: {},
    ...extra,
  };
}

export const ASSUMPTIONS: Schemas["Assumption"][] = [
  {
    id: "cuda-context",
    text: "CUDA context와 라이브러리 핸들은 0.5–1.0 GiB로 가정합니다 (정적 가정, GPU 미측정).",
    evidence: "assumption",
    source: "profile workspace_assumptions",
  },
  {
    id: "allocator-slack",
    text: "caching allocator 여유분을 동시 상주량의 5–10%로 둡니다.",
    evidence: "assumption",
    source: "profile workspace_assumptions",
  },
];
