// GRPO example (plan.md §21): no explicit completion budget -> one scenario per budget,
// reward unspecified -> conditional result with the reward footprint excluded.
// Token statistics are the verified golden values; memory items are illustrative fixture numbers.
import type { Schemas } from "@/lib/api/types";

import {
  GIB,
  MIB,
  NOT_EVALUATED_FIT,
  batchShape,
  breakdown,
  device,
  fixtureId,
  recommendation,
  stats,
  type ItemSpec,
  type PhaseSpec,
} from "./builders";
import {
  ASSUMPTIONS,
  CREATED_AT,
  VOCAB,
  datasetManifest,
  exampleRequest,
  issue,
  modelManifest,
  modelSummary,
  preferenceMapping,
  tokenizerManifest,
} from "./common";

export const BUDGETS = [1024, 2048, 4096, 8192] as const;
const PROMPT_MAX = 268;

export const STATIC_ITEMS: ItemSpec[] = [
  { name: "base weights · NF4 payload + quant state", category: "weights_base", low: 3_802_120_192, high: 3_802_120_192, evidence: "analytic" },
  { name: "base weights · non-quantized modules (bf16)", category: "weights_base", low: 4_080_218_624, high: 4_080_218_624, evidence: "analytic", note: "embedding, lm_head, norm, vision tower 비양자화 모듈" },
  { name: "LoRA adapter weights (bf16)", category: "weights_adapter", low: 102_530_048, high: 102_530_048, evidence: "analytic" },
  { name: "LoRA gradients (bf16, text path)", category: "gradients", low: 86_556_672, high: 86_556_672, evidence: "analytic" },
  { name: "AdamW states (2 × bf16)", category: "optimizer_states", low: 173_113_344, high: 173_113_344, evidence: "analytic" },
];

function overheadItems(base: ItemSpec[]): ItemSpec[] {
  const ctx: ItemSpec = { name: "CUDA context · library handles", category: "non_framework", low: 512 * MIB, high: 1 * GIB, evidence: "assumption" };
  const workspace: ItemSpec = { name: "cuBLAS / SDPA workspace", category: "workspace", low: 256 * MIB, high: 512 * MIB, evidence: "assumption" };
  const all = [...base, ctx, workspace];
  const lo = all.reduce((acc, i) => acc + (i.low ?? 0), 0);
  const hi = all.reduce((acc, i) => acc + (i.high ?? 0), 0);
  return [
    ctx,
    workspace,
    { name: "allocator slack", category: "allocator_slack", low: Math.round(lo * 0.05), high: Math.round(hi * 0.1), evidence: "assumption" },
  ];
}

function withOverhead(items: ItemSpec[]): ItemSpec[] {
  return [...items, ...overheadItems(items)];
}

function total(items: ItemSpec[], side: "low" | "high"): number {
  return items.reduce((acc, i) => acc + (i[side] ?? 0), 0);
}

function budgetScenario(budget: number): Schemas["ScenarioEstimate"] {
  const T = PROMPT_MAX + budget;
  const E = budget + 1;
  const peakItems = withOverhead([
    ...STATIC_ITEMS,
    { name: "checkpoint boundary activations", category: "saved_activations", low: 32 * 2 * 4096 * T, high: 32 * 2 * 4096 * T, evidence: "analytic", note: "[B, T, hidden] × 32 decoder layers (bf16)" },
    { name: "recompute working set (1 decoder layer)", category: "recompute_working_set", low: 48 * 4096 * T, high: 72 * 4096 * T, evidence: "analytic", note: "linear-attention torch fallback 경로 포함" },
    { name: "logits + grad_logits (fp32 fused log-prob)", category: "logits_and_loss", low: 8 * E * VOCAB, high: Math.round(8.67 * E * VOCAB), evidence: "analytic", note: "E = B_update·(L+1)·V" },
  ]);
  const peak = breakdown("POLICY_FORWARD_BACKWARD:backward", "POLICY_FORWARD_BACKWARD", peakItems);

  const kvBytes = 2 * 8 * 4 * 256 * 2 * 4 * (T - 1);
  const rolloutItems = withOverhead([
    ...STATIC_ITEMS.filter((i) => i.category !== "gradients"),
    { name: "KV cache (8 full-attention layers, C=4)", category: "generation_cache", low: kvBytes, high: kvBytes, evidence: "analytic" },
    { name: "linear-attention conv/recurrent state (C=4)", category: "recurrent_state", low: 198 * MIB, high: 204 * MIB, evidence: "analytic" },
    { name: "decode logits (C × V, fp32)", category: "logits_and_loss", low: 4 * VOCAB * 4, high: 4 * VOCAB * 4, evidence: "analytic" },
  ]);
  const loadItems = withOverhead([
    ...STATIC_ITEMS.filter((i) => i.category === "weights_base"),
    { name: "quantization transient (largest Linear)", category: "load_transient", low: 106_954_752, high: 106_954_752, evidence: "analytic" },
  ]);
  const optimizerItems = withOverhead([...STATIC_ITEMS]);

  const phases: PhaseSpec[] = [
    { phase: "MODEL_LOAD_AND_QUANTIZE", included: true, timepoint: "MODEL_LOAD_AND_QUANTIZE:quantize", low: total(loadItems, "low"), high: total(loadItems, "high") },
    { phase: "REFERENCE_PRECOMPUTE", included: false, excludedReason: "GRPO beta=0: reference 계산 없음" },
    { phase: "ROLLOUT_PREFILL_AND_DECODE", included: true, timepoint: "ROLLOUT_PREFILL_AND_DECODE:decode", low: total(rolloutItems, "low"), high: total(rolloutItems, "high") },
    { phase: "REWARD", included: false, excludedReason: "reward 미지정: reward footprint 미포함" },
    { phase: "POLICY_FORWARD_BACKWARD", included: true, timepoint: peak.timepoint, low: peak.total_low, high: peak.total_high },
    { phase: "OPTIMIZER_STEP", included: true, timepoint: "OPTIMIZER_STEP:step", low: total(optimizerItems, "low"), high: total(optimizerItems, "high") },
    { phase: "WEIGHT_SYNC", included: false, excludedReason: "공유 policy rollout: 별도 가중치 동기화 없음" },
    { phase: "EVALUATION", included: false, excludedReason: "평가 미포함 (scope.include_evaluation=false)" },
    { phase: "CHECKPOINT_SAVE_OR_CONSOLIDATE", included: false, excludedReason: "체크포인트 저장 미포함" },
  ];

  return {
    scenario_id: `budget_${budget}`,
    label: `completion budget ${budget.toLocaleString("en-US")}`,
    params: { completion_budget: budget },
    batch_shape: batchShape({
      name: `budget_${budget}`,
      objective: "grpo",
      rows: 1,
      sequences: 1,
      padded: T,
      logits: E,
      prompt: PROMPT_MAX,
      completion: budget,
      rowIds: ["train:2355", "train:3169"],
      description: `최대 prompt ${PROMPT_MAX} + budget ${budget}`,
    }),
    devices: [device(peak, peakItems, phases)],
    recommendation: recommendation(peak.total_high),
    hardware_fit: NOT_EVALUATED_FIT,
    excluded_components: [
      { name: "reward footprint", reason: "reward가 지정되지 않아 reward 모델·함수 메모리를 계산에서 제외했습니다.", code: "GRPO_REWARD_UNSPECIFIED" },
    ],
    timepoints: [],
    allocations: [],
  };
}

export const grpoScenarios = BUDGETS.map(budgetScenario);

export const grpoRequest = exampleRequest();

export const grpoResolved: Schemas["ResolvedConfig"] = {
  profile_id: "qwen3_5_hybrid.trl_1_14_1.analytic",
  profile_version: "2026.10.0",
  environment_id: "cuda-trl-1.14.1",
  dependency_lock_digest: `sha256:${fixtureId("lock-digest")}`,
  architecture_adapter: "qwen3_5_hybrid",
  trainer_adapter: "grpo",
  preprocessing_adapter: "trl_1_14_1.grpo",
  objective: "grpo",
  strategy: "qlora",
  loading_scope: "full_checkpoint",
  load_dtype: "bfloat16",
  processing_class: "tokenizer",
  effective_dtypes: {
    weights_nonquantized: "bfloat16",
    compute: "bfloat16",
    adapter: "bfloat16",
    gradient: "bfloat16",
    optimizer_state: "bfloat16",
    master_weights: null,
    logits: "float32",
    loss: "float32",
    kv_cache: "bfloat16",
    recurrent_state: "float32",
  },
  quantization: {
    enabled: true,
    method: "bnb_nf4",
    double_quant: true,
    blocksize: 64,
    nested_blocksize: 256,
    quant_storage_dtype: "uint8",
    compute_dtype: "bfloat16",
    skip_module_patterns: ["lm_head"],
  },
  upcast_to_fp32_patterns: [],
  lora: {
    r: 16,
    alpha: 32,
    dropout: 0,
    target_module_patterns: ["all-linear"],
    target_modules: ["model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.mlp.down_proj"],
    exclude_modules: [],
    modules_to_save: [],
    bias: "none",
    rank_pattern: {},
    use_dora: false,
    use_rslora: false,
  },
  trainable_full_patterns: [],
  optimizer: { name: "adamw_torch", states_per_param: 2, state_dtype: "bfloat16", eight_bit: false, block_size: null, min_8bit_size: null, paged: false, fused: false },
  microbatch: 1,
  accumulation: 4,
  pad_to_multiple_of: null,
  mixed_precision: "bf16",
  packing: false,
  assistant_only_loss: false,
  gradient_checkpointing: true,
  checkpointing_granularity: "per_decoder_layer",
  attention_path_by_layer_type: { full_attention: "sdpa", linear_attention: "torch_fallback" },
  loss_path: "grpo_fused_logprob",
  use_cache_during_training: false,
  dpo: null,
  grpo: {
    num_generations: 4,
    generation_batch_size: 4,
    steps_per_generation: 4,
    num_iterations: 1,
    completion_budgets: [...BUDGETS],
    budget_explicit: false,
    beta: 0,
    reference_needed: false,
    reward_kind: "unspecified",
    reward_model_reference: null,
    rollout_backend: "transformers_shared_policy",
    reward_on_training_gpu: true,
    live_sequences: 4,
    update_microbatch: 1,
    accumulation: 4,
  },
  workspace: {
    cuda_context_bytes: [512 * MIB, GIB],
    library_workspace_bytes: [256 * MIB, 512 * MIB],
    allocator_slack_fraction: [0.05, 0.1],
    notes: ["CUDA context·workspace는 정적 가정값이며 GPU에서 측정하지 않았습니다."],
  },
  template_kwargs: {},
  empty_system_policy: "omit",
  resolutions: [
    { field: "training.load_dtype", requested: "auto", resolved: "bfloat16", reason: "프로필 preset이 TRL 기본 float32 대신 bfloat16을 명시합니다." },
    { field: "model.loading_scope", requested: "auto_verified", resolved: "full_checkpoint", reason: "TRL 문자열 진입은 Qwen3_5ForConditionalGeneration을 로드하므로 vision tower가 상주합니다." },
    { field: "training.linear_attention_kernel", requested: "auto", resolved: "torch_fallback", reason: "환경 프로필에 flash-linear-attention이 없습니다." },
  ],
};

const promptHistogram: Array<[number, number, number]> = [
  [16, 32, 310], [32, 48, 1020], [48, 64, 1105], [64, 80, 760], [80, 96, 520], [96, 112, 340],
  [112, 128, 230], [128, 144, 170], [144, 160, 95], [160, 176, 50], [176, 192, 26], [192, 208, 14],
  [208, 224, 8], [224, 240, 3], [240, 256, 1], [256, 272, 4],
];

export const grpoScan: Schemas["DatasetScanResult"] = {
  coverage: "complete",
  objective: "grpo",
  config: "default",
  split: "train",
  split_auto_selected: false,
  mapping_applied: preferenceMapping,
  transformation_note: "GRPO: prompt만 토큰화합니다. chosen/rejected 응답을 reward로 변환하지 않습니다.",
  rows_expected: 4_656,
  rows_seen: 4_656,
  rows_ok: 4_656,
  rows_failed: 0,
  rows_unprocessed: 0,
  shards_total: 1,
  shards_completed: 1,
  branches: [
    {
      branch: "prompt",
      stats: stats({ count: 4_656, min: 18, max: 268, maxRow: "train:2355", mean: 70.098, p50: 61, p90: 123, p95: 137, p99: 166, total: 326_378, histogram: promptHistogram }),
      top_rows: [
        { row_id: "train:2355", length: 268 },
        { row_id: "train:3169", length: 268 },
        { row_id: "train:2905", length: 266 },
        { row_id: "train:3718", length: 226 },
        { row_id: "train:3136", length: 215 },
      ],
    },
  ],
  failed_rows_sample: [],
  duplicate_rows: 2,
  context_exceeded_rows: 0,
  preprocess_key: fixtureId("preprocess-grpo"),
  artifact_id: fixtureId("artifact-grpo"),
  tokenizer_fingerprint: fixtureId("tokenizer"),
  template_fingerprint: "59a64ebb4df6",
  preprocessing_adapter: "trl_1_14_1.grpo",
  preprocessing_adapter_version: "1",
  elapsed_seconds: 6.9,
};

export const preservationVerified: Schemas["PreservationAudit"] = {
  status: "verified",
  checks: [
    { name: "full_read", passed: true, detail: "4,656 / 4,656 row, shard 1/1 EOF 확인" },
    { name: "no_length_drop", passed: true, detail: "max_length 미설정: 길이 기준 삭제·절단 없음" },
    { name: "no_split_or_concat", passed: true, detail: "packing 꺼짐: 분할·연결 없음" },
    { name: "template_content_preserved", passed: true, detail: "템플릿이 제거한 content 0 row" },
    { name: "last_batch_included", passed: true, detail: "RepeatSampler: N mod U = 0" },
    { name: "packing_safe", passed: null, detail: "packing 사용 안 함" },
    { name: "context_within_limit", passed: true, detail: "최대 prompt 268 + budget 8,192 ≤ 262,144" },
    { name: "eval_scope_consistent", passed: null, detail: "평가 데이터 미선택" },
  ],
  violations: [],
};

export const contextOk: Schemas["ContextValidation"] = {
  model_declared_max: 262_144,
  tokenizer_model_max_length: 262_144,
  tokenizer_limit_is_sentinel: false,
  backend_verified_max: null,
  effective_limit: 262_144,
  limit_source: "config.max_position_embeddings",
  max_observed_length: 268,
  exceeded_rows: 0,
  exceeded_rows_exact: true,
  status: "ok",
};

export const grpoResult: Schemas["AnalysisResult"] = {
  analysis_id: fixtureId("grpo-0001"),
  schema_version: "1.0",
  created_at: CREATED_AT,
  analysis_fingerprint: fixtureId("fp-grpo-0001"),
  estimator_version: "0.1.0",
  status: {
    scan_coverage: "complete",
    data_preservation: "verified",
    training_readiness: "conditional",
    estimate_evidence: "analytic",
    hardware_fit: "not_evaluated",
  },
  source_manifests: { model: modelManifest, dataset: datasetManifest },
  tokenizer_manifest: tokenizerManifest,
  model_inventory_summary: modelSummary,
  dataset_scan: grpoScan,
  preservation_audit: preservationVerified,
  context_validation: contextOk,
  requested_config: grpoRequest,
  resolved_config: grpoResolved,
  observed_config: null,
  compatibility_report: {
    profile_id: grpoResolved.profile_id,
    architecture_adapter: "qwen3_5_hybrid",
    support_grade: "analytic",
    readiness: "conditional",
    support: [
      { objective: "grpo", strategy: "qlora", grade: "analytic", readiness: "conditional", note: "reward 필요" },
      { objective: "dpo", strategy: "qlora", grade: "analytic", readiness: "ready", note: "" },
    ],
    blockers: [],
    warnings: [],
    not_effective: [],
  },
  batch_plan: {
    objective: "grpo",
    unit: "completions",
    microbatch: 1,
    accumulation: 4,
    effective_batch: 4,
    pad_to_multiple_of: null,
    padding_side: "left",
    sampler: { kind: "repeat_sampler", seed: 42, drop_last: false, covers_all_rows: true, dropped_rows: 0, note: "U=1이므로 N mod U = 0" },
    worst_case: grpoScenarios[3]?.batch_shape ?? grpoScenarios[0]!.batch_shape,
    sampler_max: null,
    scenarios: grpoScenarios.map((s) => s.batch_shape),
    grpo: {
      unique_prompts_per_generation: 1,
      num_generations: 4,
      live_sequences: 4,
      update_microbatch: 1,
      accumulation: 4,
      generation_batch_size: 4,
      steps_per_generation: 4,
      num_iterations: 1,
      completion_budgets: [...BUDGETS],
      max_prompt_length: PROMPT_MAX,
    },
    issues: [],
    batch_key: fixtureId("batch-grpo"),
  },
  memory: {
    evidence_level: "analytic",
    scenarios: grpoScenarios,
    primary_scenario_id: null,
    assumptions: ASSUMPTIONS,
  },
  host_ram_estimate: {
    bytes_low: 18 * GIB,
    bytes_high: 26 * GIB,
    items: [
      { name: "model loading staging (largest tensor)", bytes_low: 2_034_237_440, bytes_high: 6_102_712_320, evidence: "analytic", note: "embed_tokens bf16, CPU 변환 시 fp32 추가" },
      { name: "dataloader · tokenizer", bytes_low: 2 * GIB, bytes_high: 4 * GIB, evidence: "assumption", note: null },
    ],
  },
  analysis_ram_estimate: null,
  disk_estimate: null,
  planning_margin_policy: { min_bytes: 2 * GIB, fraction: 0.15 },
  hardware_fit: null,
  assumptions: ASSUMPTIONS,
  unknown_components: [],
  excluded_components: [
    { name: "reward footprint", reason: "reward가 지정되지 않아 reward 모델·함수 메모리를 계산에서 제외했습니다.", code: "GRPO_REWARD_UNSPECIFIED" },
    { name: "evaluation", reason: "평가 단계 미포함 (scope.include_evaluation=false)", code: null },
  ],
  warnings: [
    issue("GRPO_REWARD_UNSPECIFIED", "warning", "reward가 지정되지 않았습니다. reward footprint를 포함하지 않은 조건부 결과입니다.", { stage: "estimating" }),
    issue("GRPO_BUDGET_UNSPECIFIED", "info", "completion budget이 없어 1,024 / 2,048 / 4,096 / 8,192 token 예산별로 계산했습니다.", { stage: "estimating" }),
  ],
  errors: [],
  needs_input: null,
  profile_id: grpoResolved.profile_id,
  dependency_lock_digest: grpoResolved.dependency_lock_digest,
  measurement_scope: {
    measured: false,
    phases_included: ["MODEL_LOAD_AND_QUANTIZE", "ROLLOUT_PREFILL_AND_DECODE", "POLICY_FORWARD_BACKWARD", "OPTIMIZER_STEP"],
    phases_excluded: ["REWARD", "EVALUATION", "CHECKPOINT_SAVE_OR_CONSOLIDATE"],
    note: "GPU 실측 없음. 정적 분석(analytic) 결과입니다.",
  },
};

/** The same analysis recomputed with an explicit 2,048-token budget (one primary scenario). */
export const grpoExplicitBudgetResult: Schemas["AnalysisResult"] = {
  ...grpoResult,
  analysis_fingerprint: fixtureId("fp-grpo-0001-b2048"),
  requested_config: exampleRequest({ grpoBudgetMode: "explicit", grpoCompletionBudget: "2048" }),
  memory: {
    evidence_level: "analytic",
    scenarios: [{ ...grpoScenarios[1]!, scenario_id: "default" }],
    primary_scenario_id: "default",
    assumptions: ASSUMPTIONS,
  },
  hardware_fit: NOT_EVALUATED_FIT,
  warnings: [grpoResult.warnings![0]!],
};
