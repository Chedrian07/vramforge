// DPO (24 GiB GPU preset, expected over capacity) and SFT (capacity only) fixtures.
// Token statistics are the verified golden values; memory items are illustrative.
import type { Schemas } from "@/lib/api/types";

import { STATIC_ITEMS, contextOk, grpoResolved, grpoResult, preservationVerified } from "./analysis-grpo";
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
import { ASSUMPTIONS, VOCAB, exampleRequest, preferenceMapping } from "./common";

export const RTX_24GB = { id: "vf-fixture-gpu-24gb", name: "24 GiB GPU (fixture)", total_bytes: 24 * GIB, note: "" };

function overhead(items: ItemSpec[]): ItemSpec[] {
  const ctx: ItemSpec = { name: "CUDA context · library handles", category: "non_framework", low: 512 * MIB, high: GIB, evidence: "assumption" };
  const ws: ItemSpec = { name: "cuBLAS / SDPA workspace", category: "workspace", low: 256 * MIB, high: 512 * MIB, evidence: "assumption" };
  const all = [...items, ctx, ws];
  const lo = all.reduce((a, i) => a + (i.low ?? 0), 0);
  const hi = all.reduce((a, i) => a + (i.high ?? 0), 0);
  return [...all, { name: "allocator slack", category: "allocator_slack", low: Math.round(lo * 0.05), high: Math.round(hi * 0.1), evidence: "assumption" }];
}

function phasesFor(peak: Schemas["PeakBreakdown"], objective: "sft" | "dpo"): PhaseSpec[] {
  const statics = STATIC_ITEMS.reduce((a, i) => a + (i.high ?? 0), 0);
  return [
    { phase: "MODEL_LOAD_AND_QUANTIZE", included: true, timepoint: "MODEL_LOAD_AND_QUANTIZE:quantize", low: statics - 362_200_064 + 106_954_752, high: statics - 362_200_064 + 106_954_752 + GIB },
    objective === "dpo"
      ? { phase: "REFERENCE_PRECOMPUTE", included: false, excludedReason: "reference 전략 frozen_base_switch: 사전 계산 없음" }
      : { phase: "REFERENCE_PRECOMPUTE", included: false, excludedReason: "SFT에는 해당 없음" },
    { phase: "ROLLOUT_PREFILL_AND_DECODE", included: false, excludedReason: `${objective.toUpperCase()}에는 rollout 없음` },
    { phase: "REWARD", included: false, excludedReason: `${objective.toUpperCase()}에는 reward 없음` },
    { phase: "POLICY_FORWARD_BACKWARD", included: true, timepoint: peak.timepoint, low: peak.total_low, high: peak.total_high },
    { phase: "OPTIMIZER_STEP", included: true, timepoint: "OPTIMIZER_STEP:step", low: statics + 768 * MIB, high: statics + 1536 * MIB },
    { phase: "WEIGHT_SYNC", included: false, excludedReason: "해당 없음" },
    { phase: "EVALUATION", included: false, excludedReason: "평가 미포함 (scope.include_evaluation=false)" },
    { phase: "CHECKPOINT_SAVE_OR_CONSOLIDATE", included: false, excludedReason: "체크포인트 저장 미포함" },
  ];
}

// ---------------------------------------------------------------- DPO (pair max 2,272 tokens, N = 2)

const T_DPO = 2_272;
const dpoItems = overhead([
  ...STATIC_ITEMS,
  { name: "checkpoint boundary activations (chosen + rejected)", category: "saved_activations", low: 32 * 2 * 4096 * T_DPO * 2, high: 32 * 2 * 4096 * T_DPO * 2, evidence: "analytic" },
  { name: "recompute working set (1 decoder layer)", category: "recompute_working_set", low: 48 * 4096 * T_DPO * 2, high: 72 * 4096 * T_DPO * 2, evidence: "analytic" },
  { name: "policy logits (fp32)", category: "logits_and_loss", low: 2 * T_DPO * VOCAB * 4, high: 2 * T_DPO * VOCAB * 4, evidence: "analytic", note: "N × T × V × 4 (N = 2 × pairs)" },
  { name: "reference logits (fp32, adapter off)", category: "logits_and_loss", low: 2 * T_DPO * VOCAB * 4, high: 2 * T_DPO * VOCAB * 4, evidence: "analytic", note: "policy logits와 동시 생존" },
  { name: "metric boolean-index copy", category: "logits_and_loss", low: 4_006 * VOCAB * 4, high: 4_006 * VOCAB * 4, evidence: "analytic" },
]);
const dpoPeak = breakdown("POLICY_FORWARD_BACKWARD:loss", "POLICY_FORWARD_BACKWARD", dpoItems);
const dpoRecommendation = recommendation(dpoPeak.total_high);
const dpoUtilization = (dpoRecommendation?.recommended_application_capacity_bytes ?? 0) / RTX_24GB.total_bytes;

export const dpoFit: Schemas["HardwareFitResult"] = {
  status: "exceeds",
  reason: "high_exceeds_capacity",
  message: "예상 피크 상한이 가용 VRAM을 넘습니다. 실측 또는 설정 검토가 필요합니다.",
  capacity_bytes: RTX_24GB.total_bytes,
  utilization_ratio: dpoUtilization,
};

const dpoScenario: Schemas["ScenarioEstimate"] = {
  scenario_id: "default",
  label: "worst-case batch (rows 2355 + 3169 lengths)",
  params: {},
  batch_shape: batchShape({ name: "worst_case", objective: "dpo", rows: 1, sequences: 2, padded: T_DPO, logits: 2 * T_DPO, prompt: 268, completion: 2_004, rowIds: ["train:2355", "train:3169"], description: "pair 최대 길이 2,272" }),
  devices: [device(dpoPeak, dpoItems, phasesFor(dpoPeak, "dpo"))],
  recommendation: dpoRecommendation,
  hardware_fit: dpoFit,
  excluded_components: [],
  timepoints: [],
  allocations: [
    { name: "policy logits (fp32)", category: "logits_and_loss", device: "cuda:0", shape_expression: "N × T × V", dims: { N: 2, T: T_DPO, V: VOCAB }, dtype: "float32", count: 1, bytes_low: 2 * T_DPO * VOCAB * 4, bytes_high: 2 * T_DPO * VOCAB * 4, live_at: ["POLICY_FORWARD_BACKWARD:loss"], saved_for_backward: true, recompute_group: null, storage_alias_group: null, evidence: "analytic", formula_ref: "methodology.md#dpo-logits", note: null },
  ],
};

const pairHistogram: Array<[number, number, number]> = [
  [0, 128, 402], [128, 256, 2_571], [256, 384, 1_181], [384, 512, 323], [512, 1024, 151], [1024, 2048, 19], [2048, 2304, 9],
];

export const dpoRequest = exampleRequest({ objective: "dpo", microbatch: "1", accumulation: "8", hardwareMode: "gpu_preset", gpuPresetId: RTX_24GB.id });
dpoRequest.hardware = { ...dpoRequest.hardware!, device_total_bytes: RTX_24GB.total_bytes };

export const dpoResult: Schemas["AnalysisResult"] = {
  ...grpoResult,
  analysis_id: fixtureId("dpo-0002"),
  analysis_fingerprint: fixtureId("fp-dpo-0002"),
  status: { scan_coverage: "complete", data_preservation: "verified", training_readiness: "ready", estimate_evidence: "analytic", hardware_fit: "exceeds" },
  dataset_scan: {
    ...grpoResult.dataset_scan!,
    objective: "dpo",
    transformation_note: "DPO: prompt + chosen, prompt + rejected 두 branch를 각각 그대로 사용합니다.",
    mapping_applied: preferenceMapping,
    branches: [
      { branch: "chosen_sequence", stats: stats({ count: 4_656, min: 51, max: 2_272, maxRow: "train:2355", mean: 216.317, p50: 207, p90: 316, p95: 355, p99: 455, total: 1_007_173, histogram: pairHistogram }), top_rows: [{ row_id: "train:2355", length: 2_272 }, { row_id: "train:2905", length: 2_265 }, { row_id: "train:2200", length: 2_091 }] },
      { branch: "rejected_sequence", stats: stats({ count: 4_656, min: 49, max: 2_272, maxRow: "train:3169", mean: 180.834, p50: 166, p90: 268, p95: 300, p99: 399, total: 841_963, histogram: pairHistogram }), top_rows: [{ row_id: "train:3169", length: 2_272 }, { row_id: "train:2355", length: 2_270 }, { row_id: "train:2905", length: 2_267 }] },
      { branch: "pair_max", stats: stats({ count: 4_656, min: 53, max: 2_272, maxRow: "train:2355", mean: 221.071, p50: 208, p90: 319, p95: 360, p99: 496, total: 1_029_308, histogram: pairHistogram }), top_rows: [{ row_id: "train:2355", length: 2_272 }, { row_id: "train:3169", length: 2_272 }] },
    ],
    preprocess_key: fixtureId("preprocess-dpo"),
    preprocessing_adapter: "trl_1_14_1.dpo",
  },
  preservation_audit: preservationVerified,
  context_validation: { ...contextOk, max_observed_length: 2_272 },
  requested_config: dpoRequest,
  resolved_config: {
    ...grpoResolved,
    objective: "dpo",
    trainer_adapter: "dpo",
    preprocessing_adapter: "trl_1_14_1.dpo",
    accumulation: 8,
    loss_path: "dpo_sigmoid_fp32_logits",
    grpo: null,
    dpo: { reference_strategy: "frozen_base_switch", beta: 0.1, loss_type: "sigmoid", precompute_batch_size: null, sync_ref_model: false },
  },
  compatibility_report: { ...grpoResult.compatibility_report!, readiness: "ready" },
  batch_plan: null,
  memory: { evidence_level: "analytic", scenarios: [dpoScenario], primary_scenario_id: "default", assumptions: ASSUMPTIONS },
  hardware_fit: dpoFit,
  excluded_components: [{ name: "evaluation", reason: "평가 단계 미포함 (scope.include_evaluation=false)", code: null }],
  warnings: [],
  measurement_scope: { measured: false, phases_included: ["MODEL_LOAD_AND_QUANTIZE", "POLICY_FORWARD_BACKWARD", "OPTIMIZER_STEP"], phases_excluded: ["EVALUATION", "CHECKPOINT_SAVE_OR_CONSOLIDATE"], note: "GPU 실측 없음. 정적 분석(analytic) 결과입니다." },
};

// ---------------------------------------------------------------- SFT (capacity only)

const T_SFT = 2_272;
const sftItems = overhead([
  ...STATIC_ITEMS,
  { name: "checkpoint boundary activations", category: "saved_activations", low: 32 * 2 * 4096 * T_SFT, high: 32 * 2 * 4096 * T_SFT, evidence: "analytic" },
  { name: "recompute working set (1 decoder layer)", category: "recompute_working_set", low: 48 * 4096 * T_SFT, high: 72 * 4096 * T_SFT, evidence: "analytic" },
  { name: "chunked_nll hidden copy", category: "logits_and_loss", low: (T_SFT - 1) * 4096 * 2, high: (T_SFT - 1) * 4096 * 2, evidence: "analytic" },
  { name: "chunked_nll chunk (256 × V)", category: "logits_and_loss", low: 256 * VOCAB * 16, high: 256 * VOCAB * 18, evidence: "analytic", note: "CPU 실측 15.0–16.5 B/elem, 상한 18 B" },
]);
const sftPeak = breakdown("POLICY_FORWARD_BACKWARD:loss", "POLICY_FORWARD_BACKWARD", sftItems);

export const sftResult: Schemas["AnalysisResult"] = {
  ...dpoResult,
  analysis_id: fixtureId("sft-0003"),
  analysis_fingerprint: fixtureId("fp-sft-0003"),
  status: { scan_coverage: "complete", data_preservation: "verified", training_readiness: "ready", estimate_evidence: "analytic", hardware_fit: "not_evaluated" },
  requested_config: exampleRequest({ objective: "sft", microbatch: "1", accumulation: "8" }),
  dataset_scan: {
    ...dpoResult.dataset_scan!,
    objective: "sft",
    transformation_note: "SFT: chosen 응답만 학습합니다. rejected는 사용하지 않습니다 (데이터 변환).",
    branches: [
      { branch: "sequence", stats: stats({ count: 4_656, min: 51, max: 2_272, maxRow: "train:2355", mean: 216.317, p50: 207, p90: 316, p95: 355, p99: 455, total: 1_007_173, histogram: pairHistogram }), top_rows: [{ row_id: "train:2355", length: 2_272 }, { row_id: "train:2905", length: 2_265 }] },
      { branch: "loss_tokens", stats: stats({ count: 4_656, min: 17, max: 2_004, maxRow: "train:2355", mean: 146.219, p50: 134, p90: 230, p95: 264, p99: 356, total: 680_795, histogram: pairHistogram }), top_rows: [{ row_id: "train:2355", length: 2_004 }] },
    ],
    preprocess_key: fixtureId("preprocess-sft"),
    preprocessing_adapter: "trl_1_14_1.sft",
  },
  resolved_config: { ...grpoResolved, objective: "sft", trainer_adapter: "sft", accumulation: 8, loss_path: "chunked_nll", grpo: null, dpo: null },
  memory: {
    evidence_level: "analytic",
    scenarios: [
      {
        scenario_id: "default",
        label: "worst-case batch (row 2355)",
        params: {},
        batch_shape: batchShape({ name: "worst_case", objective: "sft", rows: 1, sequences: 1, padded: T_SFT, logits: 2_004, rowIds: ["train:2355"], description: "최대 길이 2,272" }),
        devices: [device(sftPeak, sftItems, phasesFor(sftPeak, "sft"))],
        recommendation: recommendation(sftPeak.total_high),
        hardware_fit: NOT_EVALUATED_FIT,
        excluded_components: [],
        timepoints: [],
        allocations: [],
      },
    ],
    primary_scenario_id: "default",
    assumptions: ASSUMPTIONS,
  },
  hardware_fit: NOT_EVALUATED_FIT,
};
