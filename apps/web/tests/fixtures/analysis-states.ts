// Job/result states other than a clean completion: partial, failed, needs_input, cancelled,
// unknown peak. Used by UI tests and the dev mock mode.
import type { Schemas } from "@/lib/api/types";

import { dpoResult, sftResult } from "./analysis-dpo-sft";
import { grpoResult } from "./analysis-grpo";
import { GIB, MIB, fixtureId } from "./builders";
import { CREATED_AT, exampleRequest, issue, preferenceMapping } from "./common";

export function status(
  result: Schemas["AnalysisResult"] | null,
  jobStatus: Schemas["JobStatus"],
  extra: Partial<Schemas["AnalysisStatus"]> = {},
): Schemas["AnalysisStatus"] {
  return {
    analysis_id: result?.analysis_id ?? fixtureId("analysis"),
    status: jobStatus,
    fingerprint: result?.analysis_fingerprint ?? fixtureId("fp"),
    progress: null,
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    finished_at: CREATED_AT,
    last_event_id: 12,
    result,
    error: null,
    ...extra,
  };
}

/** Quota hit mid-scan: lengths of unread rows are unknown, so no memory estimate (plan §7.7). */
export const partialResult: Schemas["AnalysisResult"] = {
  ...grpoResult,
  analysis_id: fixtureId("partial-0004"),
  analysis_fingerprint: fixtureId("fp-partial-0004"),
  status: { scan_coverage: "partial", data_preservation: "unknown", training_readiness: null, estimate_evidence: null, hardware_fit: "not_evaluated" },
  dataset_scan: {
    ...grpoResult.dataset_scan!,
    coverage: "partial",
    rows_seen: 2_310,
    rows_ok: 2_310,
    rows_unprocessed: 2_346,
    shards_completed: 0,
    branches: (grpoResult.dataset_scan!.branches ?? []).map((b) => ({ ...b, stats: { ...b.stats, count: 2_310 } })),
  },
  preservation_audit: { status: "unknown", checks: [{ name: "full_read", passed: false, detail: "2,310 / 4,656 row에서 중단 (처리 시간 한도)" }], violations: [] },
  batch_plan: null,
  memory: null,
  host_ram_estimate: null,
  hardware_fit: null,
  excluded_components: [],
  warnings: [],
  errors: [
    issue("SCAN_PARTIAL", "error", "처리 시간 한도로 2,310 / 4,656 row까지만 확인했습니다. 전체 최대 길이를 확인하지 못해 메모리를 산정하지 않았습니다.", { stage: "tokenizing", retryable: true }),
  ],
};

export const partialStatus = status(partialResult, "PARTIAL", {
  progress: { stage: "TOKENIZING", processed_rows: 2_310, total_rows: 4_656, shard_progress: { completed: 0, total: 1 }, message_code: "scan_quota", message: "처리 한도 도달" },
  error: partialResult.errors![0]!,
});

const failedIssue = issue("TOKENIZER_REQUIRED", "error", "모델 저장소에 tokenizer 파일이 없어 데이터 길이를 계산할 수 없습니다.", { stage: "inspecting" });

export const failedStatus = status(
  {
    ...grpoResult,
    analysis_id: fixtureId("failed-0005"),
    analysis_fingerprint: fixtureId("fp-failed-0005"),
    status: { scan_coverage: "not_started", data_preservation: "pending", training_readiness: null, estimate_evidence: null, hardware_fit: "not_evaluated" },
    tokenizer_manifest: null,
    dataset_scan: null,
    preservation_audit: null,
    context_validation: null,
    resolved_config: null,
    compatibility_report: null,
    batch_plan: null,
    memory: null,
    host_ram_estimate: null,
    hardware_fit: null,
    excluded_components: [],
    warnings: [],
    errors: [failedIssue],
  },
  "FAILED",
  { error: failedIssue },
);

export const needsInputResult: Schemas["AnalysisResult"] = {
  ...grpoResult,
  analysis_id: fixtureId("needs-0006"),
  analysis_fingerprint: fixtureId("fp-needs-0006"),
  status: { scan_coverage: "not_started", data_preservation: "pending", training_readiness: null, estimate_evidence: null, hardware_fit: "not_evaluated" },
  dataset_scan: null,
  preservation_audit: null,
  batch_plan: null,
  memory: null,
  host_ram_estimate: null,
  hardware_fit: null,
  excluded_components: [],
  warnings: [],
  errors: [],
  needs_input: {
    choices: [
      { field: "dataset.split", options: ["train", "train_extra"], suggested: "train", reason: "학습 split 후보가 두 개입니다." },
      { field: "dataset.mapping", options: [], suggested: null, reason: "prompt 역할 컬럼을 하나로 정할 수 없습니다 (question, instruction)." },
    ],
    columns: ["system", "question", "instruction", "chosen", "rejected", "lang"],
    mapping_candidates: [
      preferenceMapping,
      { ...preferenceMapping, prompt: "instruction" },
    ],
  },
};

export const needsInputStatus = status(needsInputResult, "NEEDS_INPUT");

export const cancelledStatus = status(
  {
    ...partialResult,
    analysis_id: fixtureId("cancel-0007"),
    analysis_fingerprint: fixtureId("fp-cancel-0007"),
    dataset_scan: { ...partialResult.dataset_scan!, rows_seen: 1_200, rows_ok: 1_200, rows_unprocessed: 3_456 },
    errors: [issue("CANCELLED", "info", "사용자가 분석을 취소했습니다. 1,200 / 4,656 row까지 확인했습니다.")],
  },
  "CANCELLED",
  { progress: { stage: "TOKENIZING", processed_rows: 1_200, total_rows: 4_656, shard_progress: null, message_code: null, message: null } },
);

/** A size on the peak path cannot be estimated: totals are null (never 0), fit is withheld. */
export const unknownPeakResult: Schemas["AnalysisResult"] = (() => {
  const base = sftResult.memory!.scenarios[0]!;
  const dev = base.devices[0]!;
  const unknown: Schemas["UnknownComponent"] = {
    name: "linear-attention fla workspace",
    reason: "fla 커널의 chunk workspace 크기를 설명할 근거가 없어 산정하지 않았습니다.",
    phase: "POLICY_FORWARD_BACKWARD",
  };
  const items = [
    ...(dev.peak_breakdown?.items ?? []),
    { name: "linear-attention fla workspace", category: "workspace" as const, bytes_low: null, bytes_high: null, evidence: "unknown" as const, note: unknown.reason },
  ];
  return {
    ...sftResult,
    analysis_id: fixtureId("unknown-0008"),
    analysis_fingerprint: fixtureId("fp-unknown-0008"),
    status: { scan_coverage: "complete", data_preservation: "verified", training_readiness: "conditional", estimate_evidence: "analytic", hardware_fit: "unknown" },
    requested_config: exampleRequest({ objective: "sft", linearAttentionKernel: "fla", hardwareMode: "custom", hardwareTotalGiB: "48" }),
    memory: {
      evidence_level: "analytic",
      primary_scenario_id: "default",
      assumptions: [],
      scenarios: [
        {
          ...base,
          devices: [
            {
              ...dev,
              scenario_low_bytes: null,
              scenario_high_bytes: null,
              peak_breakdown: dev.peak_breakdown ? { ...dev.peak_breakdown, items, total_low: null, total_high: null } : null,
              unknown_components: [unknown],
            },
          ],
          recommendation: null,
          hardware_fit: { status: "unknown", reason: "unknown_components", message: "피크 경로에 크기 미상 항목이 있어 적합 판정을 보류합니다.", capacity_bytes: 48 * GIB, utilization_ratio: null },
        },
      ],
    },
    hardware_fit: { status: "unknown", reason: "unknown_components", message: "피크 경로에 크기 미상 항목이 있어 적합 판정을 보류합니다.", capacity_bytes: 48 * GIB, utilization_ratio: null },
    unknown_components: [unknown],
    host_ram_estimate: { bytes_low: null, bytes_high: null, items: [] },
  };
})();

/** Running job snapshot used when reconnecting after a reload. */
export const runningStatus = status(null, "TOKENIZING", {
  analysis_id: fixtureId("grpo-0001"),
  fingerprint: fixtureId("fp-grpo-0001"),
  finished_at: null,
  progress: { stage: "TOKENIZING", processed_rows: 1_536, total_rows: null, shard_progress: { completed: 0, total: null }, message_code: null, message: null },
});

export const completedGrpoStatus = status(grpoResult, "COMPLETED", {
  progress: { stage: "COMPLETED", processed_rows: 4_656, total_rows: 4_656, shard_progress: { completed: 1, total: 1 }, message_code: null, message: null },
});
export const completedDpoStatus = status(dpoResult, "COMPLETED");
export const completedSftStatus = status(sftResult, "COMPLETED");
export const unknownPeakStatus = status(unknownPeakResult, "COMPLETED");

export const BYTES = { GIB, MIB };
