"use client";

import { Badge } from "@/components/ui/primitives";
import type { DatasetScanResult, StatusAxes } from "@/lib/api/types";
import { DATA_PRESERVATION_LABEL, EVIDENCE_LEVEL_LABEL, HARDWARE_FIT_LABEL, READINESS_LABEL } from "@/lib/format/labels";
import { scanCoverageDisplay } from "@/lib/result/scan";
import { EVIDENCE_TONE, FIT_TONE, PRESERVATION_TONE, READINESS_TONE } from "@/lib/result/status";

const EMPTY: StatusAxes = {
  scan_coverage: "not_started",
  data_preservation: "pending",
  training_readiness: null,
  estimate_evidence: null,
  hardware_fit: "not_evaluated",
};

/**
 * The five independent result axes (plan.md §12.1). Never collapsed into one success badge:
 * a complete scan can still be conditional, preserved data can still exceed the context.
 */
export function StatusBadges({
  axes,
  scan = null,
}: {
  axes: StatusAxes | null | undefined;
  /** Row counts of the scan: failed rows keep the coverage badge from saying "전체 완료". */
  scan?: Pick<DatasetScanResult, "rows_seen" | "rows_failed"> | null;
}) {
  const a = axes ?? EMPTY;
  const coverage = scanCoverageDisplay(a.scan_coverage, { rowsSeen: scan?.rows_seen, rowsFailed: scan?.rows_failed });
  const rows = [
    { label: "스캔 범위", text: coverage.text, tone: coverage.tone },
    { label: "데이터 보존", text: DATA_PRESERVATION_LABEL[a.data_preservation], tone: PRESERVATION_TONE[a.data_preservation] },
    {
      label: "학습 준비",
      text: a.training_readiness ? READINESS_LABEL[a.training_readiness] : "판정 전",
      tone: a.training_readiness ? READINESS_TONE[a.training_readiness] : ("neutral" as const),
    },
    {
      label: "추정 근거",
      text: a.estimate_evidence ? EVIDENCE_LEVEL_LABEL[a.estimate_evidence] : "산정 전",
      tone: a.estimate_evidence ? EVIDENCE_TONE[a.estimate_evidence] : ("neutral" as const),
    },
    { label: "GPU 적합", text: HARDWARE_FIT_LABEL[a.hardware_fit], tone: FIT_TONE[a.hardware_fit] },
  ];
  return (
    <ul aria-label="결과 상태" className="grid grid-cols-1 gap-x-3 gap-y-1.5 sm:grid-cols-2">
      {rows.map((row) => (
        <li key={row.label} className="flex min-w-0 items-center justify-between gap-2 text-[13px]">
          <span className="text-muted">{row.label}</span>
          <Badge tone={row.tone}>{row.text}</Badge>
        </li>
      ))}
    </ul>
  );
}
