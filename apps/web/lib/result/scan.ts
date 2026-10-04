// Display rules for scan counts that the server may not know (plan.md §7.5, §7.7): an unknown
// count stays "산정 불가" with its reason and is never shown as 0, a lower bound says so, and a
// scan with failed rows is never called "전체 완료".
import type { ScanCoverage } from "@/lib/api/types";
import { formatCount } from "@/lib/format/bytes";
import { SCAN_COVERAGE_LABEL } from "@/lib/format/labels";

import { SCAN_TONE, type Tone } from "./status";

/** Rows over the context limit cannot be counted without a known limit. */
export const CONTEXT_LIMIT_UNKNOWN = "산정 불가 (context 상한 미상)";

/** Count of rows over the context limit: null = no known limit; `exact === false` = lower bound. */
export function exceededRowsText(count: number | null | undefined, exact = true): string {
  if (count == null) return CONTEXT_LIMIT_UNKNOWN;
  const text = formatCount(count) ?? String(count);
  return exact ? text : `${text}개 이상`;
}

export interface ScanCounts {
  rowsSeen?: number | null;
  rowsFailed?: number | null;
}

/**
 * Scan coverage badge text. With failed rows the badge shows how many rows were read and how
 * many failed ("읽음 N · 실패 M", prefixed with "부분" when the scan stopped early) instead of the
 * coverage word alone: those rows are missing from the statistics.
 */
export function scanCoverageDisplay(coverage: ScanCoverage, counts: ScanCounts = {}): { text: string; tone: Tone } {
  const failed = counts.rowsFailed ?? 0;
  if (failed > 0 && (coverage === "complete" || coverage === "partial")) {
    const read = counts.rowsSeen != null ? `읽음 ${formatCount(counts.rowsSeen)} · ` : "";
    return { text: `${coverage === "partial" ? "부분 · " : ""}${read}실패 ${formatCount(failed)}`, tone: "warn" };
  }
  return { text: SCAN_COVERAGE_LABEL[coverage], tone: SCAN_TONE[coverage] };
}

export function isScanCoverage(value: unknown): value is ScanCoverage {
  return typeof value === "string" && value in SCAN_COVERAGE_LABEL;
}
