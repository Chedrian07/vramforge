// Builders for contract-shaped fixtures (tests and NEXT_PUBLIC_VF_DEV_MOCKS=1 dev mode only).
// Every id produced here starts with FIXTURE_PREFIX so scripts/check-no-mocks.mjs can prove that
// no fixture value reached a production bundle.
import type { Schemas } from "@/lib/api/types";

export const FIXTURE_PREFIX = "vf-fixture";
export const GIB = 1_073_741_824;
export const MIB = 1_048_576;

type Phase = Schemas["Phase"];
type Category = Schemas["AllocationCategory"];
type Evidence = Schemas["Evidence"];

export interface ItemSpec {
  name: string;
  category: Category;
  low: number | null;
  high: number | null;
  evidence: Evidence;
  note?: string;
}

const RESIDENT: ReadonlySet<Category> = new Set<Category>([
  "weights_base",
  "weights_adapter",
  "weights_other_models",
  "gradients",
  "optimizer_states",
  "master_weights",
]);

export function fixtureId(name: string): string {
  return `${FIXTURE_PREFIX}-${name}`;
}

function sum(values: Array<number | null>): number | null {
  let total = 0;
  for (const v of values) {
    if (v == null) return null;
    total += v;
  }
  return total;
}

export function breakdown(timepoint: string, phase: Phase, items: ItemSpec[]): Schemas["PeakBreakdown"] {
  const byCategory: Record<string, number> = {};
  for (const item of items) {
    if (item.high != null) byCategory[item.category] = (byCategory[item.category] ?? 0) + item.high;
  }
  return {
    timepoint,
    phase,
    items: items.map((i) => ({
      name: i.name,
      category: i.category,
      bytes_low: i.low,
      bytes_high: i.high,
      evidence: i.evidence,
      note: i.note ?? null,
    })),
    by_category_high: byCategory,
    total_low: sum(items.map((i) => i.low)),
    total_high: sum(items.map((i) => i.high)),
  };
}

export function knownFloor(items: ItemSpec[]): number {
  return items
    .filter((i) => RESIDENT.has(i.category) && i.low != null && i.low === i.high)
    .reduce((acc, i) => acc + (i.high ?? 0), 0);
}

export interface PhaseSpec {
  phase: Phase;
  included: boolean;
  excludedReason?: string;
  timepoint?: string;
  low?: number | null;
  high?: number | null;
  floor?: number | null;
  unknown?: string[];
}

export function phasePeak(spec: PhaseSpec): Schemas["PhasePeak"] {
  return {
    phase: spec.phase,
    included: spec.included,
    excluded_reason: spec.excludedReason ?? null,
    peak_timepoint: spec.timepoint ?? null,
    bytes_low: spec.low ?? null,
    bytes_high: spec.high ?? null,
    known_floor_bytes: spec.floor ?? null,
    unknown_components: spec.unknown ?? [],
  };
}

export function device(
  peak: Schemas["PeakBreakdown"],
  items: ItemSpec[],
  phases: PhaseSpec[],
  unknown: Schemas["UnknownComponent"][] = [],
): Schemas["DeviceEstimate"] {
  const floor = knownFloor(items);
  return {
    device: "cuda:0",
    phases: phases.map(phasePeak),
    timepoints: phases
      .filter((p) => p.included && p.timepoint)
      .map((p) => ({
        timepoint: p.timepoint ?? "",
        phase: p.phase,
        bytes_low: p.low ?? null,
        bytes_high: p.high ?? null,
        known_floor_bytes: p.floor ?? floor,
        unknown: p.unknown ?? [],
      })),
    peak_phase: peak.phase,
    peak_timepoint: peak.timepoint,
    known_floor_bytes: floor,
    scenario_low_bytes: peak.total_low,
    scenario_high_bytes: peak.total_high,
    peak_breakdown: peak,
    unknown_components: unknown,
  };
}

/** Mirrors the server's margin policy (plan.md §10.2) so fixture numbers stay consistent. */
export function recommendation(high: number | null, externalReserved = 0): Schemas["CapacityRecommendation"] | null {
  if (high == null) return null;
  const margin = Math.max(2 * GIB, Math.round(high * 0.15));
  const recommended = high + margin;
  return {
    planning_margin_bytes: margin,
    recommended_application_capacity_bytes: recommended,
    external_reserved_bytes: externalReserved,
    required_total_device_capacity_bytes: recommended + externalReserved,
    policy: { min_bytes: 2 * GIB, fraction: 0.15 },
  };
}

export const NOT_EVALUATED_FIT: Schemas["HardwareFitResult"] = {
  status: "not_evaluated",
  reason: "not_evaluated",
  message: "하드웨어를 선택하지 않아 용량만 표시합니다.",
  capacity_bytes: null,
  utilization_ratio: null,
};

export function batchShape(spec: {
  name: string;
  objective: Schemas["Objective"];
  rows: number;
  sequences: number;
  padded: number;
  logits: number;
  prompt?: number | null;
  completion?: number | null;
  rowIds: string[];
  description: string;
}): Schemas["BatchShape"] {
  return {
    name: spec.name,
    objective: spec.objective,
    rows_per_microbatch: spec.rows,
    sequences_per_forward: spec.sequences,
    padded_length: spec.padded,
    token_slots: spec.sequences * spec.padded,
    logits_positions: spec.logits,
    prompt_length: spec.prompt ?? null,
    completion_length: spec.completion ?? null,
    source_row_ids: spec.rowIds,
    description: spec.description,
  };
}

export function stats(spec: {
  count: number;
  min: number;
  max: number;
  maxRow: string;
  mean: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  total: number;
  histogram: Array<[number, number, number]>;
}): Schemas["LengthStats"] {
  return {
    count: spec.count,
    min: spec.min,
    max: spec.max,
    max_row_id: spec.maxRow,
    mean: spec.mean,
    p50: spec.p50,
    p90: spec.p90,
    p95: spec.p95,
    p99: spec.p99,
    total_tokens: spec.total,
    quantiles_exact: true,
    histogram: spec.histogram.map(([lo, hi, count]) => ({ lo, hi, count })),
  };
}
