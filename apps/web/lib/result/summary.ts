// Pure selectors from an AnalysisResult to what the summary card shows. No memory formulas are
// recomputed here (plan.md §13.2): numbers come from the server, nulls carry a reason.
import type {
  AnalysisResult,
  CapacityRecommendation,
  DeviceEstimate,
  ExcludedComponent,
  HardwareFitResult,
  Issue,
  ScenarioEstimate,
  UnknownComponent,
} from "@/lib/api/types";

export interface BudgetRow {
  scenarioId: string;
  label: string;
  budget: number | null;
  low: number | null;
  high: number | null;
  recommended: number | null;
  fit: HardwareFitResult | null;
  nullReason: string | null;
}

export interface ResultSummary {
  kind: "single" | "budgets" | "unavailable";
  scenario: ScenarioEstimate | null;
  device: DeviceEstimate | null;
  peakLow: number | null;
  peakHigh: number | null;
  peakNullReason: string | null;
  floor: number | null;
  recommendation: CapacityRecommendation | null;
  recommendationNullReason: string | null;
  hostRamLow: number | null;
  hostRamHigh: number | null;
  hostRamNullReason: string | null;
  fit: HardwareFitResult | null;
  budgets: BudgetRow[];
  budgetRange: { low: number; high: number; minBudget: number | null; maxBudget: number | null } | null;
  rewardExcluded: boolean;
  excluded: ExcludedComponent[];
  unknown: UnknownComponent[];
  issues: Issue[];
}

export function primaryScenario(result: AnalysisResult): ScenarioEstimate | null {
  const memory = result.memory;
  if (!memory) return null;
  if (memory.primary_scenario_id == null) return null;
  return memory.scenarios.find((s) => s.scenario_id === memory.primary_scenario_id) ?? null;
}

/** Scenario for the detail tabs: requested id, else primary, else the first one. */
export function scenarioById(result: AnalysisResult | null, id: string | null): ScenarioEstimate | null {
  const scenarios = result?.memory?.scenarios ?? [];
  return scenarios.find((s) => s.scenario_id === id) ?? (result ? primaryScenario(result) : null) ?? scenarios[0] ?? null;
}

export function budgetOf(scenario: ScenarioEstimate): number | null {
  const fromParams = scenario.params?.completion_budget;
  if (typeof fromParams === "number") return fromParams;
  return scenario.batch_shape.completion_length ?? null;
}

function unknownReason(unknown: UnknownComponent[]): string {
  if (unknown.length === 0) return "근거가 부족해 산정하지 않았습니다.";
  const first = unknown[0]!;
  const more = unknown.length > 1 ? ` 외 ${unknown.length - 1}건` : "";
  return `${first.name}${more}: ${first.reason}`;
}

function memoryMissingReason(result: AnalysisResult): string {
  const blocker = result.compatibility_report?.blockers?.[0];
  if (blocker) return blocker.user_message;
  const coverage = result.dataset_scan?.coverage ?? result.status?.scan_coverage;
  if (coverage && coverage !== "complete") {
    return "전체 데이터 스캔이 끝나지 않아 메모리를 산정하지 않았습니다.";
  }
  const error = result.errors?.[0];
  if (error) return error.user_message;
  return "메모리 산정 단계에 도달하지 않았습니다.";
}

function dedupe<T>(items: T[], key: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const k = key(item);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function budgetRow(scenario: ScenarioEstimate): BudgetRow {
  const dev = scenario.devices[0] ?? null;
  const high = dev?.scenario_high_bytes ?? null;
  return {
    scenarioId: scenario.scenario_id,
    label: scenario.label,
    budget: budgetOf(scenario),
    low: dev?.scenario_low_bytes ?? null,
    high,
    recommended: scenario.recommendation?.recommended_application_capacity_bytes ?? null,
    fit: scenario.hardware_fit ?? null,
    nullReason: high == null ? unknownReason(dev?.unknown_components ?? []) : null,
  };
}

export function summarize(result: AnalysisResult): ResultSummary {
  const memory = result.memory ?? null;
  const scenarios = memory?.scenarios ?? [];
  const primary = primaryScenario(result);
  const isBudgets = memory != null && memory.primary_scenario_id == null && scenarios.length > 0;
  const scenario = primary ?? (isBudgets ? null : scenarios[0] ?? null);
  const device = scenario?.devices[0] ?? null;

  const excluded = dedupe(
    [...(result.excluded_components ?? []), ...scenarios.flatMap((s) => s.excluded_components ?? [])],
    (e) => `${e.name}|${e.reason}`,
  );
  const unknown = dedupe(
    [...(result.unknown_components ?? []), ...(device?.unknown_components ?? [])],
    (u) => `${u.name}|${u.reason}`,
  );
  const issues = [...(result.errors ?? []), ...(result.warnings ?? [])];
  const rewardExcluded =
    excluded.some((e) => e.code === "GRPO_REWARD_UNSPECIFIED") ||
    issues.some((i) => i.code === "GRPO_REWARD_UNSPECIFIED");

  const budgets = isBudgets ? scenarios.map(budgetRow).sort((a, b) => (a.budget ?? 0) - (b.budget ?? 0)) : [];
  const knownBudgets = budgets.filter((b) => b.low != null && b.high != null);
  const budgetRange =
    isBudgets && knownBudgets.length === budgets.length && budgets.length > 0
      ? {
          low: Math.min(...knownBudgets.map((b) => b.low ?? 0)),
          high: Math.max(...knownBudgets.map((b) => b.high ?? 0)),
          minBudget: budgets[0]?.budget ?? null,
          maxBudget: budgets.at(-1)?.budget ?? null,
        }
      : null;

  const peakLow = device?.scenario_low_bytes ?? null;
  const peakHigh = device?.scenario_high_bytes ?? null;
  let peakNullReason: string | null = null;
  if (!memory) peakNullReason = memoryMissingReason(result);
  else if (!isBudgets && (peakLow == null || peakHigh == null)) peakNullReason = unknownReason(unknown);

  const recommendation = scenario?.recommendation ?? null;
  const recommendationNullReason =
    recommendation != null || isBudgets
      ? null
      : peakNullReason ?? "예상 피크 상한이 없어 권장 용량을 만들지 않습니다.";

  const host = result.host_ram_estimate ?? null;
  const hostRamNullReason =
    host && host.bytes_low != null && host.bytes_high != null
      ? null
      : host
        ? "학습 노드 RAM을 산정할 근거가 부족합니다."
        : memory
          ? "학습 RAM 추정이 결과에 없습니다."
          : peakNullReason;

  return {
    kind: memory == null ? "unavailable" : isBudgets ? "budgets" : "single",
    scenario,
    device,
    peakLow,
    peakHigh,
    peakNullReason,
    floor: device?.known_floor_bytes ?? null,
    recommendation,
    recommendationNullReason,
    hostRamLow: host?.bytes_low ?? null,
    hostRamHigh: host?.bytes_high ?? null,
    hostRamNullReason,
    fit: scenario?.hardware_fit ?? result.hardware_fit ?? null,
    budgets,
    budgetRange,
    rewardExcluded,
    excluded,
    unknown,
    issues,
  };
}

/** True when the request asked for a GPU (preset or custom) so a usage gauge makes sense. */
export function hardwareSelected(result: AnalysisResult): boolean {
  const mode = result.requested_config.hardware?.mode ?? "capacity_only";
  return mode !== "capacity_only";
}
