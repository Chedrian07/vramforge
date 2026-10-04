import { describe, expect, it } from "vitest";

import { hardwareSelected, scenarioById, summarize } from "@/lib/result/summary";

import { dpoResult, sftResult } from "../fixtures/analysis-dpo-sft";
import { BUDGETS, grpoExplicitBudgetResult, grpoResult } from "../fixtures/analysis-grpo";
import { needsInputResult, partialResult, unknownPeakResult } from "../fixtures/analysis-states";

describe("summarize", () => {
  it("reports issues raised while estimating memory, once", () => {
    const loadBudget = { code: "LOAD_BUDGET_EXCEEDED" as const, severity: "warning" as const, retryable: false, user_message: "모델 로딩 단계에 필요한 여유가 부족합니다." };
    const s = summarize({ ...dpoResult, warnings: [...(dpoResult.warnings ?? []), loadBudget], memory: { ...dpoResult.memory!, issues: [loadBudget] } });
    expect(s.issues.filter((i) => i.code === "LOAD_BUDGET_EXCEEDED")).toHaveLength(1);
  });

  it("lists every completion budget when no explicit budget was given", () => {
    const s = summarize(grpoResult);
    expect(s.kind).toBe("budgets");
    expect(s.budgets.map((b) => b.budget)).toEqual([...BUDGETS]);
    expect(s.budgets.every((b) => b.high != null && b.recommended != null)).toBe(true);
    expect(s.budgetRange?.minBudget).toBe(1024);
    expect(s.budgetRange?.maxBudget).toBe(8192);
    expect(s.budgetRange!.high).toBeGreaterThan(s.budgetRange!.low);
    expect(s.rewardExcluded).toBe(true);
    expect(s.scenario).toBeNull();
  });

  it("gives no budget range when one budget is unknown, and says which", () => {
    const scenarios = grpoResult.memory!.scenarios.map((sc, i) =>
      i === 2 ? { ...sc, devices: [{ ...sc.devices[0]!, scenario_low_bytes: null, scenario_high_bytes: null }], recommendation: null } : sc,
    );
    const s = summarize({ ...grpoResult, memory: { ...grpoResult.memory!, scenarios } });
    expect(s.kind).toBe("budgets");
    expect(s.budgetRange).toBeNull();
    expect(s.peakNullReason).toContain(scenarios[2]!.label);
    expect(s.budgets.filter((b) => b.high != null)).toHaveLength(3);
    expect(s.budgets.find((b) => b.high == null)?.nullReason).toBeTruthy();
  });

  it("uses the primary scenario when the budget is explicit", () => {
    const s = summarize(grpoExplicitBudgetResult);
    expect(s.kind).toBe("single");
    expect(s.peakHigh).toBe(grpoResult.memory!.scenarios[1]!.devices[0]!.scenario_high_bytes);
    expect(s.recommendation?.recommended_application_capacity_bytes).toBeGreaterThan(s.peakHigh!);
  });

  it("keeps unknown peaks null with a reason instead of zero", () => {
    const s = summarize(unknownPeakResult);
    expect(s.peakLow).toBeNull();
    expect(s.peakHigh).toBeNull();
    expect(s.peakNullReason).toContain("linear-attention fla workspace");
    expect(s.recommendation).toBeNull();
    expect(s.recommendationNullReason).toBeTruthy();
    expect(s.hostRamNullReason).toBeTruthy();
  });

  it("explains a missing estimate on partial scans", () => {
    const s = summarize(partialResult);
    expect(s.kind).toBe("unavailable");
    expect(s.peakNullReason).toBe("전체 데이터 스캔이 끝나지 않아 메모리를 산정하지 않았습니다.");
  });

  it("says an analysis waiting for input stopped for a choice, not for a short scan", () => {
    const s = summarize(needsInputResult);
    expect(s.kind).toBe("unavailable");
    expect(s.peakNullReason).toContain("골라야 분석을 이어 갈 수 있어");
  });

  it("reports hardware fit only when hardware was requested", () => {
    expect(hardwareSelected(dpoResult)).toBe(true);
    expect(summarize(dpoResult).fit?.status).toBe("exceeds");
    expect(hardwareSelected(sftResult)).toBe(false);
  });

  it("picks scenarios for detail tabs", () => {
    expect(scenarioById(grpoResult, "budget_4096")?.scenario_id).toBe("budget_4096");
    expect(scenarioById(grpoResult, null)?.scenario_id).toBe("budget_1024");
    expect(scenarioById(dpoResult, null)?.scenario_id).toBe("default");
  });

  it("keeps the peak breakdown equal to the reported peak (one timepoint)", () => {
    for (const result of [grpoResult, dpoResult, sftResult]) {
      for (const scenario of result.memory!.scenarios) {
        const dev = scenario.devices[0]!;
        const sum = dev.peak_breakdown!.items.reduce((acc, i) => acc + (i.bytes_high ?? 0), 0);
        expect(sum).toBe(dev.scenario_high_bytes);
        expect(dev.peak_breakdown!.timepoint).toBe(dev.peak_timepoint);
      }
    }
  });
});
