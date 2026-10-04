import { describe, expect, it } from "vitest";

import { hardwareSelected, scenarioById, summarize } from "@/lib/result/summary";

import { dpoResult, sftResult } from "../fixtures/analysis-dpo-sft";
import { BUDGETS, grpoExplicitBudgetResult, grpoResult } from "../fixtures/analysis-grpo";
import { partialResult, unknownPeakResult } from "../fixtures/analysis-states";

describe("summarize", () => {
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
