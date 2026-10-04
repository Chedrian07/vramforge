import { describe, expect, it } from "vitest";

import { gibTickLabel, gibTicks } from "@/components/results/chart-kit";
import { MEMORY_GROUPS, groupOf, groupTotals } from "@/lib/result/groups";

const GIB = 1_073_741_824;

describe("chart axis ticks", () => {
  it("places ticks on clean GiB values covering the maximum", () => {
    const { ticks, max } = gibTicks(13.1 * GIB);
    expect(ticks.map(gibTickLabel)).toEqual(["0", "5", "10", "15"]);
    expect(max).toBeGreaterThanOrEqual(13.1 * GIB);
    expect(gibTicks(7.7 * GIB).ticks.map(gibTickLabel)).toEqual(["0", "2", "4", "6", "8"]);
    expect(gibTicks(0.4 * GIB).ticks.map(gibTickLabel)).toEqual(["0", "0.1", "0.2", "0.3", "0.4"]);
  });
});

describe("memory groups", () => {
  it("folds every allocation category into one of seven fixed groups", () => {
    expect(MEMORY_GROUPS).toHaveLength(7);
    const categories = MEMORY_GROUPS.flatMap((g) => g.categories);
    expect(new Set(categories).size).toBe(categories.length);
    expect(groupOf("allocator_slack").key).toBe("overhead");
    expect(groupOf("generation_cache").key).toBe("generation");
  });

  it("sums known sizes and keeps unknown items apart", () => {
    const { totals, unknown } = groupTotals([
      { name: "a", category: "weights_base", bytes_low: 1, bytes_high: 2, evidence: "analytic" },
      { name: "b", category: "weights_base", bytes_low: 3, bytes_high: 4, evidence: "analytic" },
      { name: "c", category: "workspace", bytes_low: null, bytes_high: null, evidence: "unknown" },
    ]);
    expect(totals).toEqual([expect.objectContaining({ low: 4, high: 6, items: 2 })]);
    expect(unknown.map((u) => u.name)).toEqual(["c"]);
  });
});
