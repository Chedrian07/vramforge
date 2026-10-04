import { describe, expect, it } from "vitest";

import { CONTEXT_LIMIT_UNKNOWN, exceededRowsText, scanCoverageDisplay } from "@/lib/result/scan";

describe("context-exceeded counts", () => {
  it("says the count is unknown without a context limit and never shows 0", () => {
    expect(exceededRowsText(null)).toBe("산정 불가 (context 상한 미상)");
    expect(exceededRowsText(undefined)).toBe(CONTEXT_LIMIT_UNKNOWN);
    expect(exceededRowsText(0)).toBe("0");
    expect(exceededRowsText(1_234)).toBe("1,234");
  });

  it("marks a lower bound", () => {
    expect(exceededRowsText(12, false)).toBe("12개 이상");
    expect(exceededRowsText(null, false)).toBe(CONTEXT_LIMIT_UNKNOWN);
  });
});

describe("scan coverage badge", () => {
  it("never calls a scan with failed rows complete", () => {
    expect(scanCoverageDisplay("complete", { rowsSeen: 4_656, rowsFailed: 0 })).toEqual({ text: "전체 완료", tone: "ok" });
    expect(scanCoverageDisplay("complete", { rowsSeen: 4_656, rowsFailed: 2 })).toEqual({ text: "읽음 4,656 · 실패 2", tone: "warn" });
    expect(scanCoverageDisplay("partial", { rowsSeen: 2_310, rowsFailed: 1 })).toEqual({ text: "부분 · 읽음 2,310 · 실패 1", tone: "warn" });
    expect(scanCoverageDisplay("partial", { rowsFailed: 1 }).text).toBe("부분 · 실패 1");
  });

  it("keeps the coverage word when no row failed or nothing was read", () => {
    expect(scanCoverageDisplay("partial").text).toBe("부분");
    expect(scanCoverageDisplay("failed", { rowsSeen: 0, rowsFailed: 3 }).text).toBe("실패");
    expect(scanCoverageDisplay("not_started").text).toBe("시작 전");
  });
});
