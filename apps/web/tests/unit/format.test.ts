import { describe, expect, it } from "vitest";

import {
  GIB,
  bytesToGibInput,
  exactBytes,
  exactBytesRange,
  formatGiB,
  formatGiBRange,
  formatPercent,
  formatSize,
  gibInputToBytes,
  shortDigest,
} from "@/lib/format/bytes";
import { FIT_REASON_LABEL, PROGRESS_STEPS } from "@/lib/format/labels";

describe("byte formatting", () => {
  it("shows 1 GiB for 1,073,741,824 bytes (plan §19.1)", () => {
    expect(formatGiB(1_073_741_824)).toBe("1.0 GiB");
  });

  it("keeps null as null instead of inventing zero", () => {
    expect(formatGiB(null)).toBeNull();
    expect(formatGiBRange(null, 5 * GIB)).toBeNull();
    expect(formatSize(undefined)).toBeNull();
    expect(exactBytes(null)).toBeNull();
  });

  it("renders a range and collapses equal display values", () => {
    expect(formatGiBRange(12.34 * GIB, 15.81 * GIB)).toBe("12.3 – 15.8 GiB");
    expect(formatGiBRange(7 * GIB, 7 * GIB)).toBe("7.0 GiB");
  });

  it("switches to smaller units so a real allocation never reads as 0.0 GiB", () => {
    expect(formatSize(102_525_952)).toBe("97.8 MiB");
    expect(formatSize(4_096)).toBe("4.0 KiB");
    expect(formatSize(12)).toBe("12 B");
    expect(formatSize(3.5 * GIB)).toBe("3.5 GiB");
  });

  it("formats exact bytes with grouping for tooltips", () => {
    expect(exactBytes(13_207_862_272)).toBe("13,207,862,272 bytes");
    expect(exactBytesRange(1, 2)).toBe("1 – 2 bytes");
  });

  it("keeps utilisation above 100% as text", () => {
    expect(formatPercent(1.264)).toBe("126%");
  });

  it("converts GiB input text to integer bytes and back", () => {
    expect(gibInputToBytes("24")).toBe(24 * GIB);
    expect(gibInputToBytes(" ")).toBeNull();
    expect(gibInputToBytes("abc")).toBeNull();
    expect(gibInputToBytes("-1")).toBeNull();
    expect(bytesToGibInput(80 * GIB)).toBe("80");
    expect(bytesToGibInput(null)).toBe("");
    expect(bytesToGibInput(gibInputToBytes("23.65"))).toBe("23.65");
    // More precision than 3 decimals must survive a reload round trip byte for byte.
    for (const text of ["1.23456", "0.0001", "22.4567"]) {
      const bytes = gibInputToBytes(text)!;
      expect(gibInputToBytes(bytesToGibInput(bytes))).toBe(bytes);
    }
    expect(gibInputToBytes(bytesToGibInput(1_234_567_890))).toBe(1_234_567_890);
  });

  it("shortens digests", () => {
    expect(shortDigest("sha256:0123456789abcdef0123")).toBe("0123456789ab…");
    expect(shortDigest("abc")).toBe("abc");
  });
});

describe("labels", () => {
  it("uses the plan §10.3 wording for fit reasons", () => {
    expect(FIT_REASON_LABEL.floor_exceeds_capacity).toBe("확정된 구성만으로 용량 초과");
    expect(FIT_REASON_LABEL.not_evaluated).toBe("용량만 표시, 적합 판정 없음");
  });

  it("maps every running job state to one of the four stages", () => {
    const covered = PROGRESS_STEPS.flatMap((s) => s.statuses);
    expect(covered).toEqual([
      "RESOLVING",
      "INSPECTING",
      "TOKENIZING",
      "VALIDATING_DATA",
      "PLANNING_BATCHES",
      "ESTIMATING",
    ]);
  });
});
