import { describe, expect, it } from "vitest";

import { classifyChanges, describeChange } from "@/lib/form/changes";
import { buildRequest } from "@/lib/form/convert";
import { requestFingerprint } from "@/lib/form/fingerprint";
import { DEFAULT_FORM_VALUES, type FormValues } from "@/lib/form/values";

const base: FormValues = {
  ...DEFAULT_FORM_VALUES,
  modelReference: "org/model",
  datasetReference: "org/data",
};

function request(values: FormValues) {
  const built = buildRequest(values);
  if (!built.ok) throw new Error(built.message);
  return built.request;
}

describe("change classification", () => {
  const r0 = request(base);

  it("asks for re-analysis when the objective changes", () => {
    const changes = classifyChanges(r0, request({ ...base, objective: "dpo" }));
    expect(changes.reanalysis).toEqual(["학습 방식 변경 (SFT → DPO)"]);
    expect(changes.recompute).toEqual([]);
  });

  it("recomputes LoRA, batch and hardware edits", () => {
    const changes = classifyChanges(r0, request({ ...base, loraR: "32", microbatch: "2" }));
    expect(changes.reanalysis).toEqual([]);
    expect(changes.recompute.map(describeChange)).toEqual(["LoRA r 16 → 32", "microbatch 자동 → 2"]);
  });

  it("treats template, mapping and empty-system policy as preprocessing changes", () => {
    const changed = classifyChanges(r0, request({ ...base, enableThinking: "off", mappingEnabled: true, mapPrompt: "q" }));
    expect(changed.reanalysis).toEqual(expect.arrayContaining(["템플릿 옵션 변경 (enable_thinking)", "컬럼 매핑 변경"]));
  });

  it("does not flag auto values that equal what the analysis resolved", () => {
    const explicit = request({
      ...base,
      datasetSplit: "train",
      mappingEnabled: true,
      mappingFormat: "preference",
      mapPrompt: "question",
      mapChosen: "chosen",
      mapRejected: "rejected",
    });
    const resolved = {
      split: "train",
      mapping: explicit.dataset.mapping,
    };
    expect(classifyChanges(r0, explicit, resolved).reanalysis).toEqual([]);
  });
});

describe("fingerprint", () => {
  it("is independent of key order and sensitive to values", () => {
    expect(requestFingerprint({ a: 1, b: { c: 2, d: 3 } })).toBe(requestFingerprint({ b: { d: 3, c: 2 }, a: 1 }));
    expect(requestFingerprint({ a: 1 })).not.toBe(requestFingerprint({ a: 2 }));
    expect(requestFingerprint(request(base))).toMatch(/^web-[0-9a-f]{16}$/);
  });
});
