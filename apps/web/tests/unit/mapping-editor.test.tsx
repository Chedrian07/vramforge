import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it } from "vitest";

import { MappingEditor } from "@/components/calculator/MappingEditor";
import type { DatasetInspection } from "@/lib/api/types";
import type { FormValues } from "@/lib/form/values";

import { ambiguousDatasetInspection, datasetInspection } from "../fixtures/sources";
import { FormHarness } from "../utils/form-harness";
import { renderWithProviders } from "../utils/render";

function renderEditor(inspection: DatasetInspection, values: Partial<FormValues> = {}) {
  let form: UseFormReturn<FormValues> | null = null;
  renderWithProviders(
    <FormHarness values={values} onReady={(f) => (form = f)}>
      <MappingEditor inspection={inspection} />
    </FormHarness>,
  );
  return { form: () => form! };
}

const explicitPreference: Partial<FormValues> = {
  mappingEnabled: true,
  mappingFormat: "preference",
  mapSystem: "system",
  mapPrompt: "question",
  mapChosen: "chosen",
  mapRejected: "rejected",
};

describe("mapping editor", () => {
  it("makes no metadata claim while the server auto-detects the roles", () => {
    renderEditor(datasetInspection);
    expect(screen.getByText(/현재 매핑:/)).toHaveTextContent("자동 감지");
    expect(screen.queryByRole("list", { name: "메타데이터 컬럼" })).not.toBeInTheDocument();
    expect(screen.queryByText(/prompt에 삽입하지 않음/)).not.toBeInTheDocument();
    expect(within(screen.getByLabelText("user prompt")).getByRole("option", { name: "— 자동 감지 —" })).toBeInTheDocument();
    expect(screen.getByText(/역할은 분석 시 서버가 정하고/)).toBeInTheDocument();
  });

  it("starts a manual edit from the unambiguous suggestion instead of a single role", async () => {
    const user = userEvent.setup();
    const { form } = renderEditor(datasetInspection);
    await user.selectOptions(screen.getByLabelText("user prompt"), "vulnerability");
    const v = form().getValues();
    expect(v.mappingEnabled).toBe(true);
    expect(v.mappingFormat).toBe("preference");
    expect([v.mapSystem, v.mapPrompt, v.mapChosen, v.mapRejected]).toEqual(["system", "vulnerability", "chosen", "rejected"]);
    expect(within(screen.getByRole("list", { name: "메타데이터 컬럼" })).getByText("question")).toBeInTheDocument();
  });

  it("asks for a format and the required roles when there is no suggestion", async () => {
    const user = userEvent.setup();
    const { form } = renderEditor(ambiguousDatasetInspection);
    await user.selectOptions(screen.getByLabelText("user prompt"), "instruction");
    expect(form().getValues("mappingEnabled")).toBe(true);
    expect(await screen.findByText(/데이터 형식을 골라야 합니다/)).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("데이터 형식"), "preference");
    expect(await screen.findByText(/preferred response \(chosen\) 컬럼을 선택하세요/)).toBeInTheDocument();
    expect(screen.getByLabelText("preferred response (chosen)")).toHaveAttribute("aria-invalid", "true");
    await user.selectOptions(screen.getByLabelText("preferred response (chosen)"), "chosen");
    await user.selectOptions(screen.getByLabelText("dispreferred response (rejected)"), "rejected");
    expect(screen.queryByText(/컬럼을 선택하세요/)).not.toBeInTheDocument();
  });

  it("keeps the user's empty-system policy when a suggestion is applied", async () => {
    const user = userEvent.setup();
    const { form } = renderEditor(datasetInspection, { ...explicitPreference, mapPrompt: "lang", emptySystemPolicy: "keep" });
    await user.click(screen.getByRole("button", { name: "제안 매핑 적용" }));
    expect(form().getValues("mapPrompt")).toBe("question");
    expect(form().getValues("emptySystemPolicy")).toBe("keep");
  });

  it("returns to auto-detection without leftover roles or policy", async () => {
    const user = userEvent.setup();
    const { form } = renderEditor(datasetInspection, { ...explicitPreference, emptySystemPolicy: "keep" });
    await user.click(screen.getByRole("button", { name: "자동 감지로 되돌리기" }));
    const v = form().getValues();
    expect(v.mappingEnabled).toBe(false);
    expect(v.mappingFormat).toBe("auto");
    expect([v.mapSystem, v.mapPrompt, v.mapChosen, v.mapRejected]).toEqual(["", "", "", ""]);
    expect(v.emptySystemPolicy).toBe("omit");
  });
});
