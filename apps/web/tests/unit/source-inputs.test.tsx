import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";

import { DatasetSection } from "@/components/calculator/DatasetSection";
import { ModelSection } from "@/components/calculator/ModelSection";
import { ApiError, type ApiClient } from "@/lib/api/client";
import type { FormValues } from "@/lib/form/values";
import { useDatasetInspection, useModelInspection } from "@/lib/hooks/useInspection";

import { DATASET_REF, MODEL_REF } from "../fixtures/common";
import { ambiguousDatasetInspection, datasetInspection, modelInspection, uploadResponse } from "../fixtures/sources";
import { FormHarness } from "../utils/form-harness";
import { makeEnvironment, renderWithProviders } from "../utils/render";

function ModelHost() {
  const inspection = useModelInspection();
  return <ModelSection inspection={inspection} />;
}

function DatasetHost() {
  const inspection = useDatasetInspection();
  return <DatasetSection inspection={inspection} />;
}

function renderSection(node: React.ReactNode, api: Parameters<typeof makeEnvironment>[0], values: Partial<FormValues> = {}) {
  let form: UseFormReturn<FormValues> | null = null;
  const utils = renderWithProviders(
    <FormHarness values={values} onReady={(f) => (form = f)}>
      {node}
    </FormHarness>,
    { environment: makeEnvironment(api) },
  );
  return { ...utils, form: () => form! };
}

describe("model input", () => {
  it("inspects on blur only, never per keystroke, and shows the metadata", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: modelInspection, dataset: null }));
    renderSection(<ModelHost />, { inspect });
    const input = screen.getByLabelText("Model");
    await user.type(input, MODEL_REF);
    expect(inspect).not.toHaveBeenCalled();
    await user.tab();
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    expect(inspect.mock.calls[0]![0]).toMatchObject({ model: { reference: MODEL_REF, source_type: "huggingface", revision: null } });

    expect(await screen.findByText("Qwen3_5ForConditionalGeneration")).toBeInTheDocument();
    expect(screen.getByText("qwen3_5_hybrid")).toBeInTheDocument();
    expect(screen.getByText("Qwen3_5Tokenizer")).toBeInTheDocument();
    expect(screen.getByText(/chat template · chat_template.jinja/)).toBeInTheDocument();
    expect(screen.getByText(/frozen 가중치로 VRAM에 상주/)).toBeInTheDocument();
    expect(screen.getByText(/GRPO · QLoRA/)).toBeInTheDocument();
    expect(screen.getByText(/linear_attention 24 \/ full_attention 8/)).toBeInTheDocument();

    // Leaving and re-entering without a change does not repeat the request.
    await user.click(input);
    await user.tab();
    expect(inspect).toHaveBeenCalledTimes(1);
  });

  it("validates the reference locally before any request", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn();
    renderSection(<ModelHost />, { inspect });
    await user.type(screen.getByLabelText("Model"), "not a model");
    await user.tab();
    expect(await screen.findByText(/Hugging Face ID\(org\/name\)/)).toBeInTheDocument();
    expect(inspect).not.toHaveBeenCalled();
  });

  it("shows inspection errors as Korean issues", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn(async () => {
      throw new ApiError(404, { code: "SOURCE_NOT_FOUND", severity: "error", retryable: false, user_message: "모델을 찾을 수 없습니다." });
    });
    renderSection(<ModelHost />, { inspect });
    await user.type(screen.getByLabelText("Model"), "org/missing");
    await user.click(screen.getByRole("button", { name: "모델 확인" }));
    expect(await screen.findByText("모델을 찾을 수 없습니다.")).toBeInTheDocument();
  });
});

describe("dataset input and mapping editor", () => {
  it("applies the suggested mapping and labels metadata columns", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: datasetInspection }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { objective: "grpo" });
    await user.type(screen.getByLabelText("Dataset"), DATASET_REF);
    expect(inspect).not.toHaveBeenCalled();
    await user.tab();
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    expect(inspect.mock.calls[0]![0]).toMatchObject({ objective: "grpo", dataset: { reference: DATASET_REF, scan_mode: "full" } });

    expect(await screen.findByLabelText("user prompt")).toHaveValue("question");
    expect(screen.getByLabelText("preferred response (chosen)")).toHaveValue("chosen");
    expect(screen.getByLabelText("dispreferred response (rejected)")).toHaveValue("rejected");
    expect(screen.getByLabelText("system message")).toHaveValue("system");
    const metadata = screen.getByRole("list", { name: "메타데이터 컬럼" });
    expect(within(metadata).getByText("lang")).toBeInTheDocument();
    expect(within(metadata).getByText("vulnerability")).toBeInTheDocument();
    expect(within(metadata).getAllByText(/prompt에 삽입하지 않음/)).toHaveLength(2);
    expect(screen.getByText(/GRPO: prompt만 사용합니다/)).toBeInTheDocument();
    expect(form().getValues("mappingEnabled")).toBe(true);
    expect(form().getValues("datasetSplit")).toBe("");
    expect(screen.getByText("(train 자동 선택)")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("user prompt"), "vulnerability");
    expect(form().getValues("mapPrompt")).toBe("vulnerability");
    expect(within(screen.getByRole("list", { name: "메타데이터 컬럼" })).queryByText("vulnerability")).not.toBeInTheDocument();
  });

  it("asks for config, split and mapping when the dataset is ambiguous", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: ambiguousDatasetInspection }));
    const { form } = renderSection(<DatasetHost />, { inspect });
    await user.type(screen.getByLabelText("Dataset"), DATASET_REF);
    await user.tab();
    expect(await screen.findByText(/매핑 후보가 여러 개입니다/)).toBeInTheDocument();
    expect(screen.getByLabelText("Config")).toBeInTheDocument();
    expect(screen.getByLabelText("학습 split")).toBeInTheDocument();
    expect(form().getValues("mappingEnabled")).toBe(false);
    await user.click(screen.getByRole("button", { name: /후보 2: prompt = instruction/ }));
    expect(form().getValues("mapPrompt")).toBe("instruction");
    await user.selectOptions(screen.getByLabelText("Config"), "extended");
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    expect(inspect.mock.calls[1]![0]).toMatchObject({ dataset: { config: "extended" } });
  });

  it("uploads a file and analyses it by its upload reference", async () => {
    const user = userEvent.setup();
    const upload = vi.fn<ApiClient["upload"]>(async () => uploadResponse);
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: datasetInspection }));
    const { form } = renderSection(<DatasetHost />, { upload, inspect });
    const file = new File(['{"prompt":"a"}\n'], "pairs.jsonl", { type: "application/json" });
    await user.upload(screen.getByLabelText(/데이터 파일 선택/), file);
    await waitFor(() => expect(upload).toHaveBeenCalledWith(file));
    expect(form().getValues("datasetReference")).toBe(uploadResponse.reference);
    expect(await screen.findByText("업로드됨")).toBeInTheDocument();
    await waitFor(() => expect(inspect).toHaveBeenCalled());
    expect(inspect.mock.calls[0]![0]).toMatchObject({ dataset: { source_type: "upload", reference: uploadResponse.reference } });
  });
});
