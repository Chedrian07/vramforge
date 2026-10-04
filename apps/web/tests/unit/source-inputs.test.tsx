import { QueryClient } from "@tanstack/react-query";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";

import { AdvancedSettings } from "@/components/calculator/AdvancedSettings";
import { DatasetSection } from "@/components/calculator/DatasetSection";
import { ModelSection } from "@/components/calculator/ModelSection";
import { ApiError, type ApiClient } from "@/lib/api/client";
import type { FormValues } from "@/lib/form/values";
import { shouldRetryInspection, useDatasetInspection, useModelInspection } from "@/lib/hooks/useInspection";

import { DATASET_REF, MODEL_REF, preferenceMapping } from "../fixtures/common";
import { ambiguousDatasetInspection, datasetInspection, modelInspection, uploadResponse } from "../fixtures/sources";
import type { DatasetInspection } from "@/lib/api/types";

/** A different dataset: instruction/output columns, prompt-completion suggestion. */
const promptCompletionInspection: DatasetInspection = {
  ...datasetInspection,
  columns: [
    { name: "instruction", dtype: "string", kind: "string" },
    { name: "output", dtype: "string", kind: "string" },
    { name: "source", dtype: "string", kind: "string" },
  ],
  detected_format: "prompt_completion",
  suggested_mapping: {
    format: "prompt_completion",
    system: null,
    prompt: "instruction",
    chosen: null,
    rejected: null,
    completion: "output",
    messages: null,
    text: null,
    empty_system_policy: "omit",
  },
  mapping_candidates: [],
};
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

/** The dataset input next to an open "Dataset & reproducibility" group, as on the page. */
function DatasetWithAdvanced() {
  const inspection = useDatasetInspection();
  const data = inspection.state.status === "done" ? inspection.state.data : null;
  return (
    <>
      <DatasetSection inspection={inspection} />
      <AdvancedSettings resolved={null} datasetInspection={data} open={["dataset"]} />
    </>
  );
}

/** Per-objective answers: SFT ranks prompt-completion first, GRPO/DPO the preference pair. */
const sftSuggestion = { ...preferenceMapping, format: "prompt_completion" as const, chosen: null, rejected: null, completion: "chosen" };
function inspectionFor(objective: string | null | undefined): DatasetInspection {
  if (objective === "sft") return { ...datasetInspection, suggested_mapping: sftSuggestion, mapping_candidates: [sftSuggestion, preferenceMapping] };
  if (objective === "dpo") return { ...datasetInspection, suggested_mapping: null, mapping_ambiguous: true };
  return datasetInspection;
}

/** The app's own query defaults (app/providers.tsx): one retry unless a query says otherwise. */
const appQueryClient = () => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000, retry: 1, refetchOnWindowFocus: false } } });

function renderSection(node: React.ReactNode, api: Parameters<typeof makeEnvironment>[0], values: Partial<FormValues> = {}) {
  let form: UseFormReturn<FormValues> | null = null;
  const utils = renderWithProviders(
    <FormHarness values={values} onReady={(f) => (form = f)}>
      {node}
    </FormHarness>,
    { environment: makeEnvironment(api), client: appQueryClient() },
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

  it("marks the supported combination of the current method and strategy", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: modelInspection, dataset: null }));
    const { form } = renderSection(<ModelHost />, { inspect }, { modelReference: MODEL_REF, objective: "grpo" });
    await user.click(screen.getByRole("button", { name: "모델 확인" }));
    expect(await screen.findByText(/GRPO · QLoRA — 정적 추정 · 조건부 \(선택\)/)).toBeInTheDocument();
    act(() => form().setValue("objective", "dpo"));
    expect(await screen.findByText(/DPO · QLoRA — 정적 추정 · 준비됨 \(선택\)/)).toBeInTheDocument();
    expect(screen.queryByText(/GRPO · QLoRA.*\(선택\)/)).not.toBeInTheDocument();
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
    // A 4xx answer is final: no silent retry before the message.
    expect(inspect).toHaveBeenCalledTimes(1);
  });

  it("retries a transient inspection failure once", async () => {
    const user = userEvent.setup();
    let calls = 0;
    const inspect = vi.fn<ApiClient["inspect"]>(async () => {
      calls += 1;
      if (calls === 1) throw new ApiError(503, { code: "INTERNAL_ERROR", severity: "error", retryable: true, user_message: "일시적 오류" });
      return { model: modelInspection, dataset: null };
    });
    renderSection(<ModelHost />, { inspect });
    await user.type(screen.getByLabelText("Model"), MODEL_REF);
    await user.click(screen.getByRole("button", { name: "모델 확인" }));
    expect(await screen.findByText("qwen3_5_hybrid", {}, { timeout: 4_000 })).toBeInTheDocument();
    expect(inspect).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("일시적 오류")).not.toBeInTheDocument();
  });
});

describe("inspection retry policy", () => {
  const error = (status: number) => new ApiError(status, { code: "INTERNAL_ERROR", severity: "error", retryable: false, user_message: "x" });

  it("retries only unanswered, throttled or server-failed requests, once", () => {
    expect(shouldRetryInspection(0, error(0))).toBe(true);
    expect(shouldRetryInspection(0, error(429))).toBe(true);
    expect(shouldRetryInspection(0, error(502))).toBe(true);
    expect(shouldRetryInspection(1, error(502))).toBe(false);
    for (const status of [400, 401, 403, 404, 409, 422]) expect(shouldRetryInspection(0, error(status))).toBe(false);
    expect(shouldRetryInspection(0, new TypeError("x"))).toBe(false);
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
    // The chosen candidate still fits the columns of the new config, so it is kept.
    await waitFor(() => expect(screen.getByLabelText("Config")).toHaveValue("extended"));
    expect(form().getValues("mapPrompt")).toBe("instruction");
    expect(form().getValues("mappingEnabled")).toBe(true);
  });

  it("starts again from auto-detection when the dataset changes", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async (body) => ({
      model: null,
      dataset: body.dataset?.reference === DATASET_REF ? datasetInspection : promptCompletionInspection,
    }));
    const { form } = renderSection(<DatasetHost />, { inspect });
    const input = screen.getByLabelText("Dataset");
    await user.type(input, DATASET_REF);
    await user.tab();
    await waitFor(() => expect(form().getValues("mapPrompt")).toBe("question"));

    await user.clear(input);
    await user.type(input, "org/other-data");
    await user.tab();
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(form().getValues("mappingFormat")).toBe("prompt_completion"));
    const v = form().getValues();
    expect([v.mapPrompt, v.mapCompletion, v.mapChosen, v.mapRejected, v.mapSystem]).toEqual(["instruction", "output", "", "", ""]);
  });

  it("resets the old dataset's selections when another dataset is typed after a reload", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: promptCompletionInspection }));
    // A form restored from requested_config: no inspection has run on this page yet.
    const { form } = renderSection(<DatasetHost />, { inspect }, {
      datasetReference: DATASET_REF,
      datasetConfig: "extended",
      datasetSplit: "train_extra",
      datasetEvalSplit: "test",
      datasetRevision: "81aeacf06cf43b16d7278a3a01f019a496a53c51",
      mappingEnabled: true,
      mappingFormat: "preference",
      mapPrompt: "instruction",
      mapChosen: "chosen",
      mapRejected: "rejected",
    });
    const input = screen.getByLabelText("Dataset");
    await user.clear(input);
    await user.type(input, "org/other-data");
    await user.tab();
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    expect(inspect.mock.calls[0]![0]).toMatchObject({ dataset: { reference: "org/other-data", config: null, revision: null } });
    const v = form().getValues();
    expect([v.datasetConfig, v.datasetSplit, v.datasetEvalSplit, v.datasetRevision]).toEqual(["", "", "", ""]);
    // "instruction" exists in the new dataset too, but the mapping was made for the old one.
    await waitFor(() => expect(form().getValues("mappingFormat")).toBe("prompt_completion"));
    expect(form().getValues("mapCompletion")).toBe("output");
  });

  it("keeps the selections when the same dataset is confirmed again", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: ambiguousDatasetInspection }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { datasetReference: DATASET_REF, datasetConfig: "extended", datasetSplit: "train_extra" });
    await user.click(screen.getByLabelText("Dataset"));
    await user.tab();
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await waitFor(() => expect(inspect).toHaveBeenCalled());
    expect(inspect.mock.calls.at(-1)![0]).toMatchObject({ dataset: { config: "extended" } });
    expect(form().getValues("datasetSplit")).toBe("train_extra");
  });

  it("drops a restored mapping that does not fit the inspected columns", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: promptCompletionInspection }));
    const { form } = renderSection(<DatasetHost />, { inspect }, {
      datasetReference: "org/other-data",
      mappingEnabled: true,
      mappingFormat: "preference",
      mapSystem: "system",
      mapPrompt: "question",
      mapChosen: "chosen",
      mapRejected: "rejected",
      emptySystemPolicy: "keep",
    });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await waitFor(() => expect(form().getValues("mappingFormat")).toBe("prompt_completion"));
    expect(form().getValues("mapChosen")).toBe("");
    expect(form().getValues("emptySystemPolicy")).toBe("omit");
  });

  it("asks the inspector again when the objective changed", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: datasetInspection }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { datasetReference: DATASET_REF, objective: "grpo" });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(1));
    act(() => form().setValue("objective", "dpo"));
    await user.click(screen.getByLabelText("Dataset"));
    await user.tab();
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    expect(inspect.mock.calls[1]![0]).toMatchObject({ objective: "dpo" });
  });

  it("keeps the inline and the Advanced split selects in sync (one field, one registration)", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: ambiguousDatasetInspection }));
    const { form } = renderSection(<DatasetWithAdvanced />, { inspect }, { datasetReference: DATASET_REF });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await screen.findByLabelText("Config");
    const [inline, advanced] = screen.getAllByLabelText("학습 split");
    await user.selectOptions(inline!, "train_extra");
    expect(form().getValues("datasetSplit")).toBe("train_extra");
    expect(advanced).toHaveValue("train_extra");
    await user.selectOptions(advanced!, "train");
    expect(inline).toHaveValue("train");
    expect(form().getValues("datasetSplit")).toBe("train");
  });

  it("starts the split choice over when another config is picked", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async () => ({ model: null, dataset: ambiguousDatasetInspection }));
    const { form } = renderSection(<DatasetWithAdvanced />, { inspect }, { datasetReference: DATASET_REF, datasetEvalSplit: "train" });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await screen.findByLabelText("Config");
    const [inline] = screen.getAllByLabelText("학습 split");
    await user.selectOptions(inline!, "train_extra");
    await user.selectOptions(screen.getByLabelText("Config"), "extended");
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByLabelText("Config")).toHaveValue("extended"));
    expect([form().getValues("datasetSplit"), form().getValues("datasetEvalSplit")]).toEqual(["", ""]);
    const [inlineAfter, advancedAfter] = screen.getAllByLabelText("학습 split");
    expect(inlineAfter).toHaveValue("");
    expect(advancedAfter).toHaveValue("");

    // Typing another config in Advanced resets the split as well.
    await user.selectOptions(inlineAfter!, "train");
    await user.type(screen.getByLabelText("데이터셋 config"), "x");
    expect(form().getValues("datasetSplit")).toBe("");
  });

  it("replaces an automatically applied mapping when the objective changes", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async (body) => ({ model: null, dataset: inspectionFor(body.objective) }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { datasetReference: DATASET_REF, objective: "grpo", emptySystemPolicy: "keep" });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await waitFor(() => expect(form().getValues("mappingFormat")).toBe("preference"));
    expect(form().getValues("mappingAutoApplied")).toBe(true);

    // No blur or click: the inspector ranks candidates per objective, so it is asked again.
    act(() => form().setValue("objective", "sft"));
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    expect(inspect.mock.calls[1]![0]).toMatchObject({ objective: "sft" });
    await waitFor(() => expect(form().getValues("mappingFormat")).toBe("prompt_completion"));
    const v = form().getValues();
    expect([v.mapPrompt, v.mapCompletion, v.mapChosen, v.mapRejected]).toEqual(["question", "chosen", "", ""]);
    expect(v.emptySystemPolicy).toBe("keep"); // the user's policy is kept for the next mapping
  });

  it("drops an automatically applied mapping when the new objective is ambiguous", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async (body) => ({ model: null, dataset: inspectionFor(body.objective) }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { datasetReference: DATASET_REF, objective: "grpo" });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await waitFor(() => expect(form().getValues("mappingEnabled")).toBe(true));
    act(() => form().setValue("objective", "dpo"));
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    // The server would ask for DPO; an answer given for GRPO must not decide it.
    await waitFor(() => expect(form().getValues("mappingEnabled")).toBe(false));
    expect(form().getValues("mappingFormat")).toBe("auto");
    expect(await screen.findByText(/매핑 후보가 여러 개입니다/)).toBeInTheDocument();
  });

  it("keeps a mapping the user chose when the objective changes", async () => {
    const user = userEvent.setup();
    const inspect = vi.fn<ApiClient["inspect"]>(async (body) => ({ model: null, dataset: inspectionFor(body.objective) }));
    const { form } = renderSection(<DatasetHost />, { inspect }, { datasetReference: DATASET_REF, objective: "grpo" });
    await user.click(screen.getByRole("button", { name: "데이터셋 확인" }));
    await screen.findByLabelText("user prompt");
    await waitFor(() => expect(form().getValues("mappingEnabled")).toBe(true));
    await user.selectOptions(screen.getByLabelText("user prompt"), "vulnerability");
    expect(form().getValues("mappingAutoApplied")).toBe(false);
    act(() => form().setValue("objective", "sft"));
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(form().getValues("mappingFormat")).toBe("preference");
    expect(form().getValues("mapPrompt")).toBe("vulnerability");
  });

  it("lists only the roles of the chosen format in the restored mapping summary", () => {
    renderSection(<DatasetHost />, {}, {
      datasetReference: DATASET_REF,
      mappingEnabled: true,
      mappingFormat: "messages",
      mapMessages: "conversations",
      // Left over from an earlier preference choice: not sent, so not shown.
      mapPrompt: "question",
      mapChosen: "chosen",
    });
    const summary = screen.getByText(/적용 매핑/);
    expect(summary).toHaveTextContent("conversations → messages");
    expect(summary).not.toHaveTextContent("question");
    expect(summary).not.toHaveTextContent("chosen");
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
