import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CalculatorApp } from "@/components/calculator/CalculatorApp";
import type { ApiClient } from "@/lib/api/client";
import type { AnalysisRequest, AnalysisStatus, InspectRequest, ScenarioResponse } from "@/lib/api/types";

import { dpoResult } from "../fixtures/analysis-dpo-sft";
import { completedGrpoStatus, completedSftStatus, needsInputStatus, runningStatus } from "../fixtures/analysis-states";
import { DATASET_REF, MODEL_REF } from "../fixtures/common";
import { datasetInspection, modelInspection } from "../fixtures/sources";
import { FakeEventSource } from "../utils/fake-event-source";
import { makeEvent } from "../utils/events";
import { makeEnvironment, renderWithProviders } from "../utils/render";

const ID = completedGrpoStatus.analysis_id;

function created(id = ID) {
  return { analysis_id: id, status: "QUEUED" as const, fingerprint: "vf-fixture-fp", created_at: "2026-10-04T12:00:00Z", reused: false };
}

function renderApp(api: Partial<ApiClient>) {
  const inspect = vi.fn<ApiClient["inspect"]>(async (body: InspectRequest) => ({
    model: body.model ? modelInspection : null,
    dataset: body.dataset ? datasetInspection : null,
  }));
  const environment = makeEnvironment({ inspect, ...api });
  return { ...renderWithProviders(<CalculatorApp />, { environment }), inspect, environment };
}

beforeEach(() => {
  FakeEventSource.reset();
  window.history.replaceState(null, "", "/");
});
afterEach(() => window.history.replaceState(null, "", "/"));

describe("calculator flow", () => {
  it("starts honest: no fake numbers or fit before analysis", () => {
    const { container } = renderApp({});
    expect(screen.getByText("전체 데이터 분석 필요")).toBeInTheDocument();
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/\b0(\.0)? ?(GB|GiB)\b/);
    expect(text).not.toContain("예상 적합");
    expect(screen.getByRole("button", { name: "전체 데이터 분석 및 계산" })).toBeEnabled();
  });

  it("opens the Advanced group that holds an error when the run is refused", async () => {
    const user = userEvent.setup();
    const createAnalysis = vi.fn<ApiClient["createAnalysis"]>();
    renderApp({ createAnalysis });
    await user.click(screen.getByRole("button", { name: "예시 입력 불러오기" }));
    const adapter = screen.getByRole("button", { name: /^Adapter/ });
    await user.click(adapter);
    await user.clear(screen.getByLabelText("rank (r)"));
    await user.type(screen.getByLabelText("rank (r)"), "0");
    await user.click(adapter); // collapse: the field and its message unmount
    expect(screen.queryByLabelText("rank (r)")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "전체 데이터 분석 및 계산" }));
    await waitFor(() => expect(adapter).toHaveAttribute("aria-expanded", "true"));
    expect(screen.getByLabelText("rank (r)")).toHaveAccessibleDescription("1–4,096 사이 정수를 입력하세요");
    expect(createAnalysis).not.toHaveBeenCalled();
  });

  it("runs the GRPO example end to end and asks for re-analysis on a method switch", async () => {
    const user = userEvent.setup();
    const createAnalysis = vi.fn<ApiClient["createAnalysis"]>(async () => created());
    const getAnalysis = vi.fn<ApiClient["getAnalysis"]>(async () => completedGrpoStatus);
    const scenarios = vi.fn<ApiClient["scenarios"]>();
    const { inspect } = renderApp({ createAnalysis, getAnalysis, scenarios });

    await user.click(screen.getByRole("button", { name: "예시 입력 불러오기" }));
    expect(screen.getByLabelText("Model")).toHaveValue(MODEL_REF);
    expect(screen.getByLabelText("Dataset")).toHaveValue(DATASET_REF);
    await waitFor(() => expect(inspect).toHaveBeenCalledTimes(2));
    expect(await screen.findByLabelText("user prompt")).toHaveValue("question");

    await user.click(screen.getByRole("button", { name: "전체 데이터 분석 및 계산" }));
    await waitFor(() => expect(createAnalysis).toHaveBeenCalledTimes(1));
    const sent = createAnalysis.mock.calls[0]![0] as AnalysisRequest;
    expect(sent.training.objective).toBe("grpo");
    expect(sent.training.strategy).toBe("qlora");
    expect(sent.dataset.mapping?.prompt).toBe("question");
    expect(window.location.search).toBe(`?analysis=${ID}`);

    const es = FakeEventSource.latest();
    act(() => {
      es.open();
      es.emit(
        "progress",
        makeEvent({ event_id: 1, analysis_id: ID, type: "progress", status: "TOKENIZING", progress: { stage: "TOKENIZING", processed_rows: 1_536, total_rows: null }, partial: { max_tokens: 201 } }),
      );
    });
    expect(screen.getByText("1,536 row 처리")).toBeInTheDocument();
    expect(screen.getByText(/현재까지 확인한 데이터 기준/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "분석 진행 중…" })).toBeDisabled();

    act(() => es.emit("completed", makeEvent({ event_id: 2, analysis_id: ID, type: "completed", status: "COMPLETED" })));
    expect(await screen.findByText("분석을 마쳤습니다. 결과는 요약 카드와 아래 탭에 있습니다.")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /completion budget별/ })).toBeInTheDocument();
    expect(screen.getByText("reward footprint 미포함")).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "결과 상태" })).getByText("조건부")).toBeInTheDocument();

    await user.click(within(screen.getByRole("radiogroup", { name: "Method" })).getByRole("radio", { name: "DPO" }));
    expect(await screen.findByRole("button", { name: "데이터 재분석 필요" })).toBeInTheDocument();
    expect(screen.getByText("학습 방식 변경 (GRPO → DPO)")).toBeInTheDocument();
    expect(screen.getAllByText(/이전 설정/).length).toBeGreaterThan(0);
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(scenarios).not.toHaveBeenCalled();
  });

  it("reconnects a running job after reload", async () => {
    window.history.replaceState(null, "", `/?analysis=${ID}`);
    const getAnalysis = vi.fn<ApiClient["getAnalysis"]>(async () => runningStatus);
    renderApp({ getAnalysis });
    expect(await screen.findByText("1,536 row 처리")).toBeInTheDocument();
    expect(FakeEventSource.latest().url).toBe(`/api/v1/analyses/${ID}/events?after=${runningStatus.last_event_id}`);
    expect(screen.getByRole("button", { name: "분석 취소" })).toBeInTheDocument();
  });

  it("restores the form from requested_config and recomputes light changes", async () => {
    const user = userEvent.setup();
    window.history.replaceState(null, "", `/?analysis=${completedSftStatus.analysis_id}`);
    let release: (value: ScenarioResponse) => void = () => {};
    const scenarios = vi.fn<ApiClient["scenarios"]>(
      (_id, body) =>
        new Promise<ScenarioResponse>((resolve) => {
          release = () =>
            resolve({ fingerprint: "vf-fixture-s1", client_fingerprint: body.client_fingerprint ?? null, requires_reanalysis: false, reanalysis_reasons: [], result: dpoResult });
        }),
    );
    renderApp({ getAnalysis: vi.fn(async (): Promise<AnalysisStatus> => completedSftStatus), scenarios });

    await waitFor(() => expect(screen.getByLabelText("Model")).toHaveValue(MODEL_REF));
    expect(within(screen.getByRole("radiogroup", { name: "Method" })).getByRole("radio", { name: "SFT" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("button", { name: "전체 데이터 분석 및 계산" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Adapter/ }));
    const r = screen.getByLabelText("rank (r)");
    await user.clear(r);
    await user.type(r, "32");
    expect(screen.getAllByText(/이전 설정/).length).toBeGreaterThan(0);
    await waitFor(() => expect(scenarios).toHaveBeenCalledTimes(1), { timeout: 2_000 });
    const body = scenarios.mock.calls[0]![1];
    expect(body.request.training.lora?.r).toBe(32);
    expect(body.client_fingerprint).toMatch(/^web-[0-9a-f]{16}$/);
    await act(async () => release(undefined as never));
    await waitFor(() => expect(screen.queryByText(/이전 설정 · 바뀐 조건으로 재계산 중/)).not.toBeInTheDocument());
    expect(screen.getByRole("img", { name: /GPU 사용량/ })).toBeInTheDocument();

    // The recomputed numbers are not in the stored analysis that the exports are made from.
    await user.click(screen.getByRole("button", { name: "결과 내보내기" }));
    expect(within(screen.getByRole("dialog", { name: "결과 내보내기" })).getByRole("note")).toHaveTextContent("서버에 저장된 기준 분석");
  });

  it("answers NEEDS_INPUT inline and starts a new analysis", async () => {
    const user = userEvent.setup();
    window.history.replaceState(null, "", `/?analysis=${needsInputStatus.analysis_id}`);
    const createAnalysis = vi.fn<ApiClient["createAnalysis"]>(async () => created("vf-fixture-next"));
    renderApp({ getAnalysis: vi.fn(async () => needsInputStatus), createAnalysis });
    await user.selectOptions(await screen.findByLabelText("dataset.mapping"), "1");
    await user.click(screen.getByRole("button", { name: "선택 적용 후 다시 분석" }));
    await waitFor(() => expect(createAnalysis).toHaveBeenCalledTimes(1));
    const sent = createAnalysis.mock.calls[0]![0] as AnalysisRequest;
    expect(sent.dataset.split).toBe("train");
    expect(sent.dataset.mapping?.prompt).toBe("instruction");
  });
});
