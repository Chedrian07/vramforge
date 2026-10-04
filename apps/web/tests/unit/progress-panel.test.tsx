import { act, cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";

import { ProgressPanel, stepStates } from "@/components/calculator/ProgressPanel";
import { ApiError } from "@/lib/api/client";
import type { FormValues } from "@/lib/form/values";
import { INITIAL_RUN_STATE, type RunState } from "@/lib/hooks/useAnalysisRun";

import { cancelledStatus, completedGrpoStatus, failedStatus, needsInputStatus, partialStatus } from "../fixtures/analysis-states";
import { issue } from "../fixtures/common";
import { FormHarness } from "../utils/form-harness";
import { renderWithProviders } from "../utils/render";

const BASE: RunState = INITIAL_RUN_STATE;

function renderPanel(run: Partial<RunState>, handlers: { onCancel?: () => void; onRerun?: () => void } = {}) {
  let form: UseFormReturn<FormValues> | null = null;
  renderWithProviders(
    <FormHarness onReady={(f) => (form = f)}>
      <ProgressPanel run={{ ...BASE, ...run }} onCancel={handlers.onCancel ?? vi.fn()} onRerun={handlers.onRerun ?? vi.fn()} />
    </FormHarness>,
  );
  return { form: () => form! };
}

function terminal(status: typeof completedGrpoStatus): Partial<RunState> {
  return { phase: "terminal", analysisId: status.analysis_id, jobStatus: status.status, status, progress: status.progress ?? null };
}

describe("progress panel", () => {
  it("shows the four stages before any run", () => {
    renderPanel({});
    const steps = within(screen.getByRole("list", { name: "분석 단계" })).getAllByRole("listitem");
    expect(steps.map((s) => s.textContent)).toEqual(["1구조 확인대기", "2데이터 토큰화대기", "3배치 분석대기", "4메모리 산정대기"]);
  });

  it("counts rows without a fake percentage when the total is unknown", () => {
    renderPanel({
      phase: "running",
      analysisId: "vf-fixture-a",
      jobStatus: "TOKENIZING",
      progress: { stage: "TOKENIZING", processed_rows: 1_536, total_rows: null, shard_progress: { completed: 0, total: null } },
    });
    const bar = screen.getByRole("progressbar", { name: "처리한 row" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(bar).toHaveAttribute("aria-valuetext", "1,536 row 처리 (전체 row 수 미확인)");
    expect(screen.getByText("1,536 row 처리")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toHaveTextContent(/\d%/);
    expect(screen.getByText("shard 0 / ?")).toBeInTheDocument();
    const current = screen.getAllByRole("listitem").find((li) => li.getAttribute("aria-current") === "step");
    expect(current).toHaveTextContent("데이터 토큰화");
  });

  it("shows a real percentage only when the total is known", () => {
    renderPanel({ phase: "running", analysisId: "x", jobStatus: "TOKENIZING", progress: { stage: "TOKENIZING", processed_rows: 2_328, total_rows: 4_656 } });
    const bar = screen.getByRole("progressbar", { name: "처리한 row" });
    expect(bar).toHaveAttribute("aria-valuenow", "2328");
    expect(screen.getByText("2,328 / 4,656 row · 50%")).toBeInTheDocument();
  });

  it("requests cancellation while running", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderPanel({ phase: "running", analysisId: "x", jobStatus: "PLANNING_BATCHES" }, { onCancel });
    await user.click(screen.getByRole("button", { name: "분석 취소" }));
    expect(onCancel).toHaveBeenCalled();
  });

  it("reports failed, partial and cancelled jobs", () => {
    renderPanel(terminal(failedStatus));
    expect(screen.getByText("모델 저장소에 tokenizer 파일이 없어 데이터 길이를 계산할 수 없습니다.")).toBeInTheDocument();
  });

  it("reports partial results with the reason", () => {
    renderPanel(terminal(partialStatus));
    expect(screen.getByText(/부분 결과입니다/)).toBeInTheDocument();
    expect(screen.getByText(/처리 시간 한도로 2,310/)).toBeInTheDocument();
    expect(stepStates({ jobStatus: "PARTIAL", progress: partialStatus.progress!, phase: "terminal" })).toEqual(["done", "stopped", "pending", "pending"]);
  });

  it("shows where a job stopped when the API's terminal progress names only the end status", () => {
    // What the API stores at the end (store.finish): stage = the terminal status, no row counts.
    const cancelled = {
      ...cancelledStatus,
      progress: { stage: "CANCELLED" as const, message: "사용자 요청으로 취소되었습니다." },
      error: issue("CANCELLED", "warning", "사용자 요청으로 작업이 취소되었습니다.", { stage: "tokenizing" }),
    };
    renderPanel(terminal(cancelled));
    const steps = within(screen.getByRole("list", { name: "분석 단계" })).getAllByRole("listitem");
    expect(steps.map((s) => s.textContent)).toEqual(["✓구조 확인완료", "!데이터 토큰화중단", "3배치 분석대기", "4메모리 산정대기"]);
    // The panel's own note replaces the server's closing message instead of repeating it.
    expect(screen.getByText("분석을 취소했습니다.")).toBeInTheDocument();
    expect(screen.queryByText("사용자 요청으로 취소되었습니다.")).not.toBeInTheDocument();

    // An issue without a pipeline stage (worker stopped): the furthest stage seen on this page.
    const stopped = { ...cancelled, error: { ...cancelled.error, stage: "api" as const } };
    expect(stepStates({ ...terminal(stopped), jobStatus: "FAILED", phase: "terminal", progress: stopped.progress, reachedStage: "PLANNING_BATCHES" })).toEqual([
      "done",
      "done",
      "stopped",
      "pending",
    ]);
    // A cancel request keeps the stage it interrupts current.
    expect(stepStates({ jobStatus: "CANCEL_REQUESTED", phase: "running", progress: { stage: "CANCEL_REQUESTED" }, reachedStage: "TOKENIZING" })).toEqual([
      "done",
      "current",
      "pending",
      "pending",
    ]);
  });

  it("says once that the analysis finished", () => {
    // The API's closing progress (store.finish) carries the same sentence the panel writes.
    renderPanel(terminal({ ...completedGrpoStatus, progress: { stage: "COMPLETED", message: "분석을 마쳤습니다." } }));
    expect(screen.getAllByText(/분석을 마쳤습니다/)).toHaveLength(1);
    expect(screen.getByText("분석을 마쳤습니다. 결과는 요약 카드와 아래 탭에 있습니다.")).toBeInTheDocument();
  });

  it("shows the ending event's status and issue while the final status is missing", () => {
    renderPanel({
      phase: "error",
      analysisId: "vf-fixture-a",
      jobStatus: "PARTIAL",
      endIssue: { code: "SCAN_PARTIAL", severity: "error", user_message: "처리 시간 한도로 일부만 확인했습니다." },
      error: new ApiError(0, { code: "INTERNAL_ERROR", severity: "error", retryable: true, user_message: "서버에 연결하지 못했습니다." }),
    });
    expect(screen.getByText(/부분 결과입니다/)).toBeInTheDocument();
    expect(screen.getByText("처리 시간 한도로 일부만 확인했습니다.")).toBeInTheDocument();
    expect(screen.getByText("서버에 연결하지 못했습니다.")).toBeInTheDocument();
    expect(screen.queryByText(/분석을 마쳤습니다/)).not.toBeInTheDocument();
  });

  it("stops offering cancel once an event reports the end", () => {
    renderPanel({ phase: "running", analysisId: "vf-fixture-a", jobStatus: "COMPLETED" });
    expect(screen.queryByRole("button", { name: "분석 취소" })).not.toBeInTheDocument();
    expect(screen.getByText(/저장된 최종 결과를 불러오는 중/)).toBeInTheDocument();
    // "완료" is only claimed once the stored status says so.
    expect(screen.queryByText(/분석을 마쳤습니다/)).not.toBeInTheDocument();
  });

  it("reports cancellation with the rows seen", () => {
    renderPanel(terminal(cancelledStatus));
    expect(screen.getByText(/취소 시점까지 1,200 row/)).toBeInTheDocument();
  });

  it("preselects the mapping candidate the server suggests by label", async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    const result = needsInputStatus.result!;
    const labels = ["system=system, prompt=question, chosen=chosen, rejected=rejected", "system=system, prompt=instruction, chosen=chosen, rejected=rejected"];
    const status = {
      ...needsInputStatus,
      result: {
        ...result,
        needs_input: {
          ...result.needs_input!,
          choices: [{ field: "dataset.mapping", options: labels, suggested: labels[1]!, reason: "prompt 후보가 두 개입니다." }],
        },
      },
    };
    const { form } = renderPanel(terminal(status), { onRerun });
    expect(screen.getByLabelText("dataset.mapping")).toHaveValue("1");
    await user.click(screen.getByRole("button", { name: "선택 적용 후 다시 분석" }));
    expect(form().getValues("mapPrompt")).toBe("instruction");
    expect(onRerun).toHaveBeenCalled();
  });

  it("sends a mapping question without candidates to the column mapping editor", async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    const result = needsInputStatus.result!;
    // One malformed row made every column non-text for the inspector: nothing to suggest.
    const status = {
      ...needsInputStatus,
      result: {
        ...result,
        needs_input: {
          choices: [{ field: "dataset.mapping", options: [], suggested: null, reason: "컬럼 역할을 자동으로 정할 수 없습니다." }],
          columns: ["prompt", "completion"],
          mapping_candidates: [],
        },
      },
    };
    const { form } = renderPanel(terminal(status), { onRerun });
    expect(screen.getByText(/자동으로 제안할 매핑 후보가 없습니다\. 위 ‘컬럼 매핑’에서 역할을 직접 지정한 뒤 다시 분석하세요\./)).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    const rerun = screen.getByRole("button", { name: "선택 적용 후 다시 분석" });
    expect(rerun).toBeDisabled();
    // Roles set by hand in the mapping editor answer the question.
    act(() => {
      form().setValue("mapPrompt", "prompt");
      form().setValue("mapCompletion", "completion");
      form().setValue("mappingEnabled", true);
    });
    expect(rerun).toBeEnabled();
    await user.click(rerun);
    expect(onRerun).toHaveBeenCalled();
    expect(form().getValues("mapPrompt")).toBe("prompt");
  });

  it("starts the split over when a NEEDS_INPUT answer picks another config", async () => {
    const user = userEvent.setup();
    const result = needsInputStatus.result!;
    const ask = (choices: NonNullable<typeof result.needs_input>["choices"]) =>
      terminal({ ...needsInputStatus, result: { ...result, needs_input: { ...result.needs_input!, choices } } });
    const config = { field: "dataset.config", options: ["default", "extended"], suggested: "extended", reason: "config가 두 개입니다." };
    const split = { field: "dataset.split", options: ["train", "train_extra"], suggested: "train_extra", reason: "split이 두 개입니다." };

    // Split listed before config: the chosen split must survive the config change.
    const first = renderPanel(ask([split, config]));
    act(() => first.form().setValue("datasetSplit", "old_split"));
    await user.click(screen.getByRole("button", { name: "선택 적용 후 다시 분석" }));
    expect([first.form().getValues("datasetConfig"), first.form().getValues("datasetSplit")]).toEqual(["extended", "train_extra"]);
    cleanup();

    // Only the config is asked: the split made for the previous config goes back to auto.
    const second = renderPanel(ask([config]));
    act(() => {
      second.form().setValue("datasetConfig", "default");
      second.form().setValue("datasetSplit", "old_split");
    });
    await user.click(screen.getByRole("button", { name: "선택 적용 후 다시 분석" }));
    expect([second.form().getValues("datasetConfig"), second.form().getValues("datasetSplit")]).toEqual(["extended", ""]);
  });

  it("resolves NEEDS_INPUT inline and runs again", async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    const { form } = renderPanel(terminal(needsInputStatus), { onRerun });
    expect(screen.getByText(/임의로 추측하지 않고 분석을 멈췄습니다/)).toBeInTheDocument();
    const apply = screen.getByRole("button", { name: "선택 적용 후 다시 분석" });
    expect(apply).toBeDisabled();
    await user.selectOptions(screen.getByLabelText("dataset.mapping"), "1");
    await user.click(apply);
    expect(form().getValues("datasetSplit")).toBe("train");
    expect(form().getValues("mapPrompt")).toBe("instruction");
    expect(onRerun).toHaveBeenCalled();
  });
});
