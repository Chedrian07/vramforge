import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";

import { ProgressPanel, stepStates } from "@/components/calculator/ProgressPanel";
import type { FormValues } from "@/lib/form/values";
import type { RunState } from "@/lib/hooks/useAnalysisRun";

import { cancelledStatus, completedGrpoStatus, failedStatus, needsInputStatus, partialStatus } from "../fixtures/analysis-states";
import { FormHarness } from "../utils/form-harness";
import { renderWithProviders } from "../utils/render";

const BASE: RunState = {
  phase: "idle",
  analysisId: null,
  jobStatus: null,
  progress: null,
  partial: null,
  liveIssues: [],
  status: null,
  error: null,
  connection: "idle",
  submittedRequest: null,
  cancelling: false,
};

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
