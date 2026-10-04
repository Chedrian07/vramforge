import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SummaryCard, type SummaryCardProps } from "@/components/results/SummaryCard";
import type { AnalysisResult } from "@/lib/api/types";
import { formatPercent } from "@/lib/format/bytes";
import { INITIAL_RUN_STATE, type RunState } from "@/lib/hooks/useAnalysisRun";

import { dpoFit, dpoResult, sftResult } from "../fixtures/analysis-dpo-sft";
import { grpoResult } from "../fixtures/analysis-grpo";
import { completedDpoStatus, completedGrpoStatus, completedSftStatus, partialStatus, unknownPeakStatus } from "../fixtures/analysis-states";
import { renderWithProviders } from "../utils/render";

const IDLE_RUN: RunState = INITIAL_RUN_STATE;

function terminalRun(status: typeof completedGrpoStatus): RunState {
  return { ...IDLE_RUN, phase: "terminal", analysisId: status.analysis_id, jobStatus: status.status, status };
}

function view(display: AnalysisResult | null, extra: Partial<SummaryCardProps["view"]> = {}): SummaryCardProps["view"] {
  return { display, stale: false, mode: display ? "current" : "idle", reanalysisReasons: [], error: null, retry: vi.fn(), ...extra };
}

function renderCard(props: Partial<SummaryCardProps> = {}) {
  return renderWithProviders(
    <SummaryCard run={IDLE_RUN} view={view(null)} hardwareRequested={false} gpuWorkerConnected={false} exportSlot={null} {...props} />,
  );
}

describe("SummaryCard", () => {
  it("shows — / 분석 필요 before any analysis, never 0 GB or 적합", () => {
    const { container } = renderCard();
    expect(screen.getByText("전체 데이터 분석 필요")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/\b0(\.0)? ?(GB|GiB)/);
    expect(text).not.toContain("예상 적합");
    expect(text).not.toMatch(/100%/);
    expect(screen.getByText("GPU 검증 미연결")).toBeInTheDocument();
  });

  it("labels partial numbers during the scan", () => {
    renderCard({
      run: {
        ...IDLE_RUN,
        phase: "running",
        analysisId: "vf-fixture-x",
        jobStatus: "TOKENIZING",
        progress: { stage: "TOKENIZING", processed_rows: 1_536, total_rows: null },
        partial: { max_tokens: 2_001, rows_failed: 0 },
      },
    });
    expect(screen.getByText(/현재까지 확인한 데이터 기준/)).toBeInTheDocument();
    expect(screen.getByText("1,536")).toBeInTheDocument();
    expect(screen.getByText("현재까지 최대 길이")).toBeInTheDocument();
    expect(screen.getByText(/분석 진행 중/)).toBeInTheDocument();
  });

  it("labels the scanner's partial statistics in Korean", () => {
    renderCard({
      run: {
        ...IDLE_RUN,
        phase: "running",
        analysisId: "vf-fixture-x",
        jobStatus: "TOKENIZING",
        progress: { stage: "TOKENIZING", processed_rows: 1_536, total_rows: null },
        partial: { status: "partial", rows_ok: 1_535, rows_failed: 1, max_prompt: 201, max_chosen_sequence: 1_843 },
      },
    });
    const box = within(screen.getByText(/현재까지 확인한 데이터 기준/).parentElement!);
    expect(box.getByText("스캔 범위")).toBeInTheDocument();
    // A failed row so far: how many rows were read and failed, not just the coverage word.
    expect(box.getByText("부분 · 읽음 1,536 · 실패 1")).toBeInTheDocument();
    expect(box.getByText("현재까지 최대 길이 · prompt")).toBeInTheDocument();
    expect(box.getByText("현재까지 최대 길이 · prompt + chosen")).toBeInTheDocument();
    expect(box.getByText("1,843")).toBeInTheDocument();
    expect(box.queryByText("max_prompt")).not.toBeInTheDocument();
  });

  it("never reports a scan with failed rows as complete", () => {
    renderCard({
      run: {
        ...IDLE_RUN,
        phase: "running",
        analysisId: "vf-fixture-x",
        jobStatus: "TOKENIZING",
        progress: { stage: "TOKENIZING", processed_rows: 4_656, total_rows: 4_656 },
        // The scanner's final report: every row read, two of them failed.
        partial: { status: "complete", rows_ok: 4_654, rows_failed: 2, context_exceeded_rows: null },
      },
    });
    const box = within(screen.getByText(/현재까지 확인한 데이터 기준/).parentElement!);
    expect(box.getByText("읽음 4,656 · 실패 2")).toBeInTheDocument();
    expect(box.queryByText("전체 완료")).not.toBeInTheDocument();
    expect(box.getByText("산정 불가 (context 상한 미상)")).toBeInTheDocument();

    const failedRows: AnalysisResult = { ...sftResult, dataset_scan: { ...sftResult.dataset_scan!, rows_ok: sftResult.dataset_scan!.rows_seen - 3, rows_failed: 3 } };
    renderCard({ run: terminalRun({ ...completedSftStatus, result: failedRows }), view: view(failedRows) });
    const list = screen.getAllByRole("list", { name: "결과 상태" }).at(-1)!;
    expect(list).toHaveTextContent(`읽음 ${failedRows.dataset_scan!.rows_seen.toLocaleString("ko-KR")} · 실패 3`);
    expect(list).not.toHaveTextContent("전체 완료");
  });

  it("shows five separate status badges with text after completion", () => {
    renderCard({ run: terminalRun(completedSftStatus), view: view(sftResult) });
    const list = screen.getByRole("list", { name: "결과 상태" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(5);
    expect(list).toHaveTextContent("스캔 범위");
    expect(list).toHaveTextContent("전체 완료");
    expect(list).toHaveTextContent("데이터 보존");
    expect(list).toHaveTextContent("보존 확인");
    expect(list).toHaveTextContent("학습 준비");
    expect(list).toHaveTextContent("준비됨");
    expect(list).toHaveTextContent("추정 근거");
    expect(list).toHaveTextContent("정적 추정");
    expect(list).toHaveTextContent("GPU 적합");
    expect(list).toHaveTextContent("판정 안 함");
  });

  it("points to the assumptions behind the numbers", () => {
    renderCard({ run: terminalRun(completedSftStatus), view: view(sftResult) });
    const count = new Set([...(sftResult.assumptions ?? []), ...(sftResult.memory?.assumptions ?? [])].map((a) => a.id)).size;
    expect(count).toBeGreaterThan(0);
    expect(screen.getByText(new RegExp(`명시한 가정 ${count}건에 기대고 있습니다`))).toBeInTheDocument();
  });

  it("hides the GPU gauge when no hardware was selected", () => {
    renderCard({ run: terminalRun(completedSftStatus), view: view(sftResult) });
    expect(screen.queryByRole("img", { name: /GPU 사용량/ })).not.toBeInTheDocument();
    expect(screen.getByText(/용량만 표시합니다/)).toBeInTheDocument();
  });

  it("caps the gauge at 100% but keeps the over-capacity percentage as text", () => {
    renderCard({ run: terminalRun(completedDpoStatus), view: view(dpoResult) });
    const percent = formatPercent(dpoFit.utilization_ratio)!;
    expect(Number(percent.replace("%", ""))).toBeGreaterThan(100);
    const gauge = screen.getByRole("img", { name: new RegExp(`예상 ${percent}`) });
    expect((gauge.firstChild as HTMLElement).style.width).toBe("100%");
    expect(screen.getByText(`예상 ${percent}`)).toBeInTheDocument();
    expect(screen.getByText("예상 용량 초과, 실측/설정 검토 필요")).toBeInTheDocument();
  });

  it("lists every GRPO budget with the reward footprint warning", () => {
    renderCard({ run: terminalRun(completedGrpoStatus), view: view(grpoResult) });
    const table = screen.getByRole("table", { name: /completion budget별/ });
    const rows = within(table).getAllByRole("row");
    expect(rows.map((r) => within(r).queryByRole("rowheader")?.textContent).filter(Boolean)).toEqual(["1,024", "2,048", "4,096", "8,192"]);
    expect(screen.getByText(/completion budget별 예상 피크 \(1,024–8,192 token\)/)).toBeInTheDocument();
    expect(screen.getByText("reward footprint 미포함")).toBeInTheDocument();
  });

  it("renders 산정 불가 with the reason when the peak is unknown", () => {
    renderCard({ run: terminalRun(unknownPeakStatus), view: view(unknownPeakStatus.result!) });
    expect(screen.getAllByText("산정 불가").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/linear-attention fla workspace/).length).toBeGreaterThan(0);
    expect(screen.getByText("판정 보류 (미확정 footprint)")).toBeInTheDocument();
  });

  it("marks values from an earlier setting while recomputing", () => {
    renderCard({ run: terminalRun(completedSftStatus), view: view(sftResult, { stale: true, mode: "loading" }) });
    expect(screen.getByRole("status")).toHaveTextContent("이전 설정");
  });

  it("flags partial jobs and explains the missing estimate", () => {
    renderCard({ run: terminalRun(partialStatus), view: view(null) });
    expect(screen.getByText("전체 데이터 스캔이 끝나지 않아 메모리를 산정하지 않았습니다.")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "결과 상태" })).toHaveTextContent("부분");
  });
});
