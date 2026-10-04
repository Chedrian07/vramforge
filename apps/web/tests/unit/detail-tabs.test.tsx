import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DetailTabs } from "@/components/results/DetailTabs";
import { formatSize } from "@/lib/format/bytes";

import { dpoResult, sftResult } from "../fixtures/analysis-dpo-sft";
import { grpoResult } from "../fixtures/analysis-grpo";
import { unknownPeakResult } from "../fixtures/analysis-states";
import { renderWithProviders } from "../utils/render";

function renderTabs(result = sftResult, extra: Partial<Parameters<typeof DetailTabs>[0]> = {}) {
  return renderWithProviders(<DetailTabs result={result} base={result} history={[]} partial={false} stale={false} {...extra} />);
}

describe("detail tabs", () => {
  it("has five keyboard-operable tabs", async () => {
    const user = userEvent.setup();
    renderTabs();
    const tabs = within(screen.getByRole("tablist", { name: "상세 결과" })).getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["메모리 구성", "데이터 길이", "단계별 피크", "비교", "적용 설정·근거"]);
    tabs[0]!.focus();
    await user.keyboard("{ArrowRight}");
    expect(tabs[1]).toHaveFocus();
    expect(tabs[1]).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("스캔 범위");
    await user.keyboard("{End}");
    expect(tabs[4]).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("요청 · 적용 · 관측 설정");
  });

  it("shows unknown and lower-bound context counts and failed rows honestly", async () => {
    const user = userEvent.setup();
    const scan = sftResult.dataset_scan!;
    const result = {
      ...sftResult,
      dataset_scan: { ...scan, rows_ok: scan.rows_seen - 2, rows_failed: 2, context_exceeded_rows: null, omitted_system_messages: 17 },
      context_validation: { ...sftResult.context_validation!, exceeded_rows: 3, exceeded_rows_exact: false },
    };
    renderTabs(result);
    await user.click(screen.getByRole("tab", { name: "데이터 길이" }));
    const coverage = screen.getByRole("region", { name: "스캔 범위" });
    expect(coverage).toHaveTextContent(`읽음 ${scan.rows_seen.toLocaleString("ko-KR")} · 실패 2`);
    expect(coverage).not.toHaveTextContent("전체 완료");
    expect(coverage).toHaveTextContent("context 초과 row산정 불가 (context 상한 미상)");
    expect(coverage).toHaveTextContent("생략한 빈 system 메시지17");
    expect(screen.getByText(/토큰화하지 못한 row 2개는 길이 통계에 없습니다/)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "context 검증" })).toHaveTextContent("초과 row3개 이상");
  });

  it("says the exceeded-row count is unknown when no context limit is known", async () => {
    const user = userEvent.setup();
    renderTabs({ ...sftResult, context_validation: { ...sftResult.context_validation!, status: "unknown", effective_limit: null, exceeded_rows: null } });
    await user.click(screen.getByRole("tab", { name: "데이터 길이" }));
    expect(screen.getByRole("region", { name: "context 검증" })).toHaveTextContent("초과 row산정 불가 (context 상한 미상)");
  });

  it("shows placeholders before an analysis", () => {
    renderWithProviders(<DetailTabs result={null} base={null} history={[]} partial={false} stale={false} />);
    expect(screen.getByRole("tabpanel")).toHaveTextContent("전체 데이터 분석을 마치면");
  });

  it("draws the peak composition of one timepoint with a matching table", () => {
    const { container } = renderTabs(dpoResult);
    const device = dpoResult.memory!.scenarios[0]!.devices[0]!;
    expect(screen.getByText(/한 시점에 동시에 살아 있는 allocation의 상한 합/)).toBeInTheDocument();
    const table = screen.getByRole("table", { name: /피크 시점 POLICY_FORWARD_BACKWARD:loss/ });
    expect(within(table).getAllByRole("row")).toHaveLength(device.peak_breakdown!.items.length + 2);
    expect(within(table).getByText("합계 (같은 시점)")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "구성 범례" })).toHaveTextContent("logits·loss");
    expect(container.querySelectorAll(".recharts-bar-rectangle").length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: /피크 시점 POLICY_FORWARD_BACKWARD:loss의 구성/ })).toBeInTheDocument();
  });

  it("does not draw unknown sizes and says so", () => {
    renderTabs(unknownPeakResult);
    expect(screen.getByText(/크기 미상 1건은 막대에 그리지 않았습니다/)).toBeInTheDocument();
    expect(screen.getAllByText("산정 불가").length).toBeGreaterThan(0);
  });

  it("switches the scenario for every tab", async () => {
    const user = userEvent.setup();
    renderTabs(grpoResult);
    const select = screen.getByLabelText("시나리오");
    await user.selectOptions(select, "budget_8192");
    const high = grpoResult.memory!.scenarios[3]!.devices[0]!.peak_breakdown!.total_high;
    expect(screen.getByRole("table", { name: new RegExp(`상한 합 ${formatSize(high)!.replace(".", "\\.")}`) })).toBeInTheDocument();
  });

  it("lists data lengths with row ids and lengths only", async () => {
    const user = userEvent.setup();
    renderTabs(dpoResult);
    await user.click(screen.getByRole("tab", { name: "데이터 길이" }));
    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getByRole("region", { name: "prompt + chosen 길이" })).toBeInTheDocument();
    expect(within(panel).getAllByText("train:2355").length).toBeGreaterThan(0);
    expect(panel).toHaveTextContent("원문은 기본으로 표시하지 않습니다");
    expect(within(panel).getByRole("region", { name: "데이터 보존 검사" })).toHaveTextContent("길이에 따른 삭제 없음");
    expect(within(panel).getAllByText("구간별 표 보기").length).toBe(3);
  });

  it("includes excluded phases with their reason", async () => {
    const user = userEvent.setup();
    renderTabs(grpoResult);
    await user.click(screen.getByRole("tab", { name: "단계별 피크" }));
    const table = screen.getByRole("table", { name: "단계별 피크 (제외한 단계 포함)" });
    expect(within(table).getAllByRole("row")).toHaveLength(10);
    expect(table).toHaveTextContent("reward 미지정: reward footprint 미포함");
    expect(table).toHaveTextContent("평가 미포함");
  });

  it("says which phases of unknown size the phase chart leaves out", async () => {
    const user = userEvent.setup();
    const scenario = dpoResult.memory!.scenarios[0]!;
    const device = scenario.devices[0]!;
    const phases = device.phases.map((p) =>
      p.phase === "POLICY_FORWARD_BACKWARD" ? { ...p, bytes_low: null, bytes_high: null, unknown_components: ["fla workspace"] } : p,
    );
    const result = {
      ...dpoResult,
      memory: { ...dpoResult.memory!, scenarios: [{ ...scenario, devices: [{ ...device, phases }] }] },
    };
    renderTabs(result);
    await user.click(screen.getByRole("tab", { name: "단계별 피크" }));
    expect(screen.getByText(/크기 미상 단계 1개\(policy forward·backward\)는 막대에 그리지/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "단계별 피크 (제외한 단계 포함)" })).toHaveTextContent("산정 불가: fla workspace");
  });

  it("says which scenarios of unknown size the comparison chart leaves out", async () => {
    const user = userEvent.setup();
    const scenarios = grpoResult.memory!.scenarios.map((s, i) =>
      i === 3 ? { ...s, devices: [{ ...s.devices[0]!, scenario_low_bytes: null, scenario_high_bytes: null }], recommendation: null } : s,
    );
    renderTabs({ ...grpoResult, memory: { ...grpoResult.memory!, scenarios } });
    await user.click(screen.getByRole("tab", { name: "비교" }));
    expect(screen.getByText(new RegExp(`산정하지 못한 시나리오 1개\\(${scenarios[3]!.label}\\)`))).toBeInTheDocument();
  });

  it("compares budget scenarios and recompute deltas", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <DetailTabs
        result={grpoResult}
        base={grpoResult}
        history={[{ base: "b", key: "k", changes: ["LoRA r 16 → 32"], request: dpoResult.requested_config, result: dpoResult }]}
        partial={false}
        stale={false}
      />,
    );
    await user.click(screen.getByRole("tab", { name: "비교" }));
    const table = screen.getByRole("table", { name: "시나리오 비교" });
    expect(within(table).getAllByRole("row")).toHaveLength(5);
    const history = screen.getByRole("table", { name: "기준 분석 대비 변경별 차이" });
    expect(history).toHaveTextContent("LoRA r 16 → 32");
  });

  it("shows requested, resolved and unverified observed settings with reasons", async () => {
    const user = userEvent.setup();
    renderTabs(grpoResult);
    await user.click(screen.getByRole("tab", { name: "적용 설정·근거" }));
    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getAllByText("미검증").length).toBeGreaterThan(5);
    expect(panel).toHaveTextContent("프로필 preset이 TRL 기본 float32 대신 bfloat16을 명시합니다.");
    expect(panel).toHaveTextContent("qwen3_5_hybrid.trl_1_14_1.analytic");
    expect(panel).toHaveTextContent("2367e865d009c13ac81713a2878291d33ab28177");
    expect(panel).toHaveTextContent("제외: reward footprint");
  });
});
