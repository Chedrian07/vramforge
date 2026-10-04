import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ExportMenu, RetentionNote, type ExportTarget } from "@/components/results/ExportMenu";
import { ApiError, type ApiClient } from "@/lib/api/client";
import type { AnalysisResult, JobStatus } from "@/lib/api/types";

import { dpoResult } from "../fixtures/analysis-dpo-sft";
import { grpoResult } from "../fixtures/analysis-grpo";
import { renderWithProviders } from "../utils/render";

type ExportApi = Pick<ApiClient, "exportUrl" | "exportScenario">;

function makeApi(exportScenario: ApiClient["exportScenario"] = vi.fn()): ExportApi {
  return { exportUrl: (id, f) => `/api/v1/analyses/${id}/export?format=${f}`, exportScenario };
}

function stored(result: AnalysisResult | null, jobStatus: JobStatus | null, formDiffers = false): ExportTarget {
  return { kind: "stored", result, jobStatus, formDiffers };
}

async function open(target: ExportTarget, id: string | null, api: ExportApi = makeApi(), expiresAt: string | null = null) {
  const user = userEvent.setup();
  renderWithProviders(<ExportMenu api={api} analysisId={id} target={target} expiresAt={expiresAt} />);
  await user.click(screen.getByRole("button", { name: "결과 내보내기" }));
  return { menu: screen.getByRole("dialog"), user };
}

const createObjectURL = vi.fn((_blob: Blob) => "blob:vf-fixture-export");
const revokeObjectURL = vi.fn();
let clicked: Array<{ href: string; download: string }> = [];

beforeEach(() => {
  clicked = [];
  Object.assign(URL, { createObjectURL, revokeObjectURL });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    clicked.push({ href: this.href, download: this.download });
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  createObjectURL.mockClear();
});

describe("ExportMenu", () => {
  it("disables every export before an analysis exists", async () => {
    const { menu } = await open(stored(null, null), null);
    const disabled = within(menu).getAllByRole("button");
    expect(disabled).toHaveLength(4);
    disabled.forEach((b) => expect(b).toHaveAttribute("aria-disabled", "true"));
    expect(within(menu).getAllByText("분석이 끝난 뒤 내보낼 수 있습니다.").length).toBe(4);
    expect(within(menu).queryByRole("note")).not.toBeInTheDocument();
  });

  it("keeps trainer-config disabled with the reason for a conditional result", async () => {
    const { menu } = await open(stored(grpoResult, "COMPLETED"), grpoResult.analysis_id);
    expect(within(menu).getByRole("link", { name: /analysis.json/ })).toHaveAttribute("href", `/api/v1/analyses/${grpoResult.analysis_id}/export?format=json`);
    expect(within(menu).getByRole("link", { name: /resolved-plan.yaml/ })).toBeInTheDocument();
    expect(within(menu).getByRole("link", { name: /report.md/ })).toBeInTheDocument();
    const trainer = within(menu).getByRole("button", { name: /trainer-config.yaml/ });
    expect(trainer).toHaveAttribute("aria-disabled", "true");
    expect(trainer).toHaveAccessibleDescription(/조건부/);
  });

  it("offers trainer-config when the result is ready and names the stored analysis as the source", async () => {
    const { menu } = await open(stored(dpoResult, "COMPLETED"), dpoResult.analysis_id);
    expect(within(menu).getByRole("link", { name: /trainer-config.yaml/ })).toHaveAttribute("href", `/api/v1/analyses/${dpoResult.analysis_id}/export?format=trainer-config`);
    expect(within(menu).getByRole("note")).toHaveTextContent("내보낼 결과: 서버에 저장된 분석");
    expect(within(menu).getByRole("note")).not.toHaveTextContent("바뀐 입력");
  });

  it("needs one explicit GRPO completion budget for trainer-config even when ready", async () => {
    // A local reward model on the training GPU makes the result ready; budget candidates are a
    // planning aid, not one max_completion_length (exports/trainer_config.py check_ready).
    const ready: AnalysisResult = {
      ...grpoResult,
      status: { ...grpoResult.status!, training_readiness: "ready" },
      resolved_config: { ...grpoResult.resolved_config!, grpo: { ...grpoResult.resolved_config!.grpo!, reward_kind: "local_model" } },
    };
    const { menu } = await open(stored(ready, "COMPLETED"), ready.analysis_id);
    const trainer = within(menu).getByRole("button", { name: /trainer-config.yaml/ });
    expect(trainer).toHaveAttribute("aria-disabled", "true");
    expect(trainer).toHaveAccessibleDescription(/completion budget을 하나로 지정해야/);
    expect(within(menu).getByRole("link", { name: /resolved-plan.yaml/ })).toBeInTheDocument();
  });

  it("says the changed inputs are not in the stored analysis while no scenario is shown", async () => {
    const { menu } = await open(stored(dpoResult, "COMPLETED", true), dpoResult.analysis_id);
    expect(within(menu).getByRole("note")).toHaveTextContent("서버에 저장된 분석");
    expect(within(menu).getByRole("note")).toHaveTextContent("화면의 바뀐 입력은 들어가지 않으니");
  });

  it("exports the displayed scenario by posting the request it was computed for", async () => {
    const exportScenario = vi.fn<ApiClient["exportScenario"]>(async () => ({ blob: new Blob(["# report"]), filename: "report.md" }));
    const target: ExportTarget = { kind: "scenario", result: dpoResult, request: dpoResult.requested_config, changes: ["LoRA r 16 → 32"], stale: false };
    const { menu, user } = await open(target, "vf-fixture-base", makeApi(exportScenario));
    expect(within(menu).getByRole("note")).toHaveTextContent("내보낼 결과: 화면에 표시된 재계산 시나리오 (LoRA r 16 → 32)");
    // No GET links: those would download the stored analysis instead.
    expect(within(menu).queryByRole("link")).not.toBeInTheDocument();

    await user.click(within(menu).getByRole("button", { name: /report.md/ }));
    await waitFor(() => expect(clicked).toHaveLength(1));
    expect(exportScenario).toHaveBeenCalledWith("vf-fixture-base", { request: dpoResult.requested_config, format: "md" });
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(clicked[0]).toEqual({ href: "blob:vf-fixture-export", download: "report.md" });
  });

  it("names an earlier-setting scenario and shows why an export was refused", async () => {
    const exportScenario = vi.fn<ApiClient["exportScenario"]>(async () => {
      throw new ApiError(409, { code: "REANALYSIS_REQUIRED", severity: "error", retryable: false, user_message: "데이터 재분석이 필요합니다." });
    });
    const target: ExportTarget = { kind: "scenario", result: dpoResult, request: dpoResult.requested_config, changes: ["LoRA r 16 → 32"], stale: true };
    const { menu, user } = await open(target, "vf-fixture-base", makeApi(exportScenario));
    expect(within(menu).getByRole("note")).toHaveTextContent("이전 설정의 결과");
    await user.click(within(menu).getByRole("button", { name: /analysis.json/ }));
    expect(await within(menu).findByRole("alert")).toHaveTextContent("데이터 재분석이 필요합니다.");
    expect(clicked).toHaveLength(0);
  });

  it("shows how long the result is kept", async () => {
    const { menu } = await open(stored(dpoResult, "COMPLETED"), dpoResult.analysis_id, makeApi(), "2026-10-12T05:00:00Z");
    expect(within(menu).getByText(/결과 보관 기한/)).toBeInTheDocument();
    expect(menu.querySelector("time")).toHaveAttribute("dateTime", "2026-10-12T05:00:00Z");
  });
});

describe("RetentionNote", () => {
  it("renders the expiry and nothing when it is unknown", () => {
    const { container, rerender } = render(<RetentionNote expiresAt="2026-10-12T05:00:00Z" />);
    expect(container).toHaveTextContent(/결과 보관 기한 .+까지 · 이후 결과와 길이 artifact가 자동 삭제됩니다./);
    rerender(<RetentionNote expiresAt={null} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<RetentionNote expiresAt="not a date" />);
    expect(container).toBeEmptyDOMElement();
  });
});
