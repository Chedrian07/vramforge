import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UseFormReturn } from "react-hook-form";
import { describe, expect, it } from "vitest";

import { AdvancedSettings } from "@/components/calculator/AdvancedSettings";
import { HardwareSection } from "@/components/calculator/HardwareSection";
import { MethodSection } from "@/components/calculator/MethodSection";
import type { FormValues } from "@/lib/form/values";

import { grpoResolved } from "../fixtures/analysis-grpo";
import { FormHarness } from "../utils/form-harness";
import { renderWithProviders } from "../utils/render";

function renderForm(node: React.ReactNode, values: Partial<FormValues> = {}) {
  let form: UseFormReturn<FormValues> | null = null;
  const utils = renderWithProviders(
    <FormHarness values={values} onReady={(f) => (form = f)}>
      {node}
    </FormHarness>,
  );
  return { ...utils, form: () => form! };
}

describe("method section", () => {
  it("switches objectives with the keyboard and explains the data transformation", async () => {
    const user = userEvent.setup();
    const { form } = renderForm(<MethodSection />);
    const group = screen.getByRole("radiogroup", { name: "Method" });
    within(group).getByRole("radio", { name: "SFT" }).focus();
    await user.keyboard("{ArrowRight}{ArrowRight} ");
    expect(form().getValues("objective")).toBe("grpo");
    expect(screen.getByText(/GRPO: prompt만 사용합니다/)).toBeInTheDocument();
  });

  it("links Load in 4-bit with the strategy and keeps Full + 4-bit impossible", async () => {
    const user = userEvent.setup();
    const { form } = renderForm(<MethodSection />);
    const sw = screen.getByRole("switch", { name: "Load in 4-bit" });
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(form().getValues("strategy")).toBe("qlora");
    await user.click(sw);
    expect(form().getValues("strategy")).toBe("lora");
    expect(form().getValues("load4bit")).toBe(false);
    await user.click(sw);
    expect(form().getValues("strategy")).toBe("qlora");
    const strategy = screen.getByRole("radiogroup", { name: "전략" });
    await user.click(within(strategy).getByRole("radio", { name: "Full" }));
    expect(form().getValues("load4bit")).toBe(false);
    expect(sw).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText(/전략: Full/)).toBeInTheDocument();
  });

  it("limits the SFT 'chosen only' transformation to preference data", () => {
    renderForm(<MethodSection />, { objective: "sft" });
    const note = screen.getByText(/^SFT:/);
    expect(note).toHaveTextContent("preference 데이터는 prompt + chosen만 학습하고 rejected는 사용하지 않습니다");
    expect(note).toHaveTextContent("prompt-completion·messages·text 데이터는 매핑한 내용을 그대로 학습합니다");
  });

  it("states the read-only full-scan policy", () => {
    renderForm(<MethodSection />);
    expect(screen.getByText("전체 데이터 분석 · Truncation 없음")).toBeInTheDocument();
  });
});

describe("hardware section", () => {
  it("defaults to capacity only and lists presets from /backend-profiles", async () => {
    const user = userEvent.setup();
    const { form } = renderForm(<HardwareSection />);
    const select = screen.getByLabelText("Hardware");
    expect(select).toHaveDisplayValue("용량만 계산");
    await waitFor(() => expect(within(select).getByRole("option", { name: /24 GiB GPU \(fixture\)/ })).toBeInTheDocument());
    await user.selectOptions(select, "vf-fixture-gpu-24gb");
    expect(form().getValues("hardwareMode")).toBe("gpu_preset");
    expect(form().getValues("gpuPresetId")).toBe("vf-fixture-gpu-24gb");
    expect(form().getValues("gpuPresetTotalBytes")).toBe(String(24 * 1_073_741_824));
    await user.selectOptions(select, "직접 입력 (GiB)");
    expect(form().getValues("hardwareMode")).toBe("custom");
    await user.type(screen.getByLabelText("GPU 전체 용량 (GiB)"), "48");
    expect(form().getValues("hardwareTotalGiB")).toBe("48");
  });
});

describe("hardware margin settings", () => {
  it("opens the collapsed margin settings when one of them is invalid", async () => {
    const user = userEvent.setup();
    renderForm(<HardwareSection />);
    const details = screen.getByText("외부 점유 · 계획용 여유 정책").closest("details")!;
    expect(details.open).toBe(false);
    await user.clear(screen.getByLabelText("여유 비율"));
    await user.type(screen.getByLabelText("여유 비율"), "abc");
    await waitFor(() => expect(details.open).toBe(true));
    expect(screen.getByText("0–10 사이 값을 입력하세요")).toBeInTheDocument();
  });
});

describe("advanced settings", () => {
  it("offers five groups operable from the keyboard", async () => {
    const user = userEvent.setup();
    renderForm(<AdvancedSettings resolved={null} datasetInspection={null} />);
    const triggers = ["Adapter", "Batch & precision", "Runtime", /Method-specific/, "Dataset & reproducibility"].map((name) =>
      screen.getByRole("button", { name: typeof name === "string" ? new RegExp(`^${name.replace(/[&]/g, "\\&")}`) : name }),
    );
    expect(triggers).toHaveLength(5);
    triggers[0]!.focus();
    await user.keyboard("{Enter}");
    expect(triggers[0]).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("rank (r)")).toBeVisible();
    await user.keyboard("{ArrowDown}");
    expect(triggers[1]).toHaveFocus();
    await user.keyboard(" ");
    expect(triggers[1]).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("load dtype")).toBeInTheDocument();
  });

  it("shows unsupported runtime options disabled with the reason", async () => {
    const user = userEvent.setup();
    renderForm(<AdvancedSettings resolved={null} datasetInspection={null} />);
    await user.click(screen.getByRole("button", { name: /^Runtime/ }));
    const packing = screen.getByRole("switch", { name: "packing" });
    expect(packing).toBeDisabled();
    expect(packing).toHaveAccessibleDescription(/엄격 무절단 모드에서는 packing을 끕니다/);
    expect(screen.getByRole("switch", { name: "torch.compile" })).toBeDisabled();
    expect(screen.getByRole("switch", { name: "parameter offload" })).toBeDisabled();
  });

  it("exposes GRPO knobs: generation unit, budget mode and reward", async () => {
    const user = userEvent.setup();
    const { form } = renderForm(<AdvancedSettings resolved={grpoResolved} datasetInspection={null} />, { objective: "grpo" });
    await user.click(screen.getByRole("button", { name: /Method-specific/ }));
    expect(screen.getByLabelText("generation_batch_size")).toHaveValue("4");
    await user.click(screen.getByRole("radio", { name: "steps_per_generation" }));
    expect(screen.getByLabelText("steps_per_generation")).toBeInTheDocument();
    expect(form().getValues("grpoGenerationUnit")).toBe("steps_per_generation");
    expect(screen.getByLabelText("budget 후보 (쉼표 구분)")).toHaveValue("1024, 2048, 4096, 8192");
    await user.click(screen.getByRole("radio", { name: "지정" }));
    await user.type(screen.getByLabelText("completion budget (token)"), "2048");
    expect(form().getValues("grpoCompletionBudget")).toBe("2048");
    await user.selectOptions(screen.getByLabelText("reward"), "local_model");
    expect(screen.getByLabelText("reward 모델")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /vLLM colocate \(미지원\)/ })).toBeDisabled();
  });

  it("shows DPO reference options and applied values after an analysis", async () => {
    const user = userEvent.setup();
    renderForm(<AdvancedSettings resolved={grpoResolved} datasetInspection={null} />, { objective: "dpo" });
    await user.click(screen.getByRole("button", { name: /Method-specific/ }));
    await user.selectOptions(screen.getByLabelText("reference 전략"), "precomputed_log_probs");
    expect(screen.getByLabelText("precompute batch")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^Batch & precision/ }));
    const applied = screen.getAllByText((_, el) => el?.tagName === "SPAN" && el.textContent === "적용값 bfloat16");
    expect(applied.length).toBeGreaterThan(0);
  });

  it("disables adapter fields for Full fine-tuning", async () => {
    const user = userEvent.setup();
    renderForm(<AdvancedSettings resolved={null} datasetInspection={null} />, { strategy: "full", load4bit: false });
    await user.click(screen.getByRole("button", { name: /^Adapter/ }));
    expect(screen.getByLabelText("rank (r)")).toBeDisabled();
  });
});

describe("dataset & reproducibility group", () => {
  it("applies the empty-system policy only with an explicit mapping", async () => {
    const user = userEvent.setup();
    const { form } = renderForm(<AdvancedSettings resolved={null} datasetInspection={null} />);
    await user.click(screen.getByRole("button", { name: /^Dataset & reproducibility/ }));
    const auto = screen.getByLabelText("빈 system 메시지");
    expect(auto).toBeDisabled();
    expect(auto).toHaveValue("omit");
    expect(auto).toHaveAccessibleDescription(/자동 감지 매핑에는 기본값\(빈 system 생략\)이 적용됩니다/);

    act(() => {
      form().setValue("mappingEnabled", true);
      form().setValue("mappingFormat", "preference");
    });
    await waitFor(() => expect(screen.getByLabelText("빈 system 메시지")).toBeEnabled());
    await user.selectOptions(screen.getByLabelText("빈 system 메시지"), "keep");
    expect(form().getValues("emptySystemPolicy")).toBe("keep");
  });

  it("summarises the mapping and the template/scope options", async () => {
    const user = userEvent.setup();
    renderForm(<AdvancedSettings resolved={null} datasetInspection={null} />, {
      mappingEnabled: true,
      mapPrompt: "question",
      mapChosen: "chosen",
    });
    await user.click(screen.getByRole("button", { name: /^Dataset & reproducibility/ }));
    expect(screen.getByText("question → prompt, chosen → chosen")).toBeInTheDocument();
    expect(screen.getByLabelText("enable_thinking (chat template)")).toHaveValue("default");
    expect(screen.getByLabelText("빈 system 메시지")).toHaveValue("omit");
    expect(screen.getByRole("switch", { name: "평가 단계 포함" })).toHaveAttribute("aria-checked", "false");
    expect(screen.getByLabelText("모델 revision")).toBeInTheDocument();
  });
});
