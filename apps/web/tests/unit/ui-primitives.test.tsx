import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopyButton, Segmented, SwitchField } from "@/components/ui/controls";
import { Badge } from "@/components/ui/primitives";
import { Bytes, BytesRange, IssueList } from "@/components/ui/values";

import { renderWithProviders } from "../utils/render";

function SegmentedHarness() {
  const [value, setValue] = useState<"sft" | "dpo" | "grpo">("sft");
  return (
    <>
      <Segmented label="Method" value={value} onValueChange={setValue} options={[{ value: "sft", label: "SFT" }, { value: "dpo", label: "DPO" }, { value: "grpo", label: "GRPO" }]} />
      <output>{value}</output>
    </>
  );
}

describe("ui primitives", () => {
  it("renders badges with text, not colour alone", () => {
    renderWithProviders(<Badge tone="ok">보존 확인</Badge>);
    expect(screen.getByText("보존 확인")).toBeInTheDocument();
  });

  it("operates the segmented control with the keyboard", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SegmentedHarness />);
    const sft = screen.getByRole("radio", { name: "SFT" });
    sft.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("radio", { name: "DPO" })).toHaveFocus();
    await user.keyboard(" ");
    expect(screen.getByRole("status")).toHaveTextContent("dpo");
  });

  it("toggles a labelled switch", async () => {
    const user = userEvent.setup();
    function Harness() {
      const [on, setOn] = useState(false);
      return <SwitchField id="s" label="Load in 4-bit" checked={on} onCheckedChange={setOn} />;
    }
    renderWithProviders(<Harness />);
    const sw = screen.getByRole("switch", { name: "Load in 4-bit" });
    await user.click(sw);
    expect(sw).toHaveAttribute("aria-checked", "true");
  });

  it("shows 산정 불가 for unknown sizes instead of zero", () => {
    renderWithProviders(
      <>
        <Bytes value={null} />
        <BytesRange low={null} high={5} />
      </>,
    );
    expect(screen.getAllByText("산정 불가")).toHaveLength(2);
    expect(screen.queryByText(/0\.0/)).not.toBeInTheDocument();
  });

  it("limits issue lists and points to the evidence tab", () => {
    const issues = Array.from({ length: 4 }, (_, i) => ({ code: `C${i}`, severity: "warning" as const, user_message: `경고 ${i}` }));
    renderWithProviders(<IssueList issues={issues} limit={2} />);
    expect(screen.getByText("경고 0")).toBeInTheDocument();
    expect(screen.queryByText("경고 3")).not.toBeInTheDocument();
    expect(screen.getByText(/외 2건/)).toBeInTheDocument();
  });
});

describe("copy button", () => {
  const original = Object.getOwnPropertyDescriptor(window.navigator, "clipboard");
  const originalExec = Object.getOwnPropertyDescriptor(document, "execCommand");
  afterEach(() => {
    if (original) Object.defineProperty(window.navigator, "clipboard", original);
    if (originalExec) Object.defineProperty(document, "execCommand", originalExec);
    else Reflect.deleteProperty(document, "execCommand");
  });

  function withoutClipboardApi() {
    // Plain-HTTP LAN deployments are not secure contexts: no navigator.clipboard.
    Object.defineProperty(window.navigator, "clipboard", { value: undefined, configurable: true });
  }

  it("copies with the Clipboard API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CopyButton value="2367e865d009c13ac81713a2878291d33ab28177" />);
    await user.click(screen.getByRole("button", { name: /복사: 2367e865/ }));
    expect(await screen.findByText("복사됨")).toBeInTheDocument();
    await expect(navigator.clipboard.readText()).resolves.toBe("2367e865d009c13ac81713a2878291d33ab28177");
  });

  it("falls back to a selection copy without the Clipboard API", async () => {
    const user = userEvent.setup();
    withoutClipboardApi();
    const execCommand = vi.fn(() => true);
    Object.defineProperty(document, "execCommand", { value: execCommand, configurable: true });
    renderWithProviders(<CopyButton value="local:models/" />);
    await user.click(screen.getByRole("button", { name: /복사: local:models\// }));
    expect(await screen.findByText("복사됨")).toBeInTheDocument();
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(document.querySelector("textarea")).toBeNull(); // the temporary field is gone
  });

  it("offers the value selected for a manual copy when the browser cannot copy", async () => {
    const user = userEvent.setup();
    withoutClipboardApi();
    renderWithProviders(<CopyButton value="local:models/" />);
    await user.click(screen.getByRole("button", { name: /복사: local:models\// }));
    const field = await screen.findByRole("textbox", { name: /직접 복사할 값/ });
    expect(field).toHaveValue("local:models/");
    expect(field).toHaveFocus();
    expect(screen.getByRole("status")).toHaveTextContent("자동 복사 불가");
  });
});
