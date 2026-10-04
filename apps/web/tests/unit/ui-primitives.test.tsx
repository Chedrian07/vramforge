import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { Segmented, SwitchField } from "@/components/ui/controls";
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
