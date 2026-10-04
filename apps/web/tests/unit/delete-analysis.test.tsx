import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DeleteAnalysis } from "@/components/results/DeleteAnalysis";
import { ApiError } from "@/lib/api/client";

import { renderWithProviders } from "../utils/render";

describe("delete analysis", () => {
  it("deletes only after confirmation", async () => {
    const user = userEvent.setup();
    const deleteAnalysis = vi.fn(async () => undefined);
    const onDeleted = vi.fn();
    renderWithProviders(<DeleteAnalysis api={{ deleteAnalysis }} analysisId="vf-fixture-a" onDeleted={onDeleted} />);
    await user.click(screen.getByRole("button", { name: "결과 삭제" }));
    expect(screen.getByRole("alertdialog", { name: "분석 결과를 삭제할까요?" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "취소" }));
    expect(deleteAnalysis).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "결과 삭제" }));
    await user.click(screen.getByRole("button", { name: "삭제" }));
    await waitFor(() => expect(onDeleted).toHaveBeenCalled());
    expect(deleteAnalysis).toHaveBeenCalledWith("vf-fixture-a");
  });

  it("keeps the dialog open with the server message on failure", async () => {
    const user = userEvent.setup();
    const deleteAnalysis = vi.fn(async () => {
      throw new ApiError(403, { code: "FORBIDDEN", severity: "error", retryable: false, user_message: "다른 사용자의 분석은 삭제할 수 없습니다." });
    });
    renderWithProviders(<DeleteAnalysis api={{ deleteAnalysis }} analysisId="vf-fixture-a" onDeleted={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "결과 삭제" }));
    await user.click(screen.getByRole("button", { name: "삭제" }));
    expect(await screen.findByText("다른 사용자의 분석은 삭제할 수 없습니다.")).toBeInTheDocument();
  });

  it("renders nothing without an analysis", () => {
    const { container } = renderWithProviders(<DeleteAnalysis api={{ deleteAnalysis: vi.fn() }} analysisId={null} onDeleted={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});
