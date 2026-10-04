import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Header } from "@/components/layout/Header";
import { TokenPrompt } from "@/components/layout/TokenPrompt";
import { ApiError } from "@/lib/api/client";

import { makeEnvironment, renderWithProviders } from "../utils/render";

afterEach(() => {
  window.localStorage.clear();
  document.documentElement.classList.remove("dark");
});

describe("header", () => {
  it("shows the title, subtitle, theme toggle and help", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Header />);
    expect(screen.getByRole("heading", { level: 1, name: "Fine-Tuning VRAM Calculator" })).toBeInTheDocument();
    expect(screen.getByText("데이터셋을 자르지 않고 학습할 때 필요한 GPU 메모리")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "도움말" }));
    expect(screen.getByRole("dialog", { name: "계산기 사용법" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("switches light, dark and system themes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Header />);
    const toggle = screen.getByRole("radiogroup", { name: "테마" });
    expect(screen.getByRole("radio", { name: "시스템" })).toHaveAttribute("aria-checked", "true");
    await user.click(screen.getByRole("radio", { name: "다크" }));
    expect(document.documentElement).toHaveClass("dark");
    expect(window.localStorage.getItem("vf-theme")).toBe("dark");
    await user.click(screen.getByRole("radio", { name: "라이트" }));
    expect(document.documentElement).not.toHaveClass("dark");
    await user.click(screen.getByRole("radio", { name: "시스템" }));
    expect(window.localStorage.getItem("vf-theme")).toBeNull();
    expect(toggle).toBeInTheDocument();
  });
});

describe("access token prompt", () => {
  it("opens on 401 and posts the token without storing it", async () => {
    const user = userEvent.setup();
    const createSession = vi.fn(async () => undefined);
    const environment = makeEnvironment({ createSession });
    renderWithProviders(<TokenPrompt />, { environment });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    environment.authGate.request();
    const dialog = await screen.findByRole("dialog", { name: "접근 토큰 필요" });
    await user.type(screen.getByLabelText("접근 토큰"), "s3cret-token");
    await user.click(screen.getByRole("button", { name: "확인" }));
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(createSession).toHaveBeenCalledWith("s3cret-token");
    expect(JSON.stringify(window.localStorage)).not.toContain("s3cret");
    expect(document.cookie).not.toContain("s3cret");
  });

  it("explains a rejected token", async () => {
    const user = userEvent.setup();
    const createSession = vi.fn(async () => {
      throw new ApiError(401, { code: "UNAUTHORIZED", severity: "error", retryable: false, user_message: "x" });
    });
    const environment = makeEnvironment({ createSession });
    renderWithProviders(<TokenPrompt />, { environment });
    environment.authGate.request();
    await user.type(await screen.findByLabelText("접근 토큰"), "wrong");
    await user.click(screen.getByRole("button", { name: "확인" }));
    expect(await screen.findByText("토큰이 올바르지 않습니다.")).toBeInTheDocument();
  });
});
