import { expect, test } from "@playwright/test";

test("method, advanced groups, tabs and export are keyboard operable", async ({ page }) => {
  await page.goto("/");
  const sft = page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: "SFT" });
  await sft.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Space");
  await expect(page.getByRole("radio", { name: "DPO" })).toHaveAttribute("aria-checked", "true");

  const adapter = page.getByRole("button", { name: /^Adapter/ });
  await adapter.focus();
  await page.keyboard.press("Enter");
  await expect(adapter).toHaveAttribute("aria-expanded", "true");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("button", { name: /^Batch & precision/ })).toBeFocused();

  const firstTab = page.getByRole("tab", { name: "메모리 구성" });
  await firstTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "데이터 길이" })).toHaveAttribute("aria-selected", "true");

  const exportButton = page.getByRole("button", { name: "결과 내보내기" });
  await exportButton.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog", { name: "결과 내보내기" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "결과 내보내기" })).toBeHidden();
});
