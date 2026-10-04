import { expect, test, type Page } from "@playwright/test";

import { ANALYSIS_TIMEOUT, exportItem, loadExample, openApp, startAnalysis, waitForCompletion } from "./helpers";

const TABS = ["메모리 구성", "데이터 길이", "단계별 피크", "비교", "적용 설정·근거"];

async function expectSelectedTab(page: Page, name: string) {
  const tab = page.getByRole("tab", { name });
  await expect(tab).toBeFocused();
  await expect(tab).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tabpanel", { name })).toBeVisible();
}

test("method, advanced groups, tabs and export are keyboard operable", async ({ page }) => {
  await openApp(page);
  const sft = page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: "SFT" });
  await sft.focus();
  await page.keyboard.press("ArrowRight");
  // Radix roving focus moves focus asynchronously; wait like a user would before selecting.
  await expect(page.getByRole("radio", { name: "DPO" })).toBeFocused();
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

test("result tabs and exports work from the keyboard after an analysis", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page, "DPO");
  await startAnalysis(page);
  await waitForCompletion(page);

  // One tab stop for the tab list; arrows move and select (wrapping), Home/End jump.
  const first = page.getByRole("tab", { name: TABS[0] });
  await first.focus();
  for (const name of TABS.slice(1)) {
    await page.keyboard.press("ArrowRight");
    await expectSelectedTab(page, name);
  }
  await page.keyboard.press("ArrowRight");
  await expectSelectedTab(page, TABS[0]!);
  await page.keyboard.press("ArrowLeft");
  await expectSelectedTab(page, TABS.at(-1)!);
  await page.keyboard.press("Home");
  await expectSelectedTab(page, TABS[0]!);
  await page.keyboard.press("End");
  await expectSelectedTab(page, TABS.at(-1)!);

  // Tab leaves the tab list for the panel content, Shift+Tab comes back to the selected tab.
  await page.keyboard.press("Tab");
  await expect(page.getByRole("tab", { name: TABS.at(-1)! })).not.toBeFocused();
  expect(await page.evaluate(() => document.activeElement?.closest('[role="tabpanel"]') != null)).toBe(true);
  await page.keyboard.press("Shift+Tab");
  await expect(page.getByRole("tab", { name: TABS.at(-1)! })).toBeFocused();

  // The export menu opens from the keyboard and its files are reachable with Tab.
  const exportButton = page.getByRole("button", { name: "결과 내보내기" });
  await exportButton.focus();
  await page.keyboard.press("Enter");
  const menu = page.getByRole("dialog", { name: "결과 내보내기" });
  await expect(menu).toBeVisible();
  const json = exportItem(menu, "analysis.json");
  for (let i = 0; i < 6 && !(await json.evaluate((el) => el === document.activeElement)); i++) await page.keyboard.press("Tab");
  await expect(json).toBeFocused();
  const download = page.waitForEvent("download");
  await page.keyboard.press("Enter");
  expect((await download).suggestedFilename()).toBe("analysis.json");
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(exportButton).toBeFocused();
});
