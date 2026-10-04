import { readFile } from "node:fs/promises";

import { expect, test } from "@playwright/test";

import { ANALYSIS_TIMEOUT, isDevMock, loadExample, startAnalysis, waitForCompletion } from "./helpers";

test("exports analysis.json for the stored analysis and gates trainer-config", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  test.skip(await isDevMock(page), "the dev mock mode has no export endpoint");
  await startAnalysis(page);
  await waitForCompletion(page);
  const analysisId = new URL(page.url()).searchParams.get("analysis");
  expect(analysisId).toBeTruthy();

  await page.getByRole("button", { name: "결과 내보내기" }).click();
  const menu = page.getByRole("dialog", { name: "결과 내보내기" });
  // GRPO without a reward is conditional: no runnable trainer config is offered (plan.md §12.4).
  const trainer = menu.getByRole("button", { name: /trainer-config.yaml/ });
  await expect(trainer).toHaveAttribute("aria-disabled", "true");
  await expect(trainer).toContainText("조건부");

  const downloadPromise = page.waitForEvent("download");
  await menu.getByRole("link", { name: /analysis.json/ }).click();
  const download = await downloadPromise;
  const text = await readFile((await download.path())!, "utf8");
  expect(text).toContain(analysisId!);
  expect(() => JSON.parse(text)).not.toThrow();
  // No host paths or credentials in exports (plan.md §12.4, §18).
  expect(text).not.toMatch(/\/Users\/|\/home\/[a-z]|hf_[A-Za-z0-9]{20,}/);
});
