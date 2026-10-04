import { readFile } from "node:fs/promises";

import { expect, test } from "@playwright/test";

import type { AnalysisResult } from "../../lib/api/types";
import { formatGiBRange } from "../../lib/format/bytes";

import { ANALYSIS_TIMEOUT, isDevMock, loadExample, startAnalysis, summary, waitForCompletion } from "./helpers";

// plan.md §19.4 "결과 export": byte numbers, fingerprints, settings and warnings of the exported
// analysis match what the page shows.
test("exports analysis.json that matches the displayed result and gates trainer-config", async ({ page }) => {
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
  // No host paths or credentials in exports (plan.md §12.4, §18).
  expect(text).not.toMatch(/\/Users\/|\/home\/[a-z]|hf_[A-Za-z0-9]{20,}/);
  const exported = JSON.parse(text) as AnalysisResult;
  expect(exported.analysis_id).toBe(analysisId);
  await page.keyboard.press("Escape");

  // Byte numbers: every budget row shows the exported scenario range.
  const budgets = summary(page).getByRole("table", { name: /completion budget별/ });
  const scenarios = exported.memory?.scenarios ?? [];
  expect(scenarios.length).toBeGreaterThan(1);
  for (const scenario of scenarios) {
    const device = scenario.devices[0]!;
    const range = formatGiBRange(device.scenario_low_bytes, device.scenario_high_bytes);
    if (range) await expect(budgets).toContainText(range);
  }

  // Settings, fingerprint and warnings as shown in the evidence tab.
  expect(exported.requested_config.training.objective).toBe("grpo");
  expect(exported.requested_config.training.strategy).toBe("qlora");
  await page.getByRole("tab", { name: "적용 설정·근거" }).click();
  const evidence = page.getByRole("tabpanel");
  await expect(evidence).toContainText(exported.analysis_fingerprint);
  for (const warning of exported.warnings ?? []) {
    await expect(evidence.getByRole("region", { name: "경고와 오류" })).toContainText(warning.user_message);
  }
});
