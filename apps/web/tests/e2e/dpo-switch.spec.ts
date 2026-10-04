import { expect, test } from "@playwright/test";

import { ANALYSIS_TIMEOUT, loadExample, method, startAnalysis, summary, waitForCompletion } from "./helpers";

test("switching GRPO -> DPO asks for re-analysis and then shows a single DPO estimate", async ({ page }) => {
  test.setTimeout(2 * ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  await startAnalysis(page);
  await waitForCompletion(page);

  await method(page, "DPO").click();
  const rerun = page.getByRole("button", { name: "데이터 재분석 필요" });
  await expect(rerun).toBeVisible();
  await expect(page.getByText("학습 방식 변경 (GRPO → DPO)")).toBeVisible();
  await expect(summary(page).getByText(/이전 설정/)).toBeVisible();

  await rerun.click();
  await expect(page.getByRole("button", { name: "분석 진행 중…" })).toBeVisible();
  await waitForCompletion(page);
  await expect(summary(page).getByText("예상 피크 범위")).toBeVisible();
  await expect(summary(page).getByRole("table", { name: /completion budget별/ })).toHaveCount(0);
  await page.getByRole("tab", { name: "데이터 길이" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("prompt + rejected");
});
