import { expect, test } from "@playwright/test";

import { ANALYSIS_TIMEOUT, isDevMock, loadExample, startAnalysis, summary, waitForCompletion } from "./helpers";

// plan.md §21 example: GRPO + Load in 4-bit, no completion budget, reward unspecified.
test("GRPO example: full scan, budget scenarios and the reward warning", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  await expect(summary(page).getByText("전체 데이터 분석 필요")).toBeVisible();
  await expect(summary(page)).not.toContainText(/\b0(\.0)? ?(GB|GiB)\b/);

  await startAnalysis(page);
  await expect(page.getByRole("list", { name: "분석 단계" })).toBeVisible();
  await waitForCompletion(page);

  const card = summary(page);
  const budgets = card.getByRole("table", { name: /completion budget별/ });
  await expect(budgets).toBeVisible();
  for (const budget of ["1,024", "2,048", "4,096", "8,192"]) {
    await expect(budgets.getByRole("rowheader", { name: budget })).toBeVisible();
  }
  await expect(card.getByText("reward footprint 미포함")).toBeVisible();
  await expect(card.getByText("GPU 검증 미연결")).toBeVisible();
  await expect(card.getByRole("list", { name: "결과 상태" })).toContainText("조건부");
  await expect(card.getByRole("img", { name: /GPU 사용량/ })).toHaveCount(0);

  await page.getByRole("tab", { name: "데이터 길이" }).click();
  await expect(page.getByRole("tabpanel")).toContainText("전체 완료");
  await expect(page.getByRole("tabpanel")).toContainText("원문은 기본으로 표시하지 않습니다");

  // A reload reconnects to the stored analysis through ?analysis=<id> (the dev mock keeps runs in
  // page memory only, so this part needs the real API).
  if (await isDevMock(page)) return;
  await page.reload();
  await expect(summary(page).getByRole("table", { name: /completion budget별/ })).toBeVisible({ timeout: 60_000 });
});
