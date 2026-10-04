import { expect, test } from "@playwright/test";

import {
  ANALYSIS_TIMEOUT,
  EXAMPLE_DPO_PAIR_MAX,
  EXAMPLE_ROWS,
  analysisIdOf,
  isDevMock,
  loadExample,
  method,
  openTab,
  startAnalysis,
  stat,
  statusAxis,
  summary,
  waitForCompletion,
} from "./helpers";

// plan.md §21 "DPO로 변경": the same snapshot is re-analysed with preference preprocessing, both
// full branches are measured, and a fully resolved DPO result offers a planning capacity.
test("switching GRPO -> DPO asks for re-analysis and then shows a ready DPO estimate", async ({ page }) => {
  test.setTimeout(2 * ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  await startAnalysis(page);
  await waitForCompletion(page);
  const grpoId = analysisIdOf(page);

  await method(page, "DPO").click();
  const rerun = page.getByRole("button", { name: "데이터 재분석 필요" });
  await expect(rerun).toBeVisible();
  await expect(page.getByText("학습 방식 변경 (GRPO → DPO)")).toBeVisible();
  await expect(summary(page).getByText(/이전 설정/)).toBeVisible();

  await rerun.click();
  await expect(page.getByRole("button", { name: "분석 진행 중…" })).toBeVisible();
  await expect(page).not.toHaveURL(new RegExp(`analysis=${grpoId}`));
  await waitForCompletion(page);
  expect(analysisIdOf(page)).not.toBe(grpoId);

  // One DPO estimate (no budget scenarios), ready, with a planning capacity.
  const card = summary(page);
  await expect(card.getByText("예상 피크 범위")).toBeVisible();
  await expect(card.getByRole("table", { name: /completion budget별/ })).toHaveCount(0);
  await expect(card.getByRole("note").filter({ hasText: "reward footprint" })).toHaveCount(0);
  await expect(statusAxis(page, "학습 준비")).toContainText("준비됨");
  await expect(statusAxis(page, "스캔 범위")).toContainText("전체 완료");
  await expect(stat(card, "계획용 권장 용량")).toContainText(/^\d+\.\d GiB/);
  await expect(stat(card, "계획용 권장 용량")).toContainText("상한 + 여유");
  await expect(stat(card, "확정 상주량")).toContainText(/^\d+\.\d GiB/);
  await expect(statusAxis(page, "GPU 적합")).toContainText("판정 안 함");
  await expect(card.getByRole("img", { name: /GPU 사용량/ })).toHaveCount(0);

  const data = await openTab(page, "데이터 길이");
  await expect(data).toContainText("prompt + rejected");
  if (await isDevMock(page)) return; // the facts below are the real snapshot's
  await expect(stat(data.getByRole("region", { name: "스캔 범위" }), "성공 row")).toHaveText(EXAMPLE_ROWS);
  const pair = data.getByRole("region", { name: "pair 최대 길이" });
  await expect(stat(pair, "row 수")).toHaveText(EXAMPLE_ROWS);
  await expect(stat(pair, "최대")).toHaveText(new RegExp(`^${EXAMPLE_DPO_PAIR_MAX}\\s*\\(train:\\d+\\)$`));
  for (const branch of ["prompt + chosen 길이", "prompt + rejected 길이"]) {
    await expect(stat(data.getByRole("region", { name: branch }), "row 수")).toHaveText(EXAMPLE_ROWS);
  }
});
