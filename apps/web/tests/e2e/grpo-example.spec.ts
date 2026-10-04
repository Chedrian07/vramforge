import { expect, test } from "@playwright/test";

import {
  ANALYSIS_TIMEOUT,
  EXAMPLE_DATASET_REVISION,
  EXAMPLE_MAX_PROMPT,
  EXAMPLE_ROWS,
  isDevMock,
  loadExample,
  openTab,
  startAnalysis,
  stat,
  statusAxis,
  summary,
  waitForCompletion,
} from "./helpers";

// plan.md §21 example: GRPO + Load in 4-bit, Advanced unchanged, no completion budget, reward
// unspecified, no hardware selected.
test("GRPO example: progress, full coverage, budget scenarios, reward warning and no gauge", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  await expect(summary(page).getByText("전체 데이터 분석 필요")).toBeVisible();
  await expect(summary(page)).not.toContainText(/\b0(\.0)? ?(GB|GiB)\b/);

  await startAnalysis(page);
  const steps = page.getByRole("list", { name: "분석 단계" });
  await expect(steps).toBeVisible();
  await waitForCompletion(page);
  await expect(steps.getByRole("listitem")).toHaveText([/구조 확인\s*완료/, /데이터 토큰화\s*완료/, /배치 분석\s*완료/, /메모리 산정\s*완료/]);

  // Four completion-budget scenarios (1K/2K/4K/8K), each with a peak range and a capacity.
  const card = summary(page);
  const budgets = card.getByRole("table", { name: /completion budget별/ });
  await expect(budgets).toBeVisible();
  await expect(budgets.getByRole("rowheader")).toHaveText(["1,024", "2,048", "4,096", "8,192"]);
  for (const row of await budgets.getByRole("row").filter({ has: page.getByRole("rowheader") }).all()) {
    await expect(row.getByRole("cell").nth(0)).toHaveText(/^\d+\.\d( – \d+\.\d)? GiB$/);
    await expect(row.getByRole("cell").nth(1)).toHaveText(/^\d+\.\d GiB$/);
  }

  // Reward unspecified: a conditional result with the reward footprint excluded.
  await expect(card.getByRole("note").filter({ hasText: "reward footprint 미포함" })).toBeVisible();
  await expect(statusAxis(page, "학습 준비")).toContainText("조건부");
  await expect(statusAxis(page, "스캔 범위")).toContainText("전체 완료");
  await expect(card.getByText("GPU 검증 미연결")).toBeVisible();

  // No hardware selected: capacities only, no fit verdict and no usage gauge.
  await expect(statusAxis(page, "GPU 적합")).toContainText("판정 안 함");
  await expect(card.getByText("하드웨어를 선택하지 않아 용량만 표시합니다. 적합 판정은 하지 않습니다.")).toBeVisible();
  await expect(card.getByRole("img", { name: /GPU 사용량/ })).toHaveCount(0);
  await expect(budgets.getByRole("columnheader", { name: "GPU 적합" })).toHaveCount(0);

  if (await isDevMock(page)) return; // the facts below are the real snapshot's, not the fixture's

  // The analysed snapshot, its coverage and the longest prompt (plan.md §21 "화면 단계").
  const evidence = await openTab(page, "적용 설정·근거");
  await expect(stat(evidence.getByRole("region", { name: "소스 revision" }), "데이터셋"), "dataset snapshot changed upstream: re-measure the EXAMPLE_* facts").toContainText(
    EXAMPLE_DATASET_REVISION,
  );
  await expect(stat(evidence.getByRole("region", { name: "프로필과 버전" }), "profile")).toHaveText("qwen3_5-hybrid");
  const data = await openTab(page, "데이터 길이");
  const scan = data.getByRole("region", { name: "스캔 범위" });
  await expect(scan.getByText("전체 완료")).toBeVisible();
  await expect(stat(scan, "읽은 row")).toHaveText(EXAMPLE_ROWS);
  await expect(stat(scan, "성공 row")).toHaveText(EXAMPLE_ROWS);
  await expect(stat(scan, "실패 row")).toHaveText("0");
  await expect(stat(scan, "미처리 row")).toHaveText("0");
  await expect(data).toContainText("GRPO: prompt만 사용합니다");
  await expect(data).toContainText("원문은 기본으로 표시하지 않습니다");
  const prompt = data.getByRole("region", { name: "prompt 길이" });
  await expect(stat(prompt, "row 수")).toHaveText(EXAMPLE_ROWS);
  await expect(stat(prompt, "최대")).toHaveText(new RegExp(`^${EXAMPLE_MAX_PROMPT}\\s*\\(train:\\d+\\)$`));
  await expect(data.getByRole("region", { name: /chosen|rejected/ })).toHaveCount(0);
  // The context check uses the longest prompt plus the smallest completion budget (1,024).
  const context = data.getByRole("region", { name: "context 검증" });
  await expect(context.getByText("상한 이내")).toBeVisible();
  await expect(stat(context, "prompt + 생성 예산")).toHaveText((Number(EXAMPLE_MAX_PROMPT) + 1_024).toLocaleString("ko-KR"));

  // A reload reconnects to the stored analysis through ?analysis=<id>.
  await page.reload();
  await expect(summary(page).getByRole("table", { name: /completion budget별/ })).toBeVisible({ timeout: 60_000 });
  await expect(statusAxis(page, "학습 준비")).toContainText("조건부");
});
