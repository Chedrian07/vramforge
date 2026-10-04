import { expect, test, type Locator } from "@playwright/test";

import { ANALYSIS_TIMEOUT, expectNoHorizontalScroll, loadExample, openApp, openTab, startAnalysis, summary, waitForCompletion } from "./helpers";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

/** A status pill on one line: never broken into one syllable per line by a narrow table cell. */
async function expectSingleLine(badges: Locator) {
  const count = await badges.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i++) {
    const box = (await badges.nth(i).boundingBox())!;
    expect(box.height, `badge ${i} wraps`).toBeLessThan(32);
  }
}

test("mobile layout keeps the page width and the input -> progress -> summary -> details order", async ({ page }) => {
  await openApp(page);
  await expectNoHorizontalScroll(page);
  await loadExample(page);
  await expectNoHorizontalScroll(page);

  const top = async (name: string) => (await page.getByRole("heading", { name, exact: true }).boundingBox())!.y;
  const inputs = (await page.getByRole("form", { name: "계산 입력" }).boundingBox())!.y;
  expect(inputs).toBeLessThan(await top("진행 상태"));
  expect(await top("진행 상태")).toBeLessThan(await top("예상 메모리"));
  expect(await top("예상 메모리")).toBeLessThan(await top("상세 결과"));

  await page.getByRole("button", { name: /^Adapter/ }).click();
  await page.getByRole("tab", { name: "적용 설정·근거" }).click();
  await expectNoHorizontalScroll(page);
});

test("mobile results: every tab fits the page width and tables stay readable", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page);
  await startAnalysis(page);
  await waitForCompletion(page);
  await expect(summary(page).getByRole("table", { name: /completion budget별/ })).toBeVisible();
  await expectNoHorizontalScroll(page);

  for (const name of ["메모리 구성", "데이터 길이", "단계별 피크", "비교", "적용 설정·근거"]) {
    await openTab(page, name);
    await expectNoHorizontalScroll(page);
  }

  // Wide tables scroll inside their frame; their cells are not squeezed to one character.
  const memory = await openTab(page, "메모리 구성");
  await expectSingleLine(memory.getByRole("table").first().getByText(/^(분석식|가정)$/));
  const phases = await openTab(page, "단계별 피크");
  await expectSingleLine(phases.getByRole("table").first().getByText("제외", { exact: true }));
  const timepoint = phases.getByRole("table").first().getByText(/^MODEL_LOAD_AND_QUANTIZE:/);
  expect((await timepoint.boundingBox())!.width).toBeGreaterThan(60);
  // A two-column table fits a phone without sideways scrolling.
  const data = await openTab(page, "데이터 길이");
  const longest = data.getByRole("table", { name: "가장 긴 row (ID와 길이만 표시)" }).first();
  const frame = await longest.evaluate((table) => ({ scroll: table.parentElement!.scrollWidth, client: table.parentElement!.clientWidth }));
  expect(frame.scroll).toBeLessThanOrEqual(frame.client);
});
