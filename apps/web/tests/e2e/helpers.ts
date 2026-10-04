import { expect, type Locator, type Page } from "@playwright/test";

export const MODEL_REF = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B";
/** A real full analysis downloads tokenizer and dataset files and scans every row. */
export const ANALYSIS_TIMEOUT = Number(process.env.E2E_ANALYSIS_TIMEOUT_MS ?? 15 * 60_000);

export async function isDevMock(page: Page): Promise<boolean> {
  return page.getByText(/개발용 mock 데이터 모드/).isVisible();
}

export async function loadExample(page: Page, objective?: "SFT" | "DPO" | "GRPO") {
  await page.goto("/");
  await page.getByRole("button", { name: "예시 입력 불러오기" }).click();
  await expect(page.getByRole("textbox", { name: "Model", exact: true })).toHaveValue(MODEL_REF);
  if (objective) await method(page, objective).click();
}

export function method(page: Page, name: "SFT" | "DPO" | "GRPO"): Locator {
  return page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name });
}

export async function startAnalysis(page: Page) {
  await page.getByRole("button", { name: /^(전체 데이터 분석 및 계산|데이터 재분석 필요)$/ }).click();
  await expect(page).toHaveURL(/[?&]analysis=/);
}

export async function waitForCompletion(page: Page) {
  await expect(page.getByText("분석을 마쳤습니다. 결과는 요약 카드와 아래 탭에 있습니다.")).toBeVisible({ timeout: ANALYSIS_TIMEOUT });
}

export function summary(page: Page): Locator {
  return page.getByRole("complementary", { name: "결과 요약" });
}

export async function expectNoHorizontalScroll(page: Page) {
  const { scrollWidth, innerWidth } = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  expect(scrollWidth).toBeLessThanOrEqual(innerWidth);
}
