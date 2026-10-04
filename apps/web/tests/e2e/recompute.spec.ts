import { expect, test } from "@playwright/test";

import { ANALYSIS_TIMEOUT, isDevMock, loadExample, startAnalysis, summary, waitForCompletion } from "./helpers";

test("changing LoRA r recomputes from cached lengths without a new scan", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page, "SFT");
  test.skip(await isDevMock(page), "asserts real network requests");
  await startAnalysis(page);
  await waitForCompletion(page);

  const newAnalyses: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === "/api/v1/analyses") newAnalyses.push(request.url());
  });
  await page.getByRole("button", { name: /^Adapter/ }).click();
  const scenario = page.waitForRequest((r) => r.method() === "POST" && /\/api\/v1\/analyses\/[^/]+\/scenarios$/.test(new URL(r.url()).pathname));
  await page.getByRole("textbox", { name: "rank (r)", exact: true }).fill("64");
  const request = await scenario;
  expect(request.headers()["x-vramforge-request"]).toBe("1");
  const body = request.postDataJSON() as { request: { training: { lora: { r: number } } }; client_fingerprint: string };
  expect(body.request.training.lora.r).toBe(64);
  expect(body.client_fingerprint).toMatch(/^web-[0-9a-f]{16}$/);

  await expect(summary(page).getByText(/이전 설정 · 바뀐 조건으로 재계산 중/)).toBeHidden({ timeout: 120_000 });
  expect(newAnalyses).toHaveLength(0);
  await page.getByRole("tab", { name: "비교" }).click();
  await expect(page.getByRole("table", { name: "기준 분석 대비 변경별 차이" })).toContainText("LoRA r 16 → 64");
});
