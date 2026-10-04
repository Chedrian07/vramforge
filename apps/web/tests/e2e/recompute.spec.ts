import { expect, test } from "@playwright/test";

import type { AnalysisResult, ScenarioResponse } from "../../lib/api/types";

import {
  ANALYSIS_TIMEOUT,
  analysisIdOf,
  downloadExport,
  expectRedacted,
  isDevMock,
  loadExample,
  openExportMenu,
  openTab,
  shownFingerprint,
  startAnalysis,
  stat,
  summary,
  waitForCompletion,
  yamlScalar,
} from "./helpers";

// plan.md §4.2, §19.4 "재계산": changing LoRA r reuses the stored row lengths. No new analysis is
// created and nothing is downloaded or tokenized again; the scenario gets its own fingerprint.
test("changing LoRA r recomputes from the stored lengths: same analysis, new fingerprint", async ({ page }) => {
  test.setTimeout(ANALYSIS_TIMEOUT + 120_000);
  await loadExample(page, "DPO");
  test.skip(await isDevMock(page), "asserts real network requests");
  await startAnalysis(page);
  await waitForCompletion(page);
  const analysisId = analysisIdOf(page);
  const baseFingerprint = await shownFingerprint(page);
  const basePeak = await stat(summary(page), "계획용 권장 용량").innerText();
  const stored = (await (await page.request.get(`/api/v1/analyses/${analysisId}`)).json()) as { result: AnalysisResult };

  const newAnalyses: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === "/api/v1/analyses") newAnalyses.push(request.url());
  });
  await page.getByRole("button", { name: /^Adapter/ }).click();
  const scenario = page.waitForResponse((r) => r.request().method() === "POST" && /\/api\/v1\/analyses\/[^/]+\/scenarios$/.test(new URL(r.url()).pathname));
  await page.getByRole("textbox", { name: "rank (r)", exact: true }).fill("64");
  const response = await scenario;
  expect(response.status()).toBe(200);
  const request = response.request();
  expect(request.headers()["x-vramforge-request"]).toBe("1");
  const body = request.postDataJSON() as { request: { training: { lora: { r: number } } }; client_fingerprint: string };
  expect(body.request.training.lora.r).toBe(64);
  expect(body.client_fingerprint).toMatch(/^web-[0-9a-f]{16}$/);

  // Computed from the stored scan: same analysis, same row-length artifact and scan, new numbers.
  const answer = (await response.json()) as ScenarioResponse;
  expect(answer.requires_reanalysis).toBe(false);
  const recomputed = answer.result!;
  expect(recomputed.analysis_id).toBe(analysisId);
  expect(recomputed.dataset_scan?.preprocess_key).toBe(stored.result.dataset_scan?.preprocess_key);
  expect(recomputed.dataset_scan?.elapsed_seconds).toBe(stored.result.dataset_scan?.elapsed_seconds);
  expect(recomputed.resolved_config?.lora?.r).toBe(64);
  expect(answer.fingerprint).not.toBe(baseFingerprint);

  await expect(summary(page).getByText(/이전 설정 · 바뀐 조건으로 재계산 중/)).toBeHidden({ timeout: 120_000 });
  expect(newAnalyses).toHaveLength(0);
  expect(analysisIdOf(page)).toBe(analysisId);
  expect(await stat(summary(page), "계획용 권장 용량").innerText()).not.toBe(basePeak);
  const fingerprint = await shownFingerprint(page);
  expect(fingerprint).toBe(recomputed.analysis_fingerprint);
  expect(fingerprint).not.toBe(baseFingerprint);
  const compare = await openTab(page, "비교");
  await expect(compare.getByRole("table", { name: "기준 분석 대비 변경별 차이" })).toContainText("LoRA r 16 → 64");

  // Exports follow the screen: the recomputed scenario's request is posted (plan.md §12.4).
  const menu = await openExportMenu(page);
  await expect(menu.getByRole("note")).toContainText("재계산 시나리오 (LoRA r 16 → 64)");
  const exportRequest = page.waitForRequest((r) => r.method() === "POST" && new URL(r.url()).pathname.endsWith("/scenarios/export"));
  const json = await downloadExport(page, menu, "analysis.json");
  expect((await exportRequest).postDataJSON()).toMatchObject({ format: "json", request: { training: { lora: { r: 64 } } } });
  const exported = JSON.parse(json) as AnalysisResult;
  expect(exported.analysis_fingerprint).toBe(fingerprint);
  expect(exported.resolved_config?.lora?.r).toBe(64);
  expect(exported.memory?.scenarios[0]?.devices[0]?.scenario_high_bytes).toBe(recomputed.memory?.scenarios[0]?.devices[0]?.scenario_high_bytes);
  const trainer = await downloadExport(page, menu, "trainer-config.yaml");
  expectRedacted(trainer);
  expect(yamlScalar(trainer, "analysis_fingerprint")).toBe(fingerprint);
  expect(trainer).toMatch(/^peft:\n {2}r: 64$/m);
  expect(newAnalyses).toHaveLength(0);
});
