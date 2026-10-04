import { readFile } from "node:fs/promises";

import { expect, type Locator, type Page } from "@playwright/test";

export const MODEL_REF = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B";
/** A real full analysis downloads tokenizer and dataset files and scans every row. */
export const ANALYSIS_TIMEOUT = Number(process.env.E2E_ANALYSIS_TIMEOUT_MS ?? 15 * 60_000);

/**
 * Facts of the plan §21 dataset snapshot as the analyzer measured them on the compose stack
 * (full scan with the model's own tokenizer and chat template, TRL 1.14.1 preprocessing). They are
 * not plan numbers: if the dataset changes upstream, the revision check fails first and these
 * have to be measured again.
 */
export const EXAMPLE_DATASET_REVISION = "81aeacf06cf43b16d7278a3a01f019a496a53c51";
export const EXAMPLE_ROWS = "4,656";
/** GRPO: longest rendered prompt (generation prompt included). */
export const EXAMPLE_MAX_PROMPT = "268";
/** DPO: longest of prompt+chosen / prompt+rejected over all pairs. */
export const EXAMPLE_DPO_PAIR_MAX = "2,272";

export async function isDevMock(page: Page): Promise<boolean> {
  return page.getByText(/개발용 mock 데이터 모드/).isVisible();
}

/** Opens the calculator and waits until React has hydrated (keyboard handlers attached). */
export async function openApp(page: Page, path = "/") {
  await page.goto(path);
  await expect(page.locator('main[data-hydrated="true"]')).toBeAttached();
}

export async function loadExample(page: Page, objective?: "SFT" | "DPO" | "GRPO") {
  await openApp(page);
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

/** The analysis id the page follows (?analysis=<id>). */
export function analysisIdOf(page: Page): string | null {
  return new URL(page.url()).searchParams.get("analysis");
}

export function summary(page: Page): Locator {
  return page.getByRole("complementary", { name: "결과 요약" });
}

/** The badge of one result status axis in the summary card ("학습 준비", "GPU 적합" …). */
export function statusAxis(page: Page, label: string): Locator {
  return summary(page).getByRole("list", { name: "결과 상태" }).getByRole("listitem").filter({ hasText: label });
}

/** The value (<dd>) of a labelled statistic (<dt>) inside `scope`. */
export function stat(scope: Locator, label: string): Locator {
  const term = scope.page().getByRole("term").filter({ hasText: new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`) });
  return scope.locator("dl > div").filter({ has: term }).getByRole("definition");
}

export async function openTab(page: Page, name: string): Promise<Locator> {
  await page.getByRole("tab", { name }).click();
  const panel = page.getByRole("tabpanel");
  await expect(panel).toBeVisible();
  return panel;
}

/** The analysis fingerprint shown in the evidence tab (without its copy button's label). */
export async function shownFingerprint(page: Page): Promise<string> {
  const panel = await openTab(page, "적용 설정·근거");
  const value = await stat(panel.getByRole("region", { name: "프로필과 버전" }), "analysis fingerprint").innerText();
  const fingerprint = /req_[0-9a-f]+/.exec(value)?.[0];
  expect(fingerprint, `no fingerprint in "${value}"`).toBeTruthy();
  return fingerprint!;
}

export async function openExportMenu(page: Page): Promise<Locator> {
  await summary(page).getByRole("button", { name: "결과 내보내기" }).click();
  const menu = page.getByRole("dialog", { name: "결과 내보내기" });
  await expect(menu).toBeVisible();
  return menu;
}

/** The menu entry of one export file: a link for the stored analysis, a button for a scenario. */
export function exportItem(menu: Locator, filename: string): Locator {
  return menu.locator("a, button").filter({ hasText: filename });
}

/** Downloads one export file from the open menu and returns its text. */
export async function downloadExport(page: Page, menu: Locator, filename: string): Promise<string> {
  const downloadPromise = page.waitForEvent("download");
  await exportItem(menu, filename).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(filename);
  return readFile((await download.path())!, "utf8");
}

/** Exports never carry host paths or credentials (plan.md §12.4, §18). */
export function expectRedacted(text: string) {
  expect(text).not.toMatch(/\/Users\/|\/home\/[a-z]|\/data\/(?:artifacts|uploads|hf)|\/sources\/|hf_[A-Za-z0-9]{20,}/);
}

/** report.md escapes Markdown characters ("req\_…"); this undoes it for value checks. */
export function unescapeMarkdown(text: string): string {
  return text.replace(/\\([_*[\]()#|`\\])/g, "$1");
}

/** A top-level `key: value` scalar of an exported YAML document. */
export function yamlScalar(text: string, key: string): string | null {
  const match = new RegExp(`^${key}: ?'?([^'\\n]*)'?$`, "m").exec(text);
  return match ? match[1]!.trim() : null;
}

export async function expectNoHorizontalScroll(page: Page) {
  const { scrollWidth, innerWidth } = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  expect(scrollWidth).toBeLessThanOrEqual(innerWidth);
}
