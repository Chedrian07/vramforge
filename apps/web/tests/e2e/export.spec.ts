import { expect, test, type Page } from "@playwright/test";

import type { AnalysisResult } from "../../lib/api/types";
import { formatCount, formatGiB, formatGiBRange } from "../../lib/format/bytes";

import {
  ANALYSIS_TIMEOUT,
  EXAMPLE_DATASET_REVISION,
  EXAMPLE_ROWS,
  analysisIdOf,
  downloadExport,
  expectRedacted,
  exportItem,
  isDevMock,
  loadExample,
  method,
  openExportMenu,
  openTab,
  shownFingerprint,
  startAnalysis,
  stat,
  summary,
  unescapeMarkdown,
  waitForCompletion,
  yamlScalar,
} from "./helpers";

interface Exports {
  json: AnalysisResult;
  yaml: string;
  md: string;
}

/** analysis.json, resolved-plan.yaml and report.md of the stored analysis on screen. */
async function storedExports(page: Page): Promise<Exports> {
  const menu = await openExportMenu(page);
  await expect(menu.getByRole("note")).toContainText("서버에 저장된 분석");
  const raw = await downloadExport(page, menu, "analysis.json");
  const yaml = await downloadExport(page, menu, "resolved-plan.yaml");
  const md = unescapeMarkdown(await downloadExport(page, menu, "report.md"));
  for (const text of [raw, yaml, md]) expectRedacted(text);
  await page.keyboard.press("Escape");
  return { json: JSON.parse(raw) as AnalysisResult, yaml, md };
}

/** The ids, fingerprint and every scenario's integer bytes agree across the three files. */
function expectConsistent({ json, yaml, md }: Exports, analysisId: string | null, fingerprint: string) {
  expect(json.analysis_id).toBe(analysisId);
  expect(json.analysis_fingerprint).toBe(fingerprint);
  expect(yamlScalar(yaml, "analysis_id")).toBe(analysisId);
  expect(yamlScalar(yaml, "analysis_fingerprint")).toBe(fingerprint);
  expect(md).toContain(fingerprint);
  for (const scenario of json.memory?.scenarios ?? []) {
    const device = scenario.devices[0]!;
    for (const bytes of [device.scenario_low_bytes, device.scenario_high_bytes, scenario.recommendation?.recommended_application_capacity_bytes]) {
      expect(bytes).toEqual(expect.any(Number));
      expect(yaml).toContain(`: ${bytes}\n`);
      expect(md).toContain(formatCount(bytes)!);
    }
  }
}

// plan.md §19.4 "결과 export": byte numbers, fingerprints, settings and warnings of every export
// match what the page shows; the runnable trainer config exists only for a ready result.
test("exports match the screen; trainer-config only for the ready DPO result", async ({ page }) => {
  test.setTimeout(2 * ANALYSIS_TIMEOUT + 180_000);
  await loadExample(page);
  test.skip(await isDevMock(page), "the dev mock mode has no export endpoint");
  await startAnalysis(page);
  await waitForCompletion(page);

  // GRPO without a reward is conditional: no runnable trainer config (plan.md §12.4).
  let menu = await openExportMenu(page);
  const gated = exportItem(menu, "trainer-config.yaml");
  await expect(gated).toHaveAttribute("aria-disabled", "true");
  await expect(gated).toContainText("조건부");
  await page.keyboard.press("Escape");

  const grpoId = analysisIdOf(page);
  const grpoFingerprint = await shownFingerprint(page);
  const grpo = await storedExports(page);
  expectConsistent(grpo, grpoId, grpoFingerprint);
  expect(grpo.json.requested_config.training.objective).toBe("grpo");
  expect(grpo.json.requested_config.training.strategy).toBe("qlora");
  expect(grpo.json.source_manifests?.dataset?.resolved_revision).toBe(EXAMPLE_DATASET_REVISION);
  expect(grpo.md).toContain(`읽음 ${EXAMPLE_ROWS} / 성공 ${EXAMPLE_ROWS} / 실패 0`);

  // Every budget row on screen is the exported scenario's range and capacity.
  const budgets = summary(page).getByRole("table", { name: /completion budget별/ });
  const scenarios = grpo.json.memory?.scenarios ?? [];
  expect(scenarios).toHaveLength(4);
  for (const scenario of scenarios) {
    const device = scenario.devices[0]!;
    const row = budgets.getByRole("row").filter({ has: page.getByRole("rowheader", { name: formatCount(scenario.params?.completion_budget as number)!, exact: true }) });
    await expect(row.getByRole("cell").nth(0)).toHaveText(formatGiBRange(device.scenario_low_bytes, device.scenario_high_bytes)!);
    await expect(row.getByRole("cell").nth(1)).toHaveText(formatGiB(scenario.recommendation?.recommended_application_capacity_bytes)!);
  }
  // Warnings as shown in the evidence tab.
  const evidence = await openTab(page, "적용 설정·근거");
  expect(grpo.json.warnings?.map((w) => w.code)).toContain("GRPO_REWARD_UNSPECIFIED");
  for (const warning of grpo.json.warnings ?? []) {
    await expect(evidence.getByRole("region", { name: "경고와 오류" })).toContainText(warning.user_message);
  }

  // DPO: a ready result, so all four files, the trainer config included.
  await method(page, "DPO").click();
  await startAnalysis(page);
  await expect(page).not.toHaveURL(new RegExp(`analysis=${grpoId}`));
  await waitForCompletion(page);
  const dpoId = analysisIdOf(page);
  const dpoFingerprint = await shownFingerprint(page);
  const dpo = await storedExports(page);
  expectConsistent(dpo, dpoId, dpoFingerprint);
  expect(dpo.json.status?.training_readiness).toBe("ready");
  const recommended = dpo.json.memory?.scenarios[0]?.recommendation?.recommended_application_capacity_bytes;
  await expect(stat(summary(page), "계획용 권장 용량")).toContainText(formatGiB(recommended)!);
  expect(dpo.md).toContain(`${formatGiB(recommended)} (${formatCount(recommended)} bytes)`);

  menu = await openExportMenu(page);
  await expect(exportItem(menu, "trainer-config.yaml")).not.toHaveAttribute("aria-disabled", "true");
  const trainer = await downloadExport(page, menu, "trainer-config.yaml");
  expectRedacted(trainer);
  expect(yamlScalar(trainer, "analysis_id")).toBe(dpoId);
  expect(yamlScalar(trainer, "analysis_fingerprint")).toBe(dpoFingerprint);
  expect(trainer).toMatch(/^ {2}class: DPOTrainer$/m);
  expect(trainer).toMatch(/^peft:\n {2}r: 16$/m);
  expect(trainer).toMatch(/^ {4}max_length: null$/m);
  expect(trainer).toContain(`revision: ${EXAMPLE_DATASET_REVISION}`);
});
