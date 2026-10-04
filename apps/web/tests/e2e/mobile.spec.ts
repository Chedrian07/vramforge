import { expect, test } from "@playwright/test";

import { expectNoHorizontalScroll, loadExample } from "./helpers";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

test("mobile layout keeps the page width and the input -> progress -> summary -> details order", async ({ page }) => {
  await page.goto("/");
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
