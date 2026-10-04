import { defineConfig, devices } from "@playwright/test";

// End-to-end specs run against the real compose stack (proxy on :8080 by default).
// E2E_BASE_URL overrides the target; E2E_CHANNEL=chrome uses a locally installed Chrome instead of
// the bundled Chromium (the official mcr.microsoft.com/playwright:v1.63.0-noble image needs none).
const channel = process.env.E2E_CHANNEL;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  outputDir: "test-results",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:8080",
    locale: "ko-KR",
    trace: "retain-on-failure",
    acceptDownloads: true,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], ...(channel ? { channel } : {}) } }],
});
