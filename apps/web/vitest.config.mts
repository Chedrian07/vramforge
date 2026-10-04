import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    tsconfigPaths: true, // Vite 8 built-in (docs/research/stack-compat.md §4.1)
    alias: {
      // Components get fakes injected explicitly in tests; the dev-mock module is imported
      // directly by its own tests.
      "@vf/dev-mocks": fileURLToPath(new URL("./lib/dev-mocks/disabled.ts", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
    testTimeout: 15_000,
  },
});
