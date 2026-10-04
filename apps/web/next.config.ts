import type { NextConfig } from "next";

// Dev mock mode (plan.md §20 M1: mock numbers exist only in development).
// `@vf/dev-mocks` resolves to the fixture-backed module only when the build is started with
// NEXT_PUBLIC_VF_DEV_MOCKS=1; every other build gets the empty stub, so fixture data can never
// reach a production bundle. `scripts/check-no-mocks.mjs` verifies the .next output after build.
const devMocksEnabled = process.env.NEXT_PUBLIC_VF_DEV_MOCKS === "1";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  reactStrictMode: true,
  // `next dev` would otherwise write AGENTS.md / CLAUDE.md into the project on every start.
  agentRules: false,
  turbopack: {
    resolveAlias: {
      "@vf/dev-mocks": devMocksEnabled ? "./lib/dev-mocks/index.ts" : "./lib/dev-mocks/disabled.ts",
    },
  },
  // Rewrites are frozen into server.js at build time, so production /api routing belongs to the
  // reverse proxy (Caddy). This rewrite only helps `next dev` without the proxy
  // (docs/research/stack-compat.md §1.5, W3).
  async rewrites() {
    const target = process.env.API_PROXY_TARGET;
    return target ? [{ source: "/api/:path*", destination: `${target}/api/:path*` }] : [];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;
