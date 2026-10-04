#!/usr/bin/env node
// Fails when development fixture data reached the Next.js build output (plan.md §20.2:
// "개발 mock 결과가 production build에 표시되지 않는다"). Every fixture under tests/fixtures
// carries ids starting with the marker below, so any bundled fixture value is detectable.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const MARKERS = ["vf-fixture", "VF_DEV_FIXTURE"];
const TEXT_EXT = /\.(?:js|mjs|cjs|html|json|rsc|txt|css|map|body|meta)$/;

const root = fileURLToPath(new URL("..", import.meta.url));
const outDir = join(root, ".next");

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const rel = relative(outDir, path);
    if (rel === "cache" || rel.startsWith(`cache/`) || name === "node_modules") continue;
    const st = statSync(path);
    if (st.isDirectory()) yield* walk(path);
    else if (TEXT_EXT.test(name)) yield path;
  }
}

let exists = true;
try {
  statSync(outDir);
} catch {
  exists = false;
}
if (!exists) {
  console.error("check-no-mocks: .next not found; run `next build` first");
  process.exit(2);
}

const hits = [];
let scanned = 0;
for (const file of walk(outDir)) {
  scanned += 1;
  const text = readFileSync(file, "utf8");
  for (const marker of MARKERS) {
    if (text.includes(marker)) hits.push(`${relative(root, file)} (${marker})`);
  }
}

if (hits.length > 0) {
  console.error("check-no-mocks: development fixture markers found in the build output:");
  for (const hit of hits) console.error(`  - ${hit}`);
  console.error("Rebuild without NEXT_PUBLIC_VF_DEV_MOCKS=1.");
  process.exit(1);
}
console.log(`check-no-mocks: ok (${scanned} files scanned, no fixture markers)`);
