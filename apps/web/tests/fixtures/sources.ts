// Inspection, upload, backend-profile and local-root fixtures (example model/dataset facts from
// docs/research/example-model-dataset.md; GPU presets are fixture-only entries).
import type { Schemas } from "@/lib/api/types";

import { RTX_24GB } from "./analysis-dpo-sft";
import { GIB, fixtureId } from "./builders";
import { datasetManifest, issue, modelManifest, modelSummary, preferenceMapping, tokenizerManifest } from "./common";

export const modelInspection: Schemas["ModelInspection"] = {
  manifest: modelManifest,
  summary: modelSummary,
  tokenizer: tokenizerManifest,
  architecture_adapter: "qwen3_5_hybrid",
  support: [
    { objective: "sft", strategy: "qlora", grade: "analytic", readiness: "ready", note: "" },
    { objective: "dpo", strategy: "qlora", grade: "analytic", readiness: "ready", note: "" },
    { objective: "grpo", strategy: "qlora", grade: "analytic", readiness: "conditional", note: "reward 지정 필요" },
    { objective: "sft", strategy: "full", grade: null, readiness: "unsupported", note: "검증된 profile 없음" },
  ],
  issues: [
    issue("REQUESTED_OPTION_NOT_EFFECTIVE", "info", "TRL 문자열 진입은 vision tower까지 로딩하므로 텍스트 데이터라도 frozen 가중치로 상주합니다.", { affected_component: "vision" }),
  ],
};

export const datasetInspection: Schemas["DatasetInspection"] = {
  manifest: datasetManifest,
  configs: ["default"],
  selected_config: "default",
  splits: [{ name: "train", num_rows: 4_656, num_bytes: 6_306_335 }],
  selected_split: "train",
  split_auto_selected: true,
  columns: [
    { name: "lang", dtype: "string", kind: "string" },
    { name: "vulnerability", dtype: "string", kind: "string" },
    { name: "system", dtype: "string", kind: "string" },
    { name: "question", dtype: "string", kind: "string" },
    { name: "chosen", dtype: "string", kind: "string" },
    { name: "rejected", dtype: "string", kind: "string" },
  ],
  detected_format: "preference",
  mapping_candidates: [preferenceMapping],
  suggested_mapping: preferenceMapping,
  mapping_ambiguous: false,
  issues: [],
};

export const ambiguousDatasetInspection: Schemas["DatasetInspection"] = {
  ...datasetInspection,
  configs: ["default", "extended"],
  selected_config: null,
  splits: [
    { name: "train", num_rows: 4_656, num_bytes: null },
    { name: "train_extra", num_rows: null, num_bytes: null },
  ],
  selected_split: null,
  split_auto_selected: false,
  columns: [...(datasetInspection.columns ?? []), { name: "instruction", dtype: "string", kind: "string" }],
  mapping_candidates: [preferenceMapping, { ...preferenceMapping, prompt: "instruction" }],
  suggested_mapping: null,
  mapping_ambiguous: true,
  issues: [issue("COLUMN_MAPPING_REQUIRED", "warning", "prompt 역할 후보가 두 개(question, instruction)입니다. 매핑을 선택하세요.")],
};

export const missingModelInspection: Schemas["ModelInspection"] = {
  manifest: null,
  summary: null,
  tokenizer: null,
  architecture_adapter: null,
  support: [],
  issues: [issue("SOURCE_NOT_FOUND", "error", "모델을 찾을 수 없습니다. ID와 접근 권한을 확인하세요.")],
};

export const uploadResponse: Schemas["UploadResponse"] = {
  upload_id: fixtureId("upload-01"),
  reference: `upload:${fixtureId("upload-01")}`,
  filename: "pairs.jsonl",
  size_bytes: 48_213,
  sha256: "3f5c8a0e9b1d4c7a2e6f0b9d8c7a6e5f4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a",
  expires_at: "2026-10-05T12:00:00Z",
};

export const backendProfiles: Schemas["BackendProfilesResponse"] = {
  estimator_version: "0.1.0",
  profiles: [
    {
      profile_id: "qwen3_5_hybrid.trl_1_14_1.analytic",
      profile_version: "2026.10.0",
      architecture_adapter: "qwen3_5_hybrid",
      description: "Qwen3.5 hybrid (linear 24 / full 8) · TRL 1.14.1",
      model_types: ["qwen3_5"],
      environment_id: "cuda-trl-1.14.1",
      support: modelInspection.support,
    },
  ],
  environments: [
    {
      environment_id: "cuda-trl-1.14.1",
      description: "CUDA 학습 환경 (torch 2.14.1, transformers 5.18.0, trl 1.14.1, peft 0.21.2, bitsandbytes 0.50.2)",
      packages: { torch: "2.14.1", transformers: "5.18.0", trl: "1.14.1", peft: "0.21.2", bitsandbytes: "0.50.2", accelerate: "1.15.0" },
      dependency_lock_digest: `sha256:${fixtureId("lock-digest")}`,
    },
  ],
  hardware_presets: [
    RTX_24GB,
    { id: "vf-fixture-gpu-48gb", name: "48 GiB GPU (fixture)", total_bytes: 48 * GIB, note: "" },
    { id: "vf-fixture-gpu-80gb", name: "80 GiB GPU (fixture)", total_bytes: 80 * GIB, note: "" },
  ],
  gpu_worker_connected: false,
  evidence_levels: ["metadata_only", "analytic", "calibrated", "measured"],
};

export const localRoots: Schemas["LocalRootsResponse"] = {
  roots: [
    { name: "models", reference_prefix: "local:models/", description: "읽기 전용 모델 폴더", read_only: true },
    { name: "datasets", reference_prefix: "local:datasets/", description: "읽기 전용 데이터 폴더", read_only: true },
  ],
};
