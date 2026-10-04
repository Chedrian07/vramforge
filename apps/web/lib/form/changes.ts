// Which edits need a new full scan and which only a recompute (docs/architecture.md §3.1,
// plan.md §4.2). The server stays authoritative (`requires_reanalysis`); this only avoids sending
// scenario requests for changes that obviously invalidate the tokenization.
import type { AnalysisRequest, ColumnMapping } from "@/lib/api/types";
import { formatSize } from "@/lib/format/bytes";
import { OBJECTIVE_LABEL } from "@/lib/format/labels";

import { canonicalJson } from "./fingerprint";

export interface LeafChange {
  path: string;
  label: string;
  from: unknown;
  to: unknown;
}

export interface ChangeSummary {
  /** Korean reasons why the stored token lengths no longer apply. */
  reanalysis: string[];
  /** Estimate/batch-level edits that the scenarios endpoint can recompute. */
  recompute: LeafChange[];
}

/** What the analysis actually used when the request left a value on "auto". */
export interface ResolvedSelection {
  split?: string | null;
  config?: string | null;
  mapping?: ColumnMapping | null;
}

type Flat = Map<string, unknown>;

function flatten(value: unknown, prefix: string, out: Flat): Flat {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) out.set(prefix, "{}");
    for (const [key, child] of entries) {
      if (child === undefined) continue;
      flatten(child, prefix ? `${prefix}.${key}` : key, out);
    }
    return out;
  }
  out.set(prefix, Array.isArray(value) ? canonicalJson(value) : value);
  return out;
}

function normalized(request: AnalysisRequest, resolved?: ResolvedSelection): Flat {
  const dataset = { ...request.dataset };
  if (resolved) {
    if (dataset.split == null && resolved.split != null) dataset.split = resolved.split;
    if (dataset.config == null && resolved.config != null) dataset.config = resolved.config;
    if (dataset.mapping == null && resolved.mapping != null) dataset.mapping = resolved.mapping;
  }
  return flatten({ ...request, dataset }, "", new Map());
}

function reanalysisReason(path: string, from: unknown, to: unknown): string | null {
  if (path === "training.objective") {
    const f = OBJECTIVE_LABEL[from as keyof typeof OBJECTIVE_LABEL] ?? String(from);
    const t = OBJECTIVE_LABEL[to as keyof typeof OBJECTIVE_LABEL] ?? String(to);
    return `학습 방식 변경 (${f} → ${t})`;
  }
  if (path === "model.reference" || path === "model.source_type") return "모델 변경";
  if (path === "model.revision") return "모델 revision 변경";
  if (path === "dataset.reference" || path === "dataset.source_type") return "데이터셋 변경";
  if (path === "dataset.revision") return "데이터셋 revision 변경";
  if (path === "dataset.config") return "데이터셋 config 변경";
  if (path === "dataset.split") return "학습 split 변경";
  if (path === "dataset.eval_split") return "평가 split 변경";
  if (path === "dataset.scan_mode" || path === "dataset.sample_rows") return "스캔 범위 변경";
  if (path === "dataset.mapping.empty_system_policy") return "빈 system 정책 변경";
  if (path === "dataset.mapping" || path.startsWith("dataset.mapping.")) return "컬럼 매핑 변경";
  if (path.startsWith("training.template")) return "템플릿 옵션 변경 (enable_thinking)";
  if (path === "training.packing") return "packing 변경";
  return null;
}

export const CHANGE_LABEL: Record<string, string> = {
  "model.loading_scope": "loading scope",
  "training.strategy": "전략",
  "training.quantization.enabled": "4-bit 로딩",
  "training.quantization.format": "4-bit 형식",
  "training.quantization.double_quant": "double quantization",
  "training.quantization.compute_dtype": "4-bit compute dtype",
  "training.lora.r": "LoRA r",
  "training.lora.alpha": "LoRA alpha",
  "training.lora.dropout": "LoRA dropout",
  "training.lora.target_modules": "LoRA 대상 모듈",
  "training.lora.exclude_modules": "LoRA 제외 모듈",
  "training.lora.modules_to_save": "modules_to_save",
  "training.lora.bias": "LoRA bias",
  "training.lora.rank_pattern": "rank_pattern",
  "training.lora.use_dora": "DoRA",
  "training.lora.use_rslora": "rsLoRA",
  "training.microbatch_per_device": "microbatch",
  "training.gradient_accumulation_steps": "gradient accumulation",
  "training.gradient_checkpointing": "gradient checkpointing",
  "training.precision": "precision",
  "training.optimizer": "optimizer",
  "training.load_dtype": "load dtype",
  "training.attention_backend": "attention backend",
  "training.linear_attention_kernel": "linear-attention kernel",
  "training.loss_kernel": "loss kernel",
  "training.pad_to_multiple_of": "pad_to_multiple_of",
  "training.backend_profile": "backend profile",
  "training.seed": "seed",
  "dpo.reference_strategy": "DPO reference 전략",
  "dpo.reference_model": "reference 모델",
  "dpo.beta": "DPO beta",
  "dpo.loss_type": "DPO loss type",
  "dpo.precompute_batch_size": "precompute batch",
  "dpo.sync_ref_model": "reference sync",
  "grpo.num_generations": "num_generations",
  "grpo.generation_batch_size": "generation_batch_size",
  "grpo.steps_per_generation": "steps_per_generation",
  "grpo.num_iterations": "num_iterations",
  "grpo.completion_budget": "completion budget",
  "grpo.completion_budget_candidates": "budget 후보",
  "grpo.beta": "GRPO beta",
  "grpo.reward.kind": "reward 종류",
  "grpo.reward.model_reference": "reward 모델",
  "grpo.reward.on_training_gpu": "reward 위치",
  "hardware.mode": "하드웨어",
  "hardware.gpu_preset": "GPU",
  "hardware.device_total_bytes": "GPU 전체 용량",
  "hardware.usable_bytes": "사용 가능 VRAM",
  "hardware.external_reserved_bytes": "외부 점유",
  "scope.include_evaluation": "평가 포함",
  "scope.include_checkpoint_save": "체크포인트 저장 포함",
  "margin_policy.min_bytes": "최소 여유",
  "margin_policy.fraction": "여유 비율",
};

const BYTE_PATHS = new Set([
  "hardware.device_total_bytes",
  "hardware.usable_bytes",
  "hardware.external_reserved_bytes",
  "margin_policy.min_bytes",
]);

export function formatChangeValue(path: string, value: unknown): string {
  if (value === null || value === undefined) return "자동";
  if (typeof value === "boolean") return value ? "켬" : "끔";
  if (typeof value === "number" && BYTE_PATHS.has(path)) return formatSize(value) ?? String(value);
  if (typeof value === "string" && value.startsWith("[")) {
    try {
      const parsed: unknown = JSON.parse(value);
      if (Array.isArray(parsed)) return parsed.length ? parsed.join(", ") : "없음";
    } catch {
      return value;
    }
  }
  return String(value);
}

export function describeChange(change: LeafChange): string {
  return `${change.label} ${formatChangeValue(change.path, change.from)} → ${formatChangeValue(change.path, change.to)}`;
}

export function classifyChanges(
  base: AnalysisRequest,
  next: AnalysisRequest,
  resolved?: ResolvedSelection,
): ChangeSummary {
  const a = normalized(base, resolved);
  const b = normalized(next, resolved);
  const reasons = new Set<string>();
  const recompute: LeafChange[] = [];
  const paths = new Set([...a.keys(), ...b.keys()]);
  for (const path of paths) {
    const from = a.get(path);
    const to = b.get(path);
    if (canonicalJson(from) === canonicalJson(to)) continue;
    const reason = reanalysisReason(path, from, to);
    if (reason) reasons.add(reason);
    else recompute.push({ path, label: CHANGE_LABEL[path] ?? path, from, to });
  }
  recompute.sort((x, y) => x.path.localeCompare(y.path));
  return { reanalysis: [...reasons], recompute };
}
