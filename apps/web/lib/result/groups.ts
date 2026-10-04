// Allocation categories folded into seven display groups for the stacked peak bar: the categorical
// palette stays within its validated slots (dataviz series-count ladder) and each group keeps a
// fixed colour regardless of which groups are present (colour follows the entity).
import type { AllocationCategory, BreakdownItem } from "@/lib/api/types";

export interface MemoryGroup {
  key: string;
  label: string;
  color: string;
  categories: readonly AllocationCategory[];
}

export const MEMORY_GROUPS: readonly MemoryGroup[] = [
  { key: "base", label: "기본 가중치", color: "var(--vf-series-1)", categories: ["weights_base"] },
  { key: "train", label: "adapter·학습 상태", color: "var(--vf-series-2)", categories: ["weights_adapter", "gradients", "optimizer_states", "master_weights"] },
  { key: "activations", label: "activation", color: "var(--vf-series-3)", categories: ["saved_activations", "recompute_working_set"] },
  { key: "logits", label: "logits·loss", color: "var(--vf-series-4)", categories: ["logits_and_loss"] },
  { key: "generation", label: "생성 cache·state", color: "var(--vf-series-5)", categories: ["generation_cache", "recurrent_state", "rollout_buffers"] },
  { key: "other_models", label: "다른 모델 가중치", color: "var(--vf-series-6)", categories: ["weights_other_models"] },
  { key: "overhead", label: "오버헤드·여유", color: "var(--vf-series-7)", categories: ["load_transient", "workspace", "communication_buffers", "allocator_slack", "non_framework"] },
];

const GROUP_OF = new Map<AllocationCategory, MemoryGroup>(
  MEMORY_GROUPS.flatMap((g) => g.categories.map((c) => [c, g] as const)),
);

export function groupOf(category: AllocationCategory): MemoryGroup {
  return GROUP_OF.get(category) ?? MEMORY_GROUPS[MEMORY_GROUPS.length - 1]!;
}

export interface GroupTotal {
  group: MemoryGroup;
  high: number;
  low: number;
  items: number;
}

/** Sums of one timepoint's items per display group (known sizes only; unknown items are listed apart). */
export function groupTotals(items: readonly BreakdownItem[]): { totals: GroupTotal[]; unknown: BreakdownItem[] } {
  const totals = new Map<string, GroupTotal>();
  const unknown: BreakdownItem[] = [];
  for (const item of items) {
    if (item.bytes_high == null || item.bytes_low == null) {
      unknown.push(item);
      continue;
    }
    const group = groupOf(item.category);
    const current = totals.get(group.key) ?? { group, high: 0, low: 0, items: 0 };
    current.high += item.bytes_high;
    current.low += item.bytes_low;
    current.items += 1;
    totals.set(group.key, current);
  }
  return { totals: MEMORY_GROUPS.map((g) => totals.get(g.key)).filter((t): t is GroupTotal => t != null), unknown };
}
