"use client";

import { Tabs } from "radix-ui";
import { memo, useState } from "react";

import { Field, NativeSelect } from "@/components/ui/primitives";
import type { AnalysisResult } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import type { ScenarioHistoryEntry } from "@/lib/hooks/useRecompute";
import { scenarioById } from "@/lib/result/summary";

import { CompareTab } from "./tabs/CompareTab";
import { DataTab } from "./tabs/DataTab";
import { EvidenceTab } from "./tabs/EvidenceTab";
import { MemoryTab } from "./tabs/MemoryTab";
import { PhaseTab } from "./tabs/PhaseTab";

export const DETAIL_TABS = [
  { value: "memory", label: "메모리 구성" },
  { value: "data", label: "데이터 길이" },
  { value: "phases", label: "단계별 피크" },
  { value: "compare", label: "비교" },
  { value: "evidence", label: "적용 설정·근거" },
] as const;

function Placeholder() {
  return <p className="py-6 text-[13px] text-muted">전체 데이터 분석을 마치면 이 탭에 결과를 표시합니다.</p>;
}

function DetailTabsImpl({
  result,
  base,
  history,
  partial,
  stale,
}: {
  result: AnalysisResult | null;
  base: AnalysisResult | null;
  history: ScenarioHistoryEntry[];
  partial: boolean;
  stale: boolean;
}) {
  const [tab, setTab] = useState<string>("memory");
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const scenarios = result?.memory?.scenarios ?? [];
  const scenario = scenarioById(result, scenarioId);

  return (
    <section aria-labelledby="details-title" className="rounded-xl border border-line bg-surface p-4 sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 id="details-title" className="text-lg font-semibold tracking-tight text-ink">
          상세 결과
        </h2>
        {scenarios.length > 1 ? (
          <Field label="시나리오" htmlFor="scenario-select" className="w-full sm:w-72">
            <NativeSelect id="scenario-select" value={scenario?.scenario_id ?? ""} onChange={(e) => setScenarioId(e.target.value)}>
              {scenarios.map((s) => (
                <option key={s.scenario_id} value={s.scenario_id}>
                  {s.label}
                </option>
              ))}
            </NativeSelect>
          </Field>
        ) : null}
      </div>
      {stale && result ? <p className="mt-2 text-[12px] text-warn">이전 설정의 결과입니다. 재계산이 끝나면 갱신됩니다.</p> : null}
      <Tabs.Root value={tab} onValueChange={setTab} className="mt-4">
        <Tabs.List aria-label="상세 결과" className="flex flex-wrap gap-1 border-b border-line">
          {DETAIL_TABS.map((t) => (
            <Tabs.Trigger
              key={t.value}
              value={t.value}
              className={cn(
                "-mb-px rounded-t-lg border-b-2 border-transparent px-3 py-2 text-[14px] font-medium text-muted hover:text-ink",
                "data-[state=active]:border-accent data-[state=active]:text-ink",
              )}
            >
              {t.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        <div className={cn("pt-4", stale && "opacity-70")}>
          <Tabs.Content value="memory">{result ? <MemoryTab scenario={scenario} /> : <Placeholder />}</Tabs.Content>
          <Tabs.Content value="data">{result ? <DataTab result={result} partial={partial} /> : <Placeholder />}</Tabs.Content>
          <Tabs.Content value="phases">{result ? <PhaseTab scenario={scenario} /> : <Placeholder />}</Tabs.Content>
          <Tabs.Content value="compare">{result ? <CompareTab result={result} base={base} history={history} /> : <Placeholder />}</Tabs.Content>
          <Tabs.Content value="evidence">{result ? <EvidenceTab result={result} scenario={scenario} /> : <Placeholder />}</Tabs.Content>
        </div>
      </Tabs.Root>
    </section>
  );
}

export const DetailTabs = memo(DetailTabsImpl);
