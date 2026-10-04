"use client";

import { memo, type ReactNode } from "react";

import { Badge, Button, InfoTip } from "@/components/ui/primitives";
import { Bytes, BytesRange, IssueList, NotComputed, type DisplayIssue } from "@/components/ui/values";
import type { AnalysisResult } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { exactBytesRange, formatCount, formatGiB, gibNumber } from "@/lib/format/bytes";
import { JOB_STATUS_LABEL } from "@/lib/format/labels";
import type { RunState } from "@/lib/hooks/useAnalysisRun";
import type { RecomputeView } from "@/lib/hooks/useRecompute";
import { hardwareSelected, summarize, type ResultSummary } from "@/lib/result/summary";

import { BudgetList } from "./BudgetList";
import { StatusBadges } from "./StatusBadges";
import { UsageGauge } from "./UsageGauge";

const PARTIAL_LABEL: Record<string, string> = {
  rows_seen: "읽은 row",
  rows_ok: "성공 row",
  rows_failed: "실패 row",
  max_tokens: "현재까지 최대 길이",
  max_length: "현재까지 최대 길이",
  max_sequence_tokens: "현재까지 최대 시퀀스 길이",
  max_prompt_tokens: "현재까지 최대 prompt 길이",
  total_tokens: "누적 token",
  context_exceeded_rows: "context 초과 row",
};

export interface SummaryCardProps {
  run: RunState;
  view: Pick<RecomputeView, "display" | "stale" | "mode" | "reanalysisReasons" | "error" | "retry">;
  hardwareRequested: boolean;
  gpuWorkerConnected: boolean | null;
  exportSlot: ReactNode;
}

function staleNote(view: SummaryCardProps["view"]): { text: string; action?: ReactNode } | null {
  if (!view.stale || !view.display) return null;
  switch (view.mode) {
    case "pending":
    case "loading":
      return { text: "이전 설정 · 바뀐 조건으로 재계산 중…" };
    case "reanalysis":
      return { text: "이전 설정 · 데이터 재분석 필요" };
    case "invalid":
      return { text: "이전 설정 · 입력값을 확인하세요" };
    case "error":
      return {
        text: `이전 설정 · 재계산 실패: ${view.error?.issue.user_message ?? "알 수 없는 오류"}`,
        action: (
          <Button size="sm" onClick={view.retry}>
            다시 계산
          </Button>
        ),
      };
    default:
      return { text: "이전 설정" };
  }
}

function Headline({ summary, result }: { summary: ResultSummary; result: AnalysisResult }) {
  if (summary.kind === "budgets" && summary.budgetRange) {
    const r = summary.budgetRange;
    return (
      <HeadlineValue
        label={`completion budget별 예상 피크 (${formatCount(r.minBudget)}–${formatCount(r.maxBudget)} token)`}
        value={`${gibNumber(r.low)} – ${gibNumber(r.high)}`}
        exact={exactBytesRange(r.low, r.high)}
      />
    );
  }
  if (summary.peakLow != null && summary.peakHigh != null) {
    const same = gibNumber(summary.peakLow) === gibNumber(summary.peakHigh);
    return (
      <HeadlineValue
        label="예상 피크 범위"
        value={same ? gibNumber(summary.peakHigh) : `${gibNumber(summary.peakLow)} – ${gibNumber(summary.peakHigh)}`}
        exact={exactBytesRange(summary.peakLow, summary.peakHigh)}
      />
    );
  }
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] text-muted">예상 피크 범위</span>
      <span className="text-[32px] leading-tight font-semibold text-warn">산정 불가</span>
      <span className="text-[13px] leading-snug text-ink-2">{summary.peakNullReason ?? `${result.errors?.[0]?.user_message ?? "근거가 부족해 산정하지 않았습니다."}`}</span>
    </div>
  );
}

function HeadlineValue({ label, value, exact }: { label: string; value: string; exact: string | null }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] text-muted">{label}</span>
      <InfoTip label={<span className="num text-[44px] leading-none font-semibold tracking-tight text-ink">{value}</span>} className="no-underline">
        <span className="num">{exact}</span>
      </InfoTip>
      <span className="text-[13px] text-muted">GiB / GPU</span>
    </div>
  );
}

function Row({ label, children, hint }: { label: ReactNode; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] items-start gap-2 py-1.5 text-[14px]">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 text-ink">
        {children}
        {hint ? <span className="block text-[12px] text-muted">{hint}</span> : null}
      </dd>
    </div>
  );
}

function Placeholder() {
  return <span className="text-muted">—</span>;
}

function attentionItems(summary: ResultSummary): DisplayIssue[] {
  const items: DisplayIssue[] = [];
  for (const e of summary.excluded) {
    if (e.code === "GRPO_REWARD_UNSPECIFIED") continue; // shown as its own banner
    items.push({ code: e.code ?? "EXCLUDED", severity: "info", user_message: `제외: ${e.name} — ${e.reason}` });
  }
  for (const u of summary.unknown) {
    items.push({ code: "UNKNOWN_MEMORY_COMPONENT", severity: "warning", user_message: `산정 불가: ${u.name} — ${u.reason}` });
  }
  for (const issue of summary.issues) {
    if (issue.code === "GRPO_REWARD_UNSPECIFIED") continue;
    items.push(issue);
  }
  return items;
}

function SummaryCardImpl({ run, view, hardwareRequested, gpuWorkerConnected, exportSlot }: SummaryCardProps) {
  const result = view.display ?? run.status?.result ?? null;
  const running = run.phase === "creating" || run.phase === "running";
  const hasEstimate = result != null && (run.phase === "terminal" || view.display != null);
  const summary = hasEstimate && result ? summarize(result) : null;
  const stale = staleNote(view);
  const stageLabel = run.jobStatus ? JOB_STATUS_LABEL[run.jobStatus] : "요청 중";
  const jobPartial = run.jobStatus === "PARTIAL";

  return (
    <section aria-labelledby="summary-title" className="flex flex-col gap-4 rounded-xl border border-line bg-surface p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="summary-title" className="text-lg font-semibold tracking-tight text-ink">
          예상 메모리
        </h2>
        {gpuWorkerConnected ? null : (
          <Badge tone="neutral" title="GPU 실측 검증 worker가 연결되어 있지 않습니다. 결과는 정적 추정입니다.">
            GPU 검증 미연결
          </Badge>
        )}
      </div>

      {stale ? (
        <div id="summary-stale-note" role="status" className="flex flex-wrap items-center gap-2 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <span className="font-medium">{stale.text}</span>
          {stale.action}
        </div>
      ) : null}
      {jobPartial && summary ? (
        <p className="rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">부분 결과 — 전체 데이터 기준이 아닙니다.</p>
      ) : null}

      <div className={cn("flex flex-col gap-4", stale && "opacity-60")} aria-describedby={stale ? "summary-stale-note" : undefined}>
        {summary && result ? (
          <Headline summary={summary} result={result} />
        ) : (
          <div className="flex flex-col gap-1">
            <span className="text-[13px] text-muted">예상 피크 범위</span>
            <span className="num text-[44px] leading-none font-semibold tracking-tight text-muted">—</span>
            <span className="text-[13px] text-muted">GiB / GPU</span>
            <span className="text-[14px] font-medium text-ink-2">
              {running ? `분석 진행 중 · ${stageLabel}` : "전체 데이터 분석 필요"}
            </span>
          </div>
        )}

        {running ? <ScanSoFar run={run} /> : null}

        <dl className="divide-y divide-line/70 border-y border-line/70">
          <Row
            label="계획용 권장 용량"
            hint={
              summary?.recommendation
                ? `상한 + 여유 ${formatGiB(summary.recommendation.planning_margin_bytes)} (운영 여유 정책, 오차 보증 아님)`
                : undefined
            }
          >
            {!summary ? (
              <Placeholder />
            ) : summary.kind === "budgets" ? (
              <span className="text-ink-2">budget별 표 참고</span>
            ) : summary.recommendation ? (
              <Bytes value={summary.recommendation.recommended_application_capacity_bytes} className="font-semibold" />
            ) : (
              <NotComputed reason={summary.recommendationNullReason} compact={summary.kind === "unavailable" || summary.peakNullReason != null} />
            )}
          </Row>
          {summary?.recommendation && summary.recommendation.external_reserved_bytes > 0 ? (
            <Row label="필요 총 용량" hint="외부 점유 포함">
              <Bytes value={summary.recommendation.required_total_device_capacity_bytes} />
            </Row>
          ) : null}
          <Row label="학습 RAM" hint={summary && summary.hostRamHigh != null ? "학습 노드 host RAM (단계별 최대)" : undefined}>
            {!summary ? <Placeholder /> : summary.hostRamHigh != null ? <BytesRange low={summary.hostRamLow} high={summary.hostRamHigh} unit="gib" /> : <NotComputed reason={summary.hostRamNullReason} compact={summary.kind === "unavailable"} />}
          </Row>
          <Row label="확정 상주량" hint={summary?.floor != null ? "가중치·adapter·gradient·optimizer state" : undefined}>
            {!summary || summary.kind === "budgets" || summary.kind === "unavailable" ? <Placeholder /> : <Bytes value={summary.floor} />}
          </Row>
        </dl>

        {summary?.kind === "budgets" ? <BudgetList rows={summary.budgets} showFit={result ? hardwareSelected(result) : false} /> : null}
      </div>

      {summary?.rewardExcluded ? (
        <div role="note" className="flex items-start gap-2 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <span aria-hidden className="font-mono font-bold">!</span>
          <span>
            <strong className="font-semibold">reward footprint 미포함</strong> — reward가 정해지지 않아 policy·rollout만 계산한 조건부 결과입니다. 전체 학습 시스템의 확정 권장 용량으로 쓰지 마세요.
          </span>
        </div>
      ) : null}

      <StatusBadges axes={hasEstimate ? result?.status : run.status?.result?.status} />

      <GaugeArea result={hasEstimate ? result : null} summary={summary} hardwareRequested={hardwareRequested} stale={Boolean(stale)} />

      {summary ? (
        <div className="flex flex-col gap-2">
          <h3 className="text-[13px] font-semibold text-ink">확인할 점</h3>
          <IssueList issues={attentionItems(summary)} limit={4} empty="제외·미확정 항목과 경고가 없습니다." />
        </div>
      ) : null}
      {running && run.liveIssues.length > 0 ? <IssueList issues={run.liveIssues} limit={3} /> : null}

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-4">{exportSlot}</div>
    </section>
  );
}

function ScanSoFar({ run }: { run: RunState }) {
  const progress = run.progress;
  const entries = Object.entries(run.partial ?? {});
  if (!progress?.processed_rows && entries.length === 0) return null;
  return (
    <div className="rounded-lg border border-dashed border-line-strong px-3 py-2">
      <p className="text-[12px] font-medium text-muted">현재까지 확인한 데이터 기준 (최종 값 아님)</p>
      <dl className="mt-1 grid grid-cols-1 gap-x-4 gap-y-0.5 text-[13px] sm:grid-cols-2">
        {progress?.processed_rows != null ? (
          <div className="flex justify-between gap-2">
            <dt className="text-muted">처리한 row</dt>
            <dd className="num text-ink">
              {formatCount(progress.processed_rows)}
              {progress.total_rows != null ? ` / ${formatCount(progress.total_rows)}` : ""}
            </dd>
          </div>
        ) : null}
        {entries.map(([key, value]) => (
          <div key={key} className="flex justify-between gap-2">
            <dt className="text-muted">{PARTIAL_LABEL[key] ?? <span className="font-mono text-[12px]">{key}</span>}</dt>
            <dd className="num text-ink">{typeof value === "number" ? formatCount(value) : (value ?? "—")}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function GaugeArea({
  result,
  summary,
  hardwareRequested,
  stale,
}: {
  result: AnalysisResult | null;
  summary: ResultSummary | null;
  hardwareRequested: boolean;
  stale: boolean;
}) {
  if (!result || !summary) {
    return (
      <p className="text-[13px] text-muted">
        {hardwareRequested
          ? "분석이 끝나면 선택한 GPU 대비 사용량을 표시합니다."
          : "용량만 계산합니다. Hardware에서 GPU를 고르면 적합 여부를 판정합니다."}
      </p>
    );
  }
  if (!hardwareSelected(result)) {
    return <p className="text-[13px] text-muted">하드웨어를 선택하지 않아 용량만 표시합니다. 적합 판정은 하지 않습니다.</p>;
  }
  if (summary.kind === "budgets") {
    return <p className="text-[13px] text-muted">budget별 GPU 적합 여부는 위 표에 있습니다.</p>;
  }
  if (!summary.fit) return null;
  return <UsageGauge fit={summary.fit} stale={stale} />;
}

export const SummaryCard = memo(SummaryCardImpl);
