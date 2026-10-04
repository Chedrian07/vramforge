"use client";

import { useState } from "react";
import { useFormContext } from "react-hook-form";

import { Badge, Button, Mono, NativeSelect } from "@/components/ui/primitives";
import { IssueList } from "@/components/ui/values";
import type { JobStatus, NeedsInput } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { formatCount } from "@/lib/format/bytes";
import { JOB_STATUS_LABEL, PROGRESS_STEPS } from "@/lib/format/labels";
import type { FormValues } from "@/lib/form/values";
import type { RunState } from "@/lib/hooks/useAnalysisRun";
import { jobTone } from "@/lib/result/status";

import { applyMapping } from "./MappingEditor";

type StepState = "done" | "current" | "pending" | "stopped";

const STEP_TEXT: Record<StepState, string> = { done: "완료", current: "진행 중", pending: "대기", stopped: "중단" };

function stepIndexOf(status: JobStatus | null | undefined): number {
  if (!status) return -1;
  return PROGRESS_STEPS.findIndex((step) => step.statuses.includes(status));
}

export function stepStates(run: Pick<RunState, "jobStatus" | "progress" | "phase">): StepState[] {
  const n = PROGRESS_STEPS.length;
  const status = run.jobStatus;
  if (!status || run.phase === "idle") return Array<StepState>(n).fill("pending");
  if (status === "COMPLETED") return Array<StepState>(n).fill("done");
  const running = stepIndexOf(status);
  if (running >= 0) return PROGRESS_STEPS.map((_, i) => (i < running ? "done" : i === running ? "current" : "pending"));
  if (status === "QUEUED") return Array<StepState>(n).fill("pending");
  // CANCEL_REQUESTED, NEEDS_INPUT, FAILED, PARTIAL, CANCELLED: where did it stop?
  let at = stepIndexOf(run.progress?.stage);
  if (status === "NEEDS_INPUT" && at < 0) at = 0;
  if (at < 0) return Array<StepState>(n).fill("pending");
  const atState: StepState = status === "CANCEL_REQUESTED" ? "current" : "stopped";
  return PROGRESS_STEPS.map((_, i) => (i < at ? "done" : i === at ? atState : "pending"));
}

const STEP_STYLE: Record<StepState, string> = {
  done: "border-accent bg-accent text-accent-ink",
  current: "border-accent bg-accent-soft text-accent",
  pending: "border-line-strong bg-surface text-muted",
  stopped: "border-warn bg-warn-soft text-warn",
};

const STEP_GLYPH: Record<StepState, string> = { done: "✓", current: "●", pending: "", stopped: "!" };

export function ProgressPanel({
  run,
  onCancel,
  onRerun,
  onRetry,
}: {
  run: RunState;
  onCancel: () => void;
  onRerun: () => void;
  /** Reload the job status after a failed status/stream request. */
  onRetry?: () => void;
}) {
  const states = stepStates(run);
  const progress = run.progress;
  const active = run.phase === "creating" || run.phase === "running";
  const status = run.jobStatus;
  const result = run.status?.result ?? null;
  const terminalIssue = run.status?.error ?? result?.errors?.[0] ?? null;

  return (
    <section aria-labelledby="progress-title" className="rounded-xl border border-line bg-surface p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="progress-title" className="text-lg font-semibold tracking-tight text-ink">
          진행 상태
        </h2>
        <div className="flex items-center gap-2">
          {status ? <Badge tone={jobTone(status)}>{JOB_STATUS_LABEL[status]}</Badge> : <Badge tone="neutral">시작 전</Badge>}
          {active && run.analysisId ? (
            <Button size="sm" variant="danger" onClick={onCancel} disabled={run.cancelling || status === "CANCEL_REQUESTED"}>
              {run.cancelling || status === "CANCEL_REQUESTED" ? "취소 요청됨" : "분석 취소"}
            </Button>
          ) : null}
        </div>
      </div>

      <ol className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4" aria-label="분석 단계">
        {PROGRESS_STEPS.map((step, i) => {
          const state = states[i] ?? "pending";
          return (
            <li key={step.key} aria-current={state === "current" ? "step" : undefined} className="flex min-w-0 items-center gap-2">
              <span aria-hidden className={cn("inline-flex size-6 shrink-0 items-center justify-center rounded-full border text-[12px] font-bold", STEP_STYLE[state])}>
                {STEP_GLYPH[state] || i + 1}
              </span>
              <span className="flex min-w-0 flex-col leading-tight">
                <span className={cn("text-[13px] font-medium", state === "pending" ? "text-muted" : "text-ink")}>{step.label}</span>
                <span className="text-[11px] text-muted">{STEP_TEXT[state]}</span>
              </span>
            </li>
          );
        })}
      </ol>

      {run.phase === "idle" ? (
        <p className="mt-3 text-[13px] text-muted">분석을 시작하면 구조 확인 → 데이터 토큰화 → 배치 분석 → 메모리 산정 순서로 진행 상태를 표시합니다.</p>
      ) : null}

      {progress && (progress.processed_rows != null || progress.shard_progress) ? <RowsProgress progress={progress} active={active} /> : null}
      {progress?.message ? <p className="mt-2 text-[13px] text-ink-2">{progress.message}</p> : null}

      <div aria-live="polite" className="mt-2 flex flex-col gap-2">
        {active && run.connection === "reconnecting" ? (
          <p role="status" className="text-[13px] text-warn">
            진행 상태 연결이 끊겨 다시 연결하는 중입니다. 분석은 서버에서 계속됩니다.
          </p>
        ) : null}
        {run.error ? <IssueList issues={[run.error.issue]} /> : null}
        {run.phase === "error" && run.analysisId && onRetry && run.error?.status !== 404 ? (
          <div>
            <Button size="sm" onClick={onRetry}>
              상태 다시 불러오기
            </Button>
          </div>
        ) : null}
        {status === "COMPLETED" ? <p className="text-[13px] text-ok">분석을 마쳤습니다. 결과는 요약 카드와 아래 탭에 있습니다.</p> : null}
        {status === "PARTIAL" ? (
          <>
            <p className="text-[13px] text-warn">일부 단계만 끝난 부분 결과입니다. 확인한 범위까지만 표시합니다.</p>
            {terminalIssue ? <IssueList issues={[terminalIssue]} /> : null}
          </>
        ) : null}
        {status === "FAILED" && terminalIssue ? <IssueList issues={[terminalIssue]} /> : null}
        {status === "CANCELLED" ? (
          <p className="text-[13px] text-warn">
            분석을 취소했습니다.
            {progress?.processed_rows != null ? ` 취소 시점까지 ${formatCount(progress.processed_rows)} row를 확인했습니다.` : ""}
          </p>
        ) : null}
        {status === "NEEDS_INPUT" && result?.needs_input ? <NeedsInputPanel needsInput={result.needs_input} onRerun={onRerun} /> : null}
      </div>
    </section>
  );
}

function RowsProgress({ progress, active }: { progress: NonNullable<RunState["progress"]>; active: boolean }) {
  const processed = progress.processed_rows ?? null;
  const total = progress.total_rows ?? null;
  const shards = progress.shard_progress ?? null;
  const known = processed != null && total != null && total > 0;
  const percent = known ? Math.min(100, Math.floor((processed / total) * 1000) / 10) : null;
  const text = processed == null ? null : known ? `${formatCount(processed)} / ${formatCount(total)} row` : `${formatCount(processed)} row 처리`;
  return (
    <div className="mt-4 flex flex-col gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
        <span className="num text-ink">
          {text}
          {percent != null ? ` · ${percent}%` : ""}
        </span>
        {shards ? (
          <span className="num text-muted">
            shard {formatCount(shards.completed)} / {shards.total != null ? formatCount(shards.total) : "?"}
          </span>
        ) : null}
      </div>
      {known ? (
        <div
          role="progressbar"
          aria-label="처리한 row"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={processed}
          aria-valuetext={text ?? undefined}
          className="h-2 w-full overflow-hidden rounded-full bg-neutral-soft"
        >
          <div className="h-full rounded-full bg-accent" style={{ width: `${percent}%` }} />
        </div>
      ) : (
        <div
          role="progressbar"
          aria-label="처리한 row"
          aria-valuetext={`${text ?? "진행 중"} (전체 row 수 미확인)`}
          className={cn("h-2 w-full overflow-hidden rounded-full bg-neutral-soft", active && "motion-safe:animate-pulse")}
        >
          <div className="h-full w-full bg-[repeating-linear-gradient(135deg,var(--vf-accent-soft)_0_8px,transparent_8px_16px)]" />
        </div>
      )}
      {!known && processed != null ? (
        <p className="text-[12px] text-muted">전체 row 수를 아직 알 수 없어 비율(%)은 표시하지 않습니다.</p>
      ) : null}
    </div>
  );
}

/** NEEDS_INPUT: pick config / split / mapping inline, then run again (plan.md §4.1). */
function NeedsInputPanel({ needsInput, onRerun }: { needsInput: NeedsInput; onRerun: () => void }) {
  const { setValue } = useFormContext<FormValues>();
  const [choices, setChoices] = useState<Record<string, string>>(() =>
    Object.fromEntries(needsInput.choices.map((c) => [c.field, c.suggested ?? ""])),
  );
  const candidates = needsInput.mapping_candidates ?? [];
  const ready = needsInput.choices.every((c) => (choices[c.field] ?? "") !== "");

  const apply = () => {
    const opts = { shouldDirty: true, shouldValidate: true };
    for (const choice of needsInput.choices) {
      const value = choices[choice.field] ?? "";
      if (choice.field === "dataset.config") setValue("datasetConfig", value, opts);
      else if (choice.field === "dataset.split") setValue("datasetSplit", value, opts);
      else if (choice.field === "dataset.mapping") {
        const candidate = candidates[Number(value)];
        if (candidate) applyMapping(setValue, candidate);
      }
    }
    onRerun();
  };

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-warn/40 bg-warn-soft/40 p-3">
      <p className="text-[13px] font-medium text-ink">결과를 바꾸는 선택이 필요합니다. 임의로 추측하지 않고 분석을 멈췄습니다.</p>
      {needsInput.choices.map((choice) => {
        const id = `needs-${choice.field.replace(/\W/g, "-")}`;
        return (
          <div key={choice.field} className="flex flex-col gap-1">
            <label htmlFor={id} className="text-[13px] font-medium text-ink-2">
              <Mono>{choice.field}</Mono>
            </label>
            <p className="text-[12px] text-muted">{choice.reason}</p>
            {choice.field === "dataset.mapping" ? (
              <NativeSelect id={id} value={choices[choice.field] ?? ""} onChange={(e) => setChoices((prev) => ({ ...prev, [choice.field]: e.target.value }))}>
                <option value="">— 매핑 후보 선택 —</option>
                {candidates.map((c, i) => (
                  <option key={i} value={String(i)}>
                    {[c.system && `system=${c.system}`, c.prompt && `prompt=${c.prompt}`, c.chosen && `chosen=${c.chosen}`, c.rejected && `rejected=${c.rejected}`, c.completion && `completion=${c.completion}`, c.messages && `messages=${c.messages}`, c.text && `text=${c.text}`]
                      .filter(Boolean)
                      .join(", ")}
                  </option>
                ))}
              </NativeSelect>
            ) : (
              <NativeSelect id={id} value={choices[choice.field] ?? ""} onChange={(e) => setChoices((prev) => ({ ...prev, [choice.field]: e.target.value }))}>
                <option value="">— 선택 —</option>
                {(choice.options ?? []).map((option) => (
                  <option key={option} value={option}>
                    {option}
                    {option === choice.suggested ? " (제안)" : ""}
                  </option>
                ))}
              </NativeSelect>
            )}
          </div>
        );
      })}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={apply} disabled={!ready}>
          선택 적용 후 다시 분석
        </Button>
        {!ready ? <span className="text-[12px] text-muted">모든 항목을 선택하세요.</span> : null}
      </div>
    </div>
  );
}
