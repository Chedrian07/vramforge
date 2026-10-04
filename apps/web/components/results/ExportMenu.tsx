"use client";

import { Popover } from "radix-ui";
import { useState } from "react";

import { Button } from "@/components/ui/primitives";
import { ApiError, type ApiClient } from "@/lib/api/client";
import type { AnalysisRequest, AnalysisResult, ExportFormat, JobStatus } from "@/lib/api/types";
import { saveFile } from "@/lib/download";
import { READINESS_LABEL } from "@/lib/format/labels";
import { formatDateTime } from "@/lib/format/time";

export interface ExportItem {
  format: ExportFormat;
  filename: string;
  description: string;
  disabledReason: string | null;
}

/** plan.md §12.4: which export is available and, when not, why. */
export function exportItems(result: AnalysisResult | null, jobStatus: JobStatus | null): ExportItem[] {
  const terminal = jobStatus === "COMPLETED" || jobStatus === "PARTIAL" || jobStatus === "FAILED" || jobStatus === "CANCELLED" || jobStatus === "NEEDS_INPUT";
  const base = !result || !terminal ? "분석이 끝난 뒤 내보낼 수 있습니다." : null;
  const readiness = result?.status?.training_readiness ?? null;
  const grpo = result?.resolved_config?.grpo ?? null;
  let trainerReason: string | null = base;
  if (!trainerReason) {
    if (jobStatus !== "COMPLETED") trainerReason = "완료된 분석에서만 실행용 설정을 만듭니다.";
    else if (readiness !== "ready") {
      trainerReason =
        readiness == null
          ? "학습 준비 상태가 판정되지 않아 실행용 설정을 제공하지 않습니다."
          : `학습 준비 상태가 '${READINESS_LABEL[readiness]}'라 실행용 설정을 제공하지 않습니다.${readiness === "conditional" ? " (예: GRPO reward 미지정)" : ""}`;
    } else if (grpo && (!grpo.budget_explicit || grpo.completion_budgets.length !== 1)) {
      // The trainer config needs one max_completion_length (exports/trainer_config.py check_ready).
      trainerReason = "GRPO는 completion budget을 하나로 지정해야 실행용 설정을 만듭니다 (예산별 시나리오는 계획용).";
    }
  }
  return [
    { format: "json", filename: "analysis.json", description: "원본 결과 · 가정 · 근거 (정수 bytes)", disabledReason: base },
    {
      format: "yaml",
      filename: "resolved-plan.yaml",
      description: "재현 가능한 계획 설정 (TRL 인자와 1:1이 아님)",
      disabledReason: base ?? (result?.resolved_config ? null : "적용 설정이 확정되지 않아 계획 YAML을 만들 수 없습니다."),
    },
    { format: "md", filename: "report.md", description: "사람이 읽는 보고서 · 데이터 통계 · 지원 상태", disabledReason: base },
    { format: "trainer-config", filename: "trainer-config.yaml", description: "실행용 Trainer 설정 (학습 준비 ready에서만)", disabledReason: trainerReason },
  ];
}

/**
 * What the export files are made from. The stored analysis is a plain GET download; a recomputed
 * scenario on screen is exported by POSTing the request it was computed for, so the files match
 * the numbers the user sees.
 */
export type ExportTarget =
  | {
      kind: "stored";
      result: AnalysisResult | null;
      jobStatus: JobStatus | null;
      /** The form has changes that the stored analysis (shown on screen) does not reflect. */
      formDiffers: boolean;
    }
  | {
      kind: "scenario";
      result: AnalysisResult;
      request: AnalysisRequest;
      changes: string[];
      /** The scenario is an earlier setting ("이전 설정"), not the current form. */
      stale: boolean;
    };

/** Which result the files come from, in words (always shown in the menu). */
export function exportSourceText(target: ExportTarget): string {
  if (target.kind === "scenario") {
    const changes = target.changes.length > 0 ? ` (${target.changes.join(", ")})` : "";
    const stale = target.stale ? " 현재 입력이 아니라 화면에 남아 있는 이전 설정의 결과입니다." : "";
    return `내보낼 결과: 화면에 표시된 재계산 시나리오${changes}. 저장된 분석에 이 설정을 적용해 서버가 다시 계산한 파일입니다.${stale}`;
  }
  const differs = target.formDiffers
    ? " 화면의 바뀐 입력은 들어가지 않으니, 바뀐 설정으로 내보내려면 재계산이 끝나기를 기다리거나 전체 분석을 다시 실행하세요."
    : "";
  return `내보낼 결과: 서버에 저장된 분석 (처음 분석한 설정과 수치).${differs}`;
}

/** Result retention (plan.md §16.4): when the stored result and its artifacts are deleted. */
export function RetentionNote({ expiresAt, className }: { expiresAt: string | null | undefined; className?: string }) {
  const when = formatDateTime(expiresAt);
  if (!when || !expiresAt) return null;
  return (
    <p className={className ?? "text-[12px] leading-snug text-muted"}>
      결과 보관 기한 <time dateTime={expiresAt}>{when}</time>까지 · 이후 결과와 길이 artifact가 자동 삭제됩니다.
    </p>
  );
}

export function ExportMenu({
  api,
  analysisId,
  target,
  expiresAt = null,
}: {
  api: Pick<ApiClient, "exportUrl" | "exportScenario">;
  analysisId: string | null;
  target: ExportTarget;
  expiresAt?: string | null;
}) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const items = exportItems(analysisId ? target.result : null, target.kind === "scenario" ? "COMPLETED" : target.jobStatus);
  const scenario = target.kind === "scenario" ? target : null;
  const exportable = analysisId != null && target.result != null;

  const downloadScenario = async (item: ExportItem) => {
    if (!scenario || !analysisId || busy) return;
    setBusy(item.format);
    setFailure(null);
    try {
      const file = await api.exportScenario(analysisId, { request: scenario.request, format: item.format });
      saveFile(file.blob, file.filename ?? item.filename);
    } catch (error) {
      setFailure(error instanceof ApiError ? error.issue.user_message : "시나리오 결과를 내보내지 못했습니다.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <Popover.Root onOpenChange={() => setFailure(null)}>
      <Popover.Trigger asChild>
        <Button className="w-full sm:w-auto" aria-haspopup="dialog">
          결과 내보내기
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          className="z-50 w-[min(22rem,calc(100vw-2rem))] rounded-xl border border-line bg-surface p-2 shadow-lg"
          aria-label="결과 내보내기"
        >
          {exportable ? (
            <p
              role="note"
              className={`mx-1 mb-1 rounded-lg px-2 py-1.5 text-[12px] leading-snug ${
                scenario?.stale || (target.kind === "stored" && target.formDiffers) ? "bg-warn-soft text-warn" : "bg-surface-2 text-ink-2"
              }`}
            >
              {exportSourceText(target)}
            </p>
          ) : null}
          <ul className="flex flex-col">
            {items.map((item) => {
              const descId = `export-${item.format}-desc`;
              const body = (
                <>
                  <span className="font-mono text-[13px] font-medium text-ink">{item.filename}</span>
                  <span id={descId} className="text-[12px] leading-snug text-muted">
                    {busy === item.format ? "서버에서 파일을 만드는 중…" : (item.disabledReason ?? item.description)}
                  </span>
                </>
              );
              const disabled = item.disabledReason != null || !exportable || !analysisId;
              return (
                <li key={item.format}>
                  {disabled ? (
                    <button
                      type="button"
                      aria-disabled="true"
                      aria-describedby={descId}
                      onClick={(event) => event.preventDefault()}
                      className="flex w-full cursor-not-allowed flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left opacity-60"
                    >
                      {body}
                    </button>
                  ) : scenario ? (
                    <button
                      type="button"
                      aria-describedby={descId}
                      aria-busy={busy === item.format ? true : undefined}
                      disabled={busy != null}
                      onClick={() => void downloadScenario(item)}
                      className="flex w-full flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left hover:bg-surface-2 disabled:cursor-wait"
                    >
                      {body}
                    </button>
                  ) : (
                    <a
                      href={api.exportUrl(analysisId, item.format)}
                      download={item.filename}
                      aria-describedby={descId}
                      className="flex w-full flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left hover:bg-surface-2"
                    >
                      {body}
                    </a>
                  )}
                </li>
              );
            })}
          </ul>
          {failure ? (
            <p role="alert" className="mx-1 mt-1 rounded-lg bg-err-soft px-2 py-1.5 text-[12px] leading-snug text-err">
              {failure}
            </p>
          ) : null}
          <div className="border-t border-line px-3 pt-2 pb-1 text-[11px] leading-snug text-muted">
            <p>데이터 원문, 토큰, 절대 경로는 포함하지 않습니다.</p>
            <RetentionNote expiresAt={expiresAt} className="mt-0.5" />
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
