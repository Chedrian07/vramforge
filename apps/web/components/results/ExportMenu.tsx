"use client";

import { Popover } from "radix-ui";

import { Button } from "@/components/ui/primitives";
import type { ApiClient } from "@/lib/api/client";
import type { AnalysisResult, ExportFormat, JobStatus } from "@/lib/api/types";
import { READINESS_LABEL } from "@/lib/format/labels";

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

export function ExportMenu({
  api,
  analysisId,
  result,
  jobStatus,
  differsFromScreen = false,
}: {
  api: Pick<ApiClient, "exportUrl">;
  analysisId: string | null;
  result: AnalysisResult | null;
  jobStatus: JobStatus | null;
  /** The screen shows a recomputed scenario or changed inputs that the stored analysis lacks. */
  differsFromScreen?: boolean;
}) {
  const items = exportItems(analysisId ? result : null, jobStatus);
  return (
    <Popover.Root>
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
          <ul className="flex flex-col">
            {items.map((item) => {
              const descId = `export-${item.format}-desc`;
              const body = (
                <>
                  <span className="font-mono text-[13px] font-medium text-ink">{item.filename}</span>
                  <span id={descId} className="text-[12px] leading-snug text-muted">
                    {item.disabledReason ?? item.description}
                  </span>
                </>
              );
              return (
                <li key={item.format}>
                  {item.disabledReason || !analysisId ? (
                    <button
                      type="button"
                      aria-disabled="true"
                      aria-describedby={descId}
                      onClick={(event) => event.preventDefault()}
                      className="flex w-full cursor-not-allowed flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left opacity-60"
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
          {differsFromScreen && analysisId ? (
            <p role="note" className="mx-1 mt-1 rounded-lg bg-warn-soft px-2 py-1.5 text-[12px] leading-snug text-warn">
              내보내기 파일은 서버에 저장된 기준 분석(처음 분석한 설정과 수치)으로 만들어집니다. 화면의 재계산 결과나 바꾼 입력은 들어가지 않으니, 바뀐 설정으로 내보내려면 전체 분석을 다시 실행하세요.
            </p>
          ) : null}
          <p className="border-t border-line px-3 pt-2 pb-1 text-[11px] leading-snug text-muted">
            서버에 저장된 분석 결과로 생성합니다. 데이터 원문, 토큰, 절대 경로는 포함하지 않습니다.
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
