"use client";

import { Button } from "@/components/ui/primitives";

export interface RunBarProps {
  running: boolean;
  needsReanalysis: boolean;
  reanalysisReasons: string[];
  invalidCount: number;
  onExample: () => void;
}

/** The analyse button; heavy edits relabel it "데이터 재분석 필요" with the reasons (plan.md §4.2). */
export function RunBar({ running, needsReanalysis, reanalysisReasons, invalidCount, onExample }: RunBarProps) {
  const label = running ? "분석 진행 중…" : needsReanalysis ? "데이터 재분석 필요" : "전체 데이터 분석 및 계산";
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-surface p-4 sm:p-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <Button type="submit" variant="primary" size="lg" disabled={running} aria-describedby="run-help" className="w-full sm:w-auto">
          {label}
        </Button>
        <Button variant="ghost" onClick={onExample} disabled={running} className="w-full sm:w-auto">
          예시 입력 불러오기
        </Button>
      </div>
      {needsReanalysis && reanalysisReasons.length > 0 ? (
        <div role="status" className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <p className="font-medium">토큰화 결과를 다시 쓸 수 없는 변경이 있습니다.</p>
          <ul className="mt-1 list-disc pl-5">
            {reanalysisReasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {invalidCount > 0 ? (
        <p role="alert" className="text-[13px] text-err">
          입력값 {invalidCount}곳을 확인하세요.
        </p>
      ) : null}
      <p id="run-help" className="text-[12px] leading-snug text-muted">
        선택한 데이터 전체를 실제 tokenizer로 분석합니다. 모델 가중치를 내려받거나 GPU를 사용하지 않으며, 가벼운 설정 변경은 토큰화 결과를 재사용해 다시 계산합니다.
      </p>
    </div>
  );
}
