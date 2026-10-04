"use client";

import { Dialog } from "radix-ui";

import { Button } from "@/components/ui/primitives";

export function HelpDialog() {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <Button size="sm" variant="secondary">
          도움말
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content className="fixed top-1/2 left-1/2 z-50 max-h-[85vh] w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-line bg-surface p-6 shadow-lg">
          <Dialog.Title className="text-lg font-semibold text-ink">계산기 사용법</Dialog.Title>
          <Dialog.Description className="mt-1 text-[13px] text-muted">
            모델·데이터셋·학습 방식을 넣으면 실제 tokenizer와 학습 전처리로 전체 데이터 길이를 분석하고, 데이터를 자르지 않는 조건의 GPU 피크 메모리를 산정합니다.
          </Dialog.Description>
          <div className="mt-4 flex flex-col gap-3 text-[14px] leading-relaxed text-ink-2">
            <section>
              <h3 className="font-semibold text-ink">숫자 읽는 법</h3>
              <ul className="mt-1 list-disc pl-5">
                <li><strong>예상 피크 범위</strong>: 한 시점에 동시에 살아 있는 메모리 합의 하한–상한입니다. 통계적 신뢰구간이 아닙니다.</li>
                <li><strong>계획용 권장 용량</strong>: 상한에 운영 여유(기본 max(2 GiB, 상한×15%))를 더한 값입니다. 오차 보증이 아닙니다.</li>
                <li><strong>확정 상주량</strong>: 가중치·adapter·gradient·optimizer state처럼 크기가 확정된 상주분입니다.</li>
                <li><strong>산정 불가</strong>: 근거가 없어 숫자를 만들지 않은 항목입니다. 0으로 채우지 않습니다.</li>
              </ul>
            </section>
            <section>
              <h3 className="font-semibold text-ink">상태 배지 5가지</h3>
              <p>스캔 범위, 데이터 보존, 학습 준비, 추정 근거, GPU 적합은 서로 독립입니다. 전체 스캔이 끝나도 reward가 없으면 학습 준비는 조건부일 수 있습니다.</p>
            </section>
            <section>
              <h3 className="font-semibold text-ink">재계산과 재분석</h3>
              <p>LoRA, batch, optimizer, kernel, 하드웨어처럼 가벼운 변경은 토큰화 결과를 재사용해 자동으로 다시 계산합니다. 학습 방식, 데이터·모델, 매핑, 템플릿 옵션을 바꾸면 &lsquo;데이터 재분석 필요&rsquo;가 표시됩니다.</p>
            </section>
            <section>
              <h3 className="font-semibold text-ink">GPU 검증 미연결</h3>
              <p>이 배포에는 GPU 실측 worker가 없습니다. 모든 결과는 CPU에서 계산한 정적 추정이며 실측으로 표시하지 않습니다.</p>
            </section>
            <section>
              <h3 className="font-semibold text-ink">개인정보</h3>
              <p>데이터 원문과 token id는 결과·내보내기에 넣지 않습니다. 로컬 경로는 브라우저 PC가 아니라 API 서버 기준이며 읽기 전용입니다.</p>
            </section>
            <section>
              <h3 className="font-semibold text-ink">키보드</h3>
              <p>Tab으로 이동하고, 선택 그룹과 탭은 방향키로, 접이식 설정은 Enter·Space로 엽니다.</p>
            </section>
          </div>
          <div className="mt-6 flex justify-end">
            <Dialog.Close asChild>
              <Button>닫기</Button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
