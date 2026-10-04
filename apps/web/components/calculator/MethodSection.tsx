"use client";

import { useFormContext, useWatch } from "react-hook-form";

import { Segmented, SwitchField } from "@/components/ui/controls";
import { InfoTip } from "@/components/ui/primitives";
import type { Objective, Strategy } from "@/lib/api/types";
import { OBJECTIVE_TRANSFORMATION, STRATEGY_LABEL } from "@/lib/format/labels";
import type { FormValues } from "@/lib/form/values";

const OBJECTIVES: Array<{ value: Objective; label: string }> = [
  { value: "sft", label: "SFT" },
  { value: "dpo", label: "DPO" },
  { value: "grpo", label: "GRPO" },
];

const STRATEGIES: Array<{ value: Strategy; label: string; description: string }> = [
  { value: "full", label: "Full", description: "전체 파라미터 학습 (4-bit 로딩과 함께 쓰지 않음)" },
  { value: "lora", label: "LoRA", description: "원본 dtype 가중치 + LoRA adapter" },
  { value: "qlora", label: "QLoRA", description: "4-bit NF4 가중치 + LoRA adapter" },
];

export function MethodSection() {
  const { setValue, formState } = useFormContext<FormValues>();
  const objective = useWatch<FormValues, "objective">({ name: "objective" });
  const strategy = useWatch<FormValues, "strategy">({ name: "strategy" });
  const load4bit = useWatch<FormValues, "load4bit">({ name: "load4bit" });
  const opts = { shouldDirty: true, shouldValidate: true };

  // plan.md §5.1: the 4-bit switch and the strategy move together; Full turns 4-bit off.
  const setLoad4bit = (on: boolean) => {
    setValue("load4bit", on, opts);
    if (on) setValue("strategy", "qlora", opts);
    else if (strategy === "qlora") setValue("strategy", "lora", opts);
  };
  const setStrategy = (next: Strategy) => {
    setValue("strategy", next, opts);
    setValue("load4bit", next === "qlora", opts);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <span id="method-label" className="text-[13px] font-medium text-ink-2">
          Method
        </span>
        <Segmented labelId="method-label" label="Method" className="self-start" value={objective} onValueChange={(v) => setValue("objective", v, opts)} options={OBJECTIVES} />
        <p className="text-[12px] leading-snug text-muted">{OBJECTIVE_TRANSFORMATION[objective]}</p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
        <SwitchField
          id="load-in-4bit"
          label="Load in 4-bit"
          checked={load4bit}
          onCheckedChange={setLoad4bit}
          description={`전략: ${STRATEGY_LABEL[strategy]} · 켜면 QLoRA, 끄면 LoRA를 사용합니다.`}
        />
        <div className="flex flex-col gap-1">
          <span id="strategy-label" className="text-[13px] font-medium text-ink-2">
            전략
          </span>
          <Segmented labelId="strategy-label" label="전략" size="sm" value={strategy} onValueChange={setStrategy} options={STRATEGIES} />
        </div>
      </div>
      {formState.errors.load4bit?.message ? (
        <p role="alert" className="text-[12px] text-err">
          {formState.errors.load4bit.message}
        </p>
      ) : null}

      <div className="flex items-start gap-3 rounded-lg border border-line bg-surface-2 px-3 py-2">
        <span aria-hidden className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-md bg-accent text-[12px] font-bold text-accent-ink">
          ✓
        </span>
        <div className="min-w-0 text-[13px]">
          <p className="font-medium text-ink" aria-readonly="true">
            전체 데이터 분석 · Truncation 없음
          </p>
          <p className="text-[12px] leading-snug text-muted">
            선택한 split의 모든 row를 실제 tokenizer와 chat template으로 끝까지 읽습니다. 길이 기준 절단·삭제·분할은 하지 않고,{" "}
            <InfoTip label="위반은 결과에 표시">
              context 초과, 실패 row, 템플릿 content 손실은 숨기지 않고 결과의 데이터 보존 상태와 경고로 보고합니다.
            </InfoTip>
            합니다.
          </p>
        </div>
      </div>
    </div>
  );
}
