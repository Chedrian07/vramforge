"use client";

import { useFormContext, useWatch } from "react-hook-form";

import { Field, NativeSelect, TextInput, describedBy } from "@/components/ui/primitives";
import { formatGiB } from "@/lib/format/bytes";
import type { FormValues } from "@/lib/form/values";
import { useBackendProfiles } from "@/lib/hooks/useInspection";

const CUSTOM = "__custom__";
const CAPACITY = "__capacity__";

export function HardwareSection() {
  const { register, setValue, formState } = useFormContext<FormValues>();
  const mode = useWatch<FormValues, "hardwareMode">({ name: "hardwareMode" });
  const presetId = useWatch<FormValues, "gpuPresetId">({ name: "gpuPresetId" });
  const profiles = useBackendProfiles();
  const presets = profiles.data?.hardware_presets ?? [];
  const errors = formState.errors;
  const selectValue = mode === "capacity_only" ? CAPACITY : mode === "custom" ? CUSTOM : presetId;
  const opts = { shouldDirty: true, shouldValidate: true };
  const preset = presets.find((p) => p.id === presetId) ?? null;

  return (
    <div className="flex flex-col gap-3">
      <Field
        label="Hardware"
        htmlFor="hardware-select"
        hint={
          mode === "capacity_only"
            ? "GPU를 고르지 않아도 필요한 용량을 계산합니다. 적합 판정은 GPU를 지정했을 때만 표시합니다."
            : "단일 NVIDIA GPU 기준 정적 추정입니다. 다중 GPU 분산 구성은 지원하지 않습니다."
        }
        error={errors.gpuPresetId?.message}
      >
        <NativeSelect
          id="hardware-select"
          value={selectValue}
          onChange={(event) => {
            const value = event.target.value;
            if (value === CAPACITY) {
              setValue("hardwareMode", "capacity_only", opts);
              setValue("gpuPresetId", "", opts);
              setValue("gpuPresetTotalBytes", "", opts);
            } else if (value === CUSTOM) {
              setValue("hardwareMode", "custom", opts);
              setValue("gpuPresetId", "", opts);
              setValue("gpuPresetTotalBytes", "", opts);
            } else {
              const selected = presets.find((p) => p.id === value);
              setValue("hardwareMode", "gpu_preset", opts);
              setValue("gpuPresetId", value, opts);
              setValue("gpuPresetTotalBytes", selected ? String(selected.total_bytes) : "", opts);
            }
          }}
          aria-describedby={describedBy("hardware-select", { hint: true, error: errors.gpuPresetId })}
        >
          <option value={CAPACITY}>용량만 계산</option>
          {presets.length > 0 ? (
            <optgroup label="GPU preset">
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {formatGiB(p.total_bytes)}
                </option>
              ))}
            </optgroup>
          ) : null}
          {mode === "gpu_preset" && presetId && !preset ? <option value={presetId}>{presetId} (목록에 없음)</option> : null}
          <option value={CUSTOM}>직접 입력 (GiB)</option>
        </NativeSelect>
      </Field>
      {profiles.isError ? <p className="text-[12px] text-warn">GPU preset 목록을 불러오지 못했습니다. 직접 입력은 사용할 수 있습니다.</p> : null}
      {preset?.note ? <p className="text-[12px] text-muted">{preset.note}</p> : null}

      {mode !== "capacity_only" ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {mode === "custom" ? (
            <Field label="GPU 전체 용량 (GiB)" htmlFor="hardware-total" error={errors.hardwareTotalGiB?.message}>
              <TextInput
                id="hardware-total"
                inputMode="decimal"
                className="num"
                placeholder="예: 24"
                {...register("hardwareTotalGiB")}
                aria-invalid={errors.hardwareTotalGiB ? true : undefined}
                aria-describedby={describedBy("hardware-total", { error: errors.hardwareTotalGiB })}
              />
            </Field>
          ) : null}
          <Field
            label="실제 사용 가능 VRAM (GiB, 선택)"
            htmlFor="hardware-usable"
            hint="다른 프로세스 점유를 이미 뺀 값이면 외부 점유를 0으로 두세요."
            error={errors.hardwareUsableGiB?.message}
          >
            <TextInput
              id="hardware-usable"
              inputMode="decimal"
              className="num"
              placeholder="비워 두면 전체 용량"
              {...register("hardwareUsableGiB")}
              aria-invalid={errors.hardwareUsableGiB ? true : undefined}
              aria-describedby={describedBy("hardware-usable", { hint: true, error: errors.hardwareUsableGiB })}
            />
          </Field>
        </div>
      ) : null}

      <details className="group rounded-lg border border-line px-3 py-2">
        <summary className="cursor-pointer select-none text-[13px] font-medium text-ink-2">외부 점유 · 계획용 여유 정책</summary>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Field label="외부 점유 (GiB)" htmlFor="external-reserved" hint="필요 총 용량 = 권장 용량 + 외부 점유" error={errors.externalReservedGiB?.message}>
            <TextInput id="external-reserved" inputMode="decimal" className="num" {...register("externalReservedGiB")} aria-invalid={errors.externalReservedGiB ? true : undefined} />
          </Field>
          <Field label="최소 여유 (GiB)" htmlFor="margin-min" hint="여유 = max(최소, 상한 × 비율)" error={errors.marginMinGiB?.message}>
            <TextInput id="margin-min" inputMode="decimal" className="num" {...register("marginMinGiB")} aria-invalid={errors.marginMinGiB ? true : undefined} />
          </Field>
          <Field label="여유 비율" htmlFor="margin-fraction" hint="0.15 = 상한의 15%" error={errors.marginFraction?.message}>
            <TextInput id="margin-fraction" inputMode="decimal" className="num" {...register("marginFraction")} aria-invalid={errors.marginFraction ? true : undefined} />
          </Field>
        </div>
        <p className="mt-2 text-[12px] text-muted">계획용 여유는 운영 정책이며 오차 보증이 아닙니다.</p>
      </details>
    </div>
  );
}
