"use client";

import { useFormContext, useWatch, type UseFormSetValue } from "react-hook-form";

import { Badge, Button, Field, Mono, NativeSelect } from "@/components/ui/primitives";
import type { ColumnMapping, DatasetFormat, DatasetInspection } from "@/lib/api/types";
import { DATASET_FORMAT_LABEL, OBJECTIVE_TRANSFORMATION } from "@/lib/format/labels";
import type { FormValues } from "@/lib/form/values";

type RoleField = "mapSystem" | "mapPrompt" | "mapChosen" | "mapRejected" | "mapCompletion" | "mapMessages" | "mapText";

const ROLES: Record<RoleField, { label: string; key: keyof ColumnMapping }> = {
  mapSystem: { label: "system message", key: "system" },
  mapPrompt: { label: "user prompt", key: "prompt" },
  mapChosen: { label: "preferred response (chosen)", key: "chosen" },
  mapRejected: { label: "dispreferred response (rejected)", key: "rejected" },
  mapCompletion: { label: "completion", key: "completion" },
  mapMessages: { label: "messages (대화 목록)", key: "messages" },
  mapText: { label: "text", key: "text" },
};

const ROLES_BY_FORMAT: Record<DatasetFormat, RoleField[]> = {
  auto: ["mapSystem", "mapPrompt", "mapChosen", "mapRejected", "mapCompletion", "mapMessages", "mapText"],
  preference: ["mapSystem", "mapPrompt", "mapChosen", "mapRejected"],
  prompt_completion: ["mapSystem", "mapPrompt", "mapCompletion"],
  prompt_only: ["mapSystem", "mapPrompt"],
  messages: ["mapMessages"],
  text: ["mapText"],
};

/** Applies a server mapping (suggestion or candidate) to the form. */
export function applyMapping(setValue: UseFormSetValue<FormValues>, mapping: ColumnMapping) {
  const opts = { shouldDirty: true, shouldValidate: true };
  setValue("mappingEnabled", true, opts);
  setValue("mappingFormat", mapping.format, opts);
  setValue("mapSystem", mapping.system ?? "", opts);
  setValue("mapPrompt", mapping.prompt ?? "", opts);
  setValue("mapChosen", mapping.chosen ?? "", opts);
  setValue("mapRejected", mapping.rejected ?? "", opts);
  setValue("mapCompletion", mapping.completion ?? "", opts);
  setValue("mapMessages", mapping.messages ?? "", opts);
  setValue("mapText", mapping.text ?? "", opts);
  setValue("emptySystemPolicy", mapping.empty_system_policy, opts);
}

/** Column → role mapping (plan.md §7.2). Unmapped columns are metadata and never rendered. */
export function MappingEditor({ inspection }: { inspection: DatasetInspection | null }) {
  const { register, setValue } = useFormContext<FormValues>();
  const values = useWatch<FormValues>();
  const enabled = values.mappingEnabled ?? false;
  const format = (values.mappingFormat ?? "auto") as DatasetFormat;
  const objective = values.objective ?? "sft";
  const columns = (inspection?.columns ?? []).map((c) => c.name);
  const roleFields = ROLES_BY_FORMAT[format];
  const used = new Set(roleFields.map((f) => (values[f] as string | undefined) ?? "").filter(Boolean));
  const metadata = columns.filter((c) => !used.has(c));
  const suggested = inspection?.suggested_mapping ?? null;
  const candidates = inspection?.mapping_candidates ?? [];

  const markEdited = { onChange: () => setValue("mappingEnabled", true, { shouldDirty: true }) };

  return (
    <fieldset className="flex min-w-0 flex-col gap-3 rounded-lg border border-line p-3">
      <legend className="px-1 text-[13px] font-semibold text-ink">컬럼 매핑</legend>
      <p className="text-[12px] leading-snug text-muted">{OBJECTIVE_TRANSFORMATION[objective]}</p>
      {inspection?.mapping_ambiguous ? (
        <Badge tone="warn">매핑 후보가 여러 개입니다. 역할을 직접 선택해야 분석할 수 있습니다.</Badge>
      ) : null}
      {!enabled ? (
        <p className="text-[12px] text-ink-2">
          현재 매핑: <strong>자동 감지</strong> (분석 시 서버가 적용한 매핑을 결과에 표시합니다)
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        {suggested ? (
          <Button size="sm" onClick={() => applyMapping(setValue, suggested)}>
            제안 매핑 적용
          </Button>
        ) : null}
        {candidates.length > 1
          ? candidates.map((candidate, index) => (
              <Button key={index} size="sm" variant="ghost" onClick={() => applyMapping(setValue, candidate)}>
                후보 {index + 1}: prompt = {candidate.prompt ?? candidate.messages ?? candidate.text ?? "—"}
              </Button>
            ))
          : null}
        {enabled ? (
          <Button size="sm" variant="ghost" onClick={() => setValue("mappingEnabled", false, { shouldDirty: true })}>
            자동 감지로 되돌리기
          </Button>
        ) : null}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="데이터 형식" htmlFor="mapping-format">
          <NativeSelect id="mapping-format" {...register("mappingFormat", markEdited)}>
            {(Object.keys(DATASET_FORMAT_LABEL) as DatasetFormat[]).map((f) => (
              <option key={f} value={f}>
                {DATASET_FORMAT_LABEL[f]}
              </option>
            ))}
          </NativeSelect>
        </Field>
        {roleFields.map((field) => {
          const current = (values[field] as string | undefined) ?? "";
          const options = current && !columns.includes(current) ? [...columns, current] : columns;
          return (
            <Field key={field} label={ROLES[field].label} htmlFor={`mapping-${field}`}>
              <NativeSelect id={`mapping-${field}`} {...register(field, markEdited)}>
                <option value="">— 사용 안 함 —</option>
                {options.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </NativeSelect>
            </Field>
          );
        })}
      </div>
      {metadata.length > 0 ? (
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-muted">메타데이터 컬럼</p>
          <ul className="flex flex-wrap gap-1.5" aria-label="메타데이터 컬럼">
            {metadata.map((column) => (
              <li key={column} className="inline-flex max-w-full items-center gap-1 rounded-md border border-line bg-surface px-2 py-0.5 text-[12px]">
                <Mono>{column}</Mono>
                <span className="text-muted">· prompt에 삽입하지 않음</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </fieldset>
  );
}
