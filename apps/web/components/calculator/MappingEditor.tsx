"use client";

import { useFormContext, useWatch, type UseFormSetValue } from "react-hook-form";

import { Badge, Button, Field, Mono, NativeSelect } from "@/components/ui/primitives";
import type { ColumnMapping, DatasetFormat, DatasetInspection } from "@/lib/api/types";
import { DATASET_FORMAT_LABEL, OBJECTIVE_TRANSFORMATION } from "@/lib/format/labels";
import { FORMAT_ROLES, MAPPING_ROLES, ROLE_FIELD, ROLE_LABEL, type RoleField } from "@/lib/form/mapping";
import type { FormValues } from "@/lib/form/values";

const OPTS = { shouldDirty: true, shouldValidate: true } as const;

/**
 * Applies a server mapping (suggestion or candidate) to the form. The empty-system policy stays
 * the user's choice: inspection suggestions only echo the default policy.
 */
export function applyMapping(setValue: UseFormSetValue<FormValues>, mapping: ColumnMapping) {
  setValue("mappingEnabled", true, OPTS);
  setValue("mappingFormat", mapping.format, OPTS);
  for (const role of MAPPING_ROLES) setValue(ROLE_FIELD[role], mapping[role] ?? "", OPTS);
}

/**
 * Back to server auto-detection: no explicit roles left in the form, and the default empty-system
 * policy, because the policy travels with an explicit mapping only (ColumnMapping in
 * schemas/request.py; a null mapping is analysed with "omit").
 */
export function clearMapping(setValue: UseFormSetValue<FormValues>) {
  setValue("mappingEnabled", false, OPTS);
  setValue("mappingFormat", "auto", OPTS);
  for (const role of MAPPING_ROLES) setValue(ROLE_FIELD[role], "", OPTS);
  setValue("emptySystemPolicy", "omit", OPTS);
}

/** Column → role mapping (plan.md §7.2). Unmapped columns are metadata and never rendered. */
export function MappingEditor({ inspection }: { inspection: DatasetInspection | null }) {
  const { register, setValue, getValues, trigger, formState } = useFormContext<FormValues>();
  const values = useWatch<FormValues>();
  const enabled = values.mappingEnabled ?? false;
  const format = (values.mappingFormat ?? "auto") as DatasetFormat;
  const objective = values.objective ?? "sft";
  const columns = (inspection?.columns ?? []).map((c) => c.name);
  const roleFields = FORMAT_ROLES[format].map((role) => ROLE_FIELD[role]);
  const used = new Set(roleFields.map((f) => (values[f] as string | undefined) ?? "").filter(Boolean));
  const metadata = columns.filter((c) => !used.has(c));
  const suggested = inspection?.suggested_mapping ?? null;
  const candidates = inspection?.mapping_candidates ?? [];
  const errors = formState.errors;

  // A first manual edit turns auto-detection into an explicit mapping. It starts from the
  // unambiguous suggestion (never from a guess) so that one changed role does not drop the others.
  const startEditing = (changed: RoleField | "mappingFormat") => {
    if (getValues("mappingEnabled")) {
      // A new format has other required roles: show what is missing right away.
      if (changed === "mappingFormat") void trigger(Object.values(ROLE_FIELD));
      return;
    }
    if (suggested) {
      if (changed !== "mappingFormat") setValue("mappingFormat", suggested.format, OPTS);
      for (const role of MAPPING_ROLES) {
        if (ROLE_FIELD[role] !== changed) setValue(ROLE_FIELD[role], suggested[role] ?? "", OPTS);
      }
    }
    setValue("mappingEnabled", true, OPTS);
    void trigger(["mappingFormat", ...Object.values(ROLE_FIELD)]);
  };

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
          <Button size="sm" variant="ghost" onClick={() => clearMapping(setValue)}>
            자동 감지로 되돌리기
          </Button>
        ) : null}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="데이터 형식" htmlFor="mapping-format" error={enabled ? errors.mappingFormat?.message : undefined}>
          <NativeSelect
            id="mapping-format"
            aria-invalid={enabled && errors.mappingFormat ? true : undefined}
            aria-describedby={enabled && errors.mappingFormat ? "mapping-format-error" : undefined}
            {...register("mappingFormat", { onChange: () => startEditing("mappingFormat") })}
          >
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
          const role = MAPPING_ROLES.find((r) => ROLE_FIELD[r] === field)!;
          const error = enabled ? errors[field]?.message : undefined;
          const id = `mapping-${field}`;
          return (
            <Field key={field} label={ROLE_LABEL[role]} htmlFor={id} error={error}>
              <NativeSelect
                id={id}
                aria-invalid={error ? true : undefined}
                aria-describedby={error ? `${id}-error` : undefined}
                {...register(field, { onChange: () => startEditing(field) })}
              >
                <option value="">{enabled ? "— 사용 안 함 —" : "— 자동 감지 —"}</option>
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
      {enabled && metadata.length > 0 ? (
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
      {!enabled && columns.length > 0 ? (
        <p className="text-[12px] leading-snug text-muted">
          컬럼 {columns.length}개: <Mono>{columns.join(", ")}</Mono>. 역할은 분석 시 서버가 정하고, 역할이 없는 컬럼은 prompt에 넣지 않습니다.
        </p>
      ) : null}
    </fieldset>
  );
}
