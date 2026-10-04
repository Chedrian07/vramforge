// Column-mapping rules shared by the editor, the request builder and form validation. They mirror
// packages/estimator/src/vramforge_estimator/inspection/dataset_mapping.py (FORMAT_ROLES,
// REQUIRED_ROLES): an explicit mapping is used as is by the analysis, so it must be complete for
// its format and must not carry roles of another format.
import type { DatasetFormat } from "@/lib/api/types";

import type { FormValues } from "./values";

export type MappingRole = "system" | "prompt" | "chosen" | "rejected" | "completion" | "messages" | "text";
export type RoleField = "mapSystem" | "mapPrompt" | "mapChosen" | "mapRejected" | "mapCompletion" | "mapMessages" | "mapText";
export type ConcreteFormat = Exclude<DatasetFormat, "auto">;

export const MAPPING_ROLES: readonly MappingRole[] = ["system", "prompt", "chosen", "rejected", "completion", "messages", "text"];

export const ROLE_FIELD: Record<MappingRole, RoleField> = {
  system: "mapSystem",
  prompt: "mapPrompt",
  chosen: "mapChosen",
  rejected: "mapRejected",
  completion: "mapCompletion",
  messages: "mapMessages",
  text: "mapText",
};

export const ROLE_LABEL: Record<MappingRole, string> = {
  system: "system message",
  prompt: "user prompt",
  chosen: "preferred response (chosen)",
  rejected: "dispreferred response (rejected)",
  completion: "completion",
  messages: "messages (대화 목록)",
  text: "text",
};

/** Roles each format uses; `auto` (auto-detection) lists every role. */
export const FORMAT_ROLES: Record<DatasetFormat, readonly MappingRole[]> = {
  auto: MAPPING_ROLES,
  preference: ["system", "prompt", "chosen", "rejected"],
  prompt_completion: ["system", "prompt", "completion"],
  prompt_only: ["system", "prompt"],
  messages: ["messages"],
  text: ["text"],
};

/** Roles a format cannot do without. */
export const REQUIRED_ROLES: Record<ConcreteFormat, readonly MappingRole[]> = {
  preference: ["chosen", "rejected"],
  prompt_completion: ["prompt", "completion"],
  prompt_only: ["prompt"],
  messages: ["messages"],
  text: ["text"],
};

type MappingValues = Pick<FormValues, "mappingFormat" | RoleField>;

/** The columns an explicit mapping uses, by role (only the roles of its format). */
export function mappedColumns(values: MappingValues): Partial<Record<MappingRole, string>> {
  const out: Partial<Record<MappingRole, string>> = {};
  for (const role of FORMAT_ROLES[values.mappingFormat]) {
    const column = values[ROLE_FIELD[role]].trim();
    if (column) out[role] = column;
  }
  return out;
}

/** Validation messages of an explicit mapping, keyed by form field (empty when complete). */
export function explicitMappingErrors(values: MappingValues): Partial<Record<"mappingFormat" | RoleField, string>> {
  if (values.mappingFormat === "auto") {
    return { mappingFormat: "직접 지정한 매핑은 데이터 형식을 골라야 합니다. 형식을 모르면 '자동 감지로 되돌리기'를 쓰세요." };
  }
  const used = mappedColumns(values);
  const errors: Partial<Record<RoleField, string>> = {};
  for (const role of REQUIRED_ROLES[values.mappingFormat]) {
    if (!used[role]) errors[ROLE_FIELD[role]] = `이 형식에 필요한 ${ROLE_LABEL[role]} 컬럼을 선택하세요.`;
  }
  return errors;
}

/** True when every column of the explicit mapping exists in `columns` (e.g. after a dataset or
 * config change, or for a mapping restored from an earlier analysis). */
export function mappingFitsColumns(values: MappingValues, columns: readonly string[]): boolean {
  const available = new Set(columns);
  return Object.values(mappedColumns(values)).every((column) => available.has(column));
}
