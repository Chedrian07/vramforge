// Metadata inspection requests built from the form (shared by the inputs and the example loader).
import type { ColumnMapping, DatasetInspection, InspectRequest } from "@/lib/api/types";

import { mappingFitsColumns, type RoleField } from "./mapping";
import { datasetSourceType, isValidDatasetReference, isValidModelReference, modelSourceType } from "./references";
import type { FormValues } from "./values";

export interface InspectCall {
  key: string;
  body: InspectRequest;
}

export function modelInspectCall(values: Pick<FormValues, "modelReference" | "modelRevision" | "loadingScope">): InspectCall | null {
  const reference = values.modelReference.trim();
  if (!isValidModelReference(reference)) return null;
  const revision = values.modelRevision.trim();
  return {
    key: JSON.stringify([reference, revision, values.loadingScope]),
    body: {
      model: { source_type: modelSourceType(reference), reference, revision: revision || null, loading_scope: values.loadingScope },
      dataset: null,
      objective: null,
    },
  };
}

export function datasetInspectCall(
  values: Pick<FormValues, "datasetReference" | "datasetUploadRef" | "datasetRevision" | "datasetConfig" | "objective">,
): InspectCall | null {
  const reference = values.datasetReference.trim();
  if (!isValidDatasetReference(reference)) return null;
  const revision = values.datasetRevision.trim();
  const config = values.datasetConfig.trim();
  return {
    // The objective is part of the key: the server ranks mapping candidates (and decides whether
    // they are ambiguous) for it.
    key: JSON.stringify([reference, revision, config, values.objective]),
    body: {
      model: null,
      objective: values.objective,
      dataset: {
        source_type: datasetSourceType(reference, values.datasetUploadRef),
        reference,
        revision: revision || null,
        config: config || null,
        split: null,
        eval_split: null,
        scan_mode: "full",
        sample_rows: null,
        mapping: null,
      },
    },
  };
}

/** The inspected reference string of an inspection key. */
export function referenceOfKey(key: string | null): string | null {
  if (!key) return null;
  try {
    const parsed: unknown = JSON.parse(key);
    return Array.isArray(parsed) && typeof parsed[0] === "string" ? parsed[0] : null;
  } catch {
    return null;
  }
}

/** What an inspection answer means for the mapping in the form (steps run in this order). */
export interface MappingSync {
  /** Back to auto-detection first. */
  clear: boolean;
  /** The empty-system policy belonged to a mapping for other columns: back to the default. */
  resetPolicy: boolean;
  /** Then apply this unambiguous suggestion automatically. */
  apply: ColumnMapping | null;
}

type MappingState = Pick<FormValues, "mappingEnabled" | "mappingAutoApplied" | "mappingFormat" | RoleField>;

/**
 * A mapping the user chose stays as long as its columns exist. A suggestion that was applied
 * automatically is not a choice: it follows the latest inspection (which is ranked for the current
 * objective and config), and it is dropped when that inspection has no unambiguous suggestion, so
 * an earlier answer never decides an ambiguity the server would ask about.
 */
export function mappingSyncAfterInspection(data: DatasetInspection, values: MappingState): MappingSync {
  const columns = (data.columns ?? []).map((c) => c.name);
  const suggestion = data.suggested_mapping && !data.mapping_ambiguous ? data.suggested_mapping : null;
  const gone = values.mappingEnabled && columns.length > 0 && !mappingFitsColumns(values, columns);
  if (gone) return { clear: true, resetPolicy: true, apply: suggestion };
  if (values.mappingEnabled && !values.mappingAutoApplied) return { clear: false, resetPolicy: false, apply: null };
  if (suggestion) return { clear: false, resetPolicy: false, apply: suggestion };
  return { clear: values.mappingEnabled, resetPolicy: false, apply: null };
}
