// Metadata inspection requests built from the form (shared by the inputs and the example loader).
import type { DatasetInspection, InspectRequest } from "@/lib/api/types";

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
    key: JSON.stringify([reference, revision, config]),
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

/** True when an unambiguous server suggestion should be applied to an unedited mapping. */
export function shouldApplySuggestion(data: DatasetInspection, mappingEdited: boolean): boolean {
  return data.suggested_mapping != null && !data.mapping_ambiguous && !mappingEdited;
}
