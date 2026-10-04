// Dataset-specific selections belong to the dataset (and config) they were made for
// (plan.md §4.1, §7.2): another dataset or config starts again from auto-selection.
import type { UseFormGetValues, UseFormSetValue } from "react-hook-form";

import type { FormValues } from "@/lib/form/values";

import { clearMapping, dropAutoAppliedMapping } from "./MappingEditor";

const OPTS = { shouldDirty: true, shouldValidate: true } as const;

/** Another dataset (typed or uploaded): config, splits, revision and mapping start over. */
export function resetDatasetSelections(setValue: UseFormSetValue<FormValues>) {
  for (const name of ["datasetConfig", "datasetSplit", "datasetEvalSplit", "datasetRevision"] as const) {
    setValue(name, "", OPTS);
  }
  clearMapping(setValue);
}

/**
 * Another config of the same dataset: its splits may differ, so the split choices start over, and
 * a mapping applied automatically for the previous config goes too (the next inspection suggests
 * again). A mapping the user chose stays; the inspection drops it if its columns are gone.
 */
export function resetConfigSelections(setValue: UseFormSetValue<FormValues>, getValues: UseFormGetValues<FormValues>) {
  setValue("datasetSplit", "", OPTS);
  setValue("datasetEvalSplit", "", OPTS);
  dropAutoAppliedMapping(setValue, getValues);
}
