"use client";

import { useEffect } from "react";
import type { FieldPath, UseFormReturn } from "react-hook-form";

import type { FormValues } from "./values";

type Name = FieldPath<FormValues>;

/**
 * Value comparisons of formSchema.superRefine: editing the key re-checks these fields as soon as
 * they hold a value (an empty field is never compared, so no error appears before it is used).
 */
const COMPARED_WITH: Partial<Record<Name, readonly Name[]>> = {
  hardwareTotalGiB: ["hardwareUsableGiB"],
  hardwareMode: ["hardwareUsableGiB"],
};

/**
 * Cross-field rules (formSchema.superRefine) report on one field but depend on others, while React
 * Hook Form re-validates only the edited field. Without this an error could outlive its cause, e.g.
 * "사용 가능 VRAM은 전체 용량 이하여야 합니다" after the total was raised, or stay hidden after the
 * total was lowered again. Every value change re-checks the other fields that show an error and the
 * filled fields compared with the edited one; required-when errors on untouched fields still wait
 * for their own edit or the submit.
 */
export function useRevalidateShownErrors(form: Pick<UseFormReturn<FormValues>, "watch" | "trigger" | "formState" | "getValues">): void {
  useEffect(() => {
    // watch(callback) fires on value changes only, so the trigger below cannot loop. formState is
    // read at call time: the form object is stable while its formState is replaced every render.
    const subscription = form.watch((_values, { name }) => {
      const shown = Object.keys(form.formState.errors) as Name[];
      const compared = (name ? (COMPARED_WITH[name as Name] ?? []) : []).filter((field) => {
        const value = form.getValues(field);
        return typeof value === "string" ? value.trim() !== "" : value != null;
      });
      const fields = [...new Set([...shown, ...compared])].filter((field) => field !== name);
      if (fields.length > 0) void form.trigger(fields);
    });
    return () => subscription.unsubscribe();
  }, [form]);
}
