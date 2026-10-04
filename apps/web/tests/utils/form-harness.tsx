import { zodResolver } from "@hookform/resolvers/zod";
import type { ReactNode } from "react";
import { FormProvider, useForm, type UseFormReturn } from "react-hook-form";

import { useRevalidateShownErrors } from "@/lib/form/revalidate";
import { DEFAULT_FORM_VALUES, formSchema, type FormValues } from "@/lib/form/values";

/** Wraps form sections in the same React Hook Form setup as the calculator. */
export function FormHarness({
  children,
  values = {},
  onReady,
}: {
  children: ReactNode;
  values?: Partial<FormValues>;
  onReady?: (form: UseFormReturn<FormValues>) => void;
}) {
  const form = useForm<FormValues>({
    defaultValues: { ...DEFAULT_FORM_VALUES, ...values },
    resolver: zodResolver(formSchema),
    mode: "onChange",
  });
  useRevalidateShownErrors(form);
  onReady?.(form);
  return <FormProvider {...form}>{children}</FormProvider>;
}
