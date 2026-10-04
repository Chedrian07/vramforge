"use client";

import type { ReactNode } from "react";
import { Controller, useFormContext, type FieldPath } from "react-hook-form";

import { SwitchField } from "@/components/ui/controls";
import { Badge, Field, NativeSelect, TextArea, TextInput, describedBy } from "@/components/ui/primitives";
import type { FormValues } from "@/lib/form/values";

type Name = FieldPath<FormValues>;

/** Registered text input with error + hint wiring. */
export function TextField({
  name,
  label,
  hint,
  placeholder,
  numeric,
  disabled,
  applied,
}: {
  name: Name;
  label: ReactNode;
  hint?: ReactNode;
  placeholder?: string;
  numeric?: boolean;
  disabled?: boolean;
  applied?: ReactNode;
}) {
  const { register, getFieldState, formState } = useFormContext<FormValues>();
  const error = getFieldState(name, formState).error?.message;
  const id = `f-${name}`;
  return (
    <Field label={label} htmlFor={id} hint={hint} error={error} trailing={applied != null && applied !== "" ? <AppliedValue>{applied}</AppliedValue> : undefined}>
      <TextInput
        id={id}
        inputMode={numeric ? "decimal" : undefined}
        className={numeric ? "num" : undefined}
        placeholder={placeholder}
        disabled={disabled}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy(id, { hint, error })}
        {...register(name)}
      />
    </Field>
  );
}

export function ListField({ name, label, hint, placeholder, disabled }: { name: Name; label: ReactNode; hint?: ReactNode; placeholder?: string; disabled?: boolean }) {
  const { register, getFieldState, formState } = useFormContext<FormValues>();
  const error = getFieldState(name, formState).error?.message;
  const id = `f-${name}`;
  return (
    <Field label={label} htmlFor={id} hint={hint} error={error}>
      <TextArea
        id={id}
        rows={2}
        placeholder={placeholder}
        disabled={disabled}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy(id, { hint, error })}
        {...register(name)}
      />
    </Field>
  );
}

export function SelectField<T extends string>({
  name,
  label,
  options,
  hint,
  disabled,
  applied,
}: {
  name: Name;
  label: ReactNode;
  options: Array<{ value: T; label: string; disabled?: boolean }>;
  hint?: ReactNode;
  disabled?: boolean;
  applied?: ReactNode;
}) {
  const { register } = useFormContext<FormValues>();
  const id = `f-${name}`;
  return (
    <Field label={label} htmlFor={id} hint={hint} trailing={applied != null && applied !== "" ? <AppliedValue>{applied}</AppliedValue> : undefined}>
      <NativeSelect id={id} disabled={disabled} aria-describedby={describedBy(id, { hint })} {...register(name)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </NativeSelect>
    </Field>
  );
}

export function SwitchControl({ name, label, description, disabled }: { name: Name; label: ReactNode; description?: ReactNode; disabled?: boolean }) {
  const { control } = useFormContext<FormValues>();
  return (
    <Controller
      control={control}
      name={name}
      render={({ field }) => (
        <SwitchField id={`f-${name}`} label={label} checked={Boolean(field.value)} onCheckedChange={field.onChange} description={description} disabled={disabled} />
      )}
    />
  );
}

/** A switch that cannot be enabled in this release; the reason is always visible. */
export function UnsupportedSwitch({ id, label, reason }: { id: string; label: ReactNode; reason: string }) {
  return <SwitchField id={id} label={label} checked={false} onCheckedChange={() => {}} disabled description={reason} trailing={<Badge tone="neutral">미지원</Badge>} />;
}

/** "적용값" shown next to an input after an analysis resolved it (requested vs resolved). */
export function AppliedValue({ children }: { children: ReactNode }) {
  return <span className="text-[11px] whitespace-nowrap text-muted">적용값 <span className="num font-medium text-ink-2">{children}</span></span>;
}

export function enumOptions<T extends string>(labels: Record<T, string>, disabled: Partial<Record<T, boolean>> = {}) {
  return (Object.keys(labels) as T[]).map((value) => ({ value, label: labels[value], disabled: disabled[value] }));
}
