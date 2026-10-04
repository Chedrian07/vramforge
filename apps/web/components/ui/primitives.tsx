"use client";

import { Tooltip } from "radix-ui";
import {
  Fragment,
  forwardRef,
  useId,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { cn } from "@/lib/cn";
import type { Tone } from "@/lib/result/status";

// ---------------------------------------------------------------- buttons

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-accent-ink border border-transparent hover:brightness-110 active:brightness-95",
  secondary: "bg-surface text-ink border border-line-strong hover:bg-surface-2",
  ghost: "bg-transparent text-ink-2 border border-transparent hover:bg-surface-2",
  danger: "bg-surface text-err border border-err/50 hover:bg-err-soft",
};

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "sm" | "md" | "lg" }
>(function Button({ variant = "secondary", size = "md", className, type = "button", ...props }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium whitespace-nowrap transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50 aria-disabled:cursor-not-allowed aria-disabled:opacity-60",
        size === "sm" && "h-8 px-3 text-[13px]",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-[15px]",
        BUTTON_VARIANT[variant],
        className,
      )}
      {...props}
    />
  );
});

// ---------------------------------------------------------------- form controls

export const controlClass = cn(
  "w-full rounded-lg border border-line-strong bg-surface px-3 text-[15px] text-ink",
  "placeholder:text-muted/80 disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-muted",
  "aria-[invalid=true]:border-err",
);

export const TextInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function TextInput({ className, ...props }, ref) {
    return <input ref={ref} className={cn(controlClass, "h-10", className)} {...props} />;
  },
);

export const TextArea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function TextArea({ className, ...props }, ref) {
    return <textarea ref={ref} className={cn(controlClass, "min-h-20 py-2 font-mono text-[13px]", className)} {...props} />;
  },
);

export const NativeSelect = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  function NativeSelect({ className, children, ...props }, ref) {
    return (
      <select ref={ref} className={cn(controlClass, "h-10 appearance-auto pr-8", className)} {...props}>
        {children}
      </select>
    );
  },
);

/** Label + control + hint + error, wired with htmlFor / aria-describedby (plan §3.2). */
export function Field({
  label,
  htmlFor,
  hint,
  error,
  children,
  className,
  trailing,
}: {
  label: ReactNode;
  htmlFor: string;
  hint?: ReactNode;
  error?: string;
  children: ReactNode;
  className?: string;
  trailing?: ReactNode;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="text-[13px] font-medium text-ink-2">
          {label}
        </label>
        {trailing}
      </div>
      {children}
      {hint && !error ? (
        <p id={`${htmlFor}-hint`} className="text-[12px] leading-snug text-muted">
          {hint}
        </p>
      ) : null}
      {error ? (
        <p id={`${htmlFor}-error`} role="alert" className="text-[12px] leading-snug text-err">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function describedBy(id: string, opts: { hint?: unknown; error?: unknown }): string | undefined {
  const ids = [opts.error ? `${id}-error` : null, opts.hint && !opts.error ? `${id}-hint` : null].filter(Boolean);
  return ids.length ? ids.join(" ") : undefined;
}

// ---------------------------------------------------------------- badges and tone

export const TONE_CLASS: Record<Tone, string> = {
  ok: "bg-ok-soft text-ok border-ok/30",
  warn: "bg-warn-soft text-warn border-warn/30",
  err: "bg-err-soft text-err border-err/30",
  info: "bg-info-soft text-info border-info/30",
  neutral: "bg-neutral-soft text-muted border-line",
};

const TONE_GLYPH: Record<Tone, string> = { ok: "✓", warn: "!", err: "×", info: "i", neutral: "–" };

/** Labels up to this length ("분석식", "판정 안 함", "제외") never wrap. */
const SHORT_BADGE_LABEL = 12;

/**
 * Status pill: tone colour AND a text label (never colour alone). A short label stays on one
 * line and a longer one wraps between words only: in a narrow table cell or flex row a pill is
 * never broken syllable by syllable (Korean breaks between any two syllables by default).
 */
export function Badge({ tone, children, className, title }: { tone: Tone; children: ReactNode; className?: string; title?: string }) {
  const short = typeof children === "string" && children.length <= SHORT_BADGE_LABEL;
  return (
    <span
      title={title}
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 rounded-full border px-2 py-0.5 text-[12px] leading-5 font-medium",
        TONE_CLASS[tone],
        className,
      )}
    >
      <span aria-hidden className="font-mono text-[11px]">
        {TONE_GLYPH[tone]}
      </span>
      <span className={cn("min-w-0", short ? "whitespace-nowrap" : "break-keep wrap-break-word")}>{children}</span>
    </span>
  );
}

// ---------------------------------------------------------------- layout

export function Card({
  children,
  className,
  as: Tag = "section",
  ...rest
}: { children: ReactNode; className?: string; as?: "section" | "div" | "aside" } & Record<string, unknown>) {
  return (
    <Tag className={cn("rounded-xl border border-line bg-surface", className)} {...rest}>
      {children}
    </Tag>
  );
}

export function SectionTitle({ children, id, aside }: { children: ReactNode; id?: string; aside?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 id={id} className="text-lg font-semibold tracking-tight text-ink">
        {children}
      </h2>
      {aside}
    </div>
  );
}

/** Monospace identifier that wraps instead of overflowing (plan §3.2). */
export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("font-mono text-[13px] wrap-anywhere", className)}>{children}</span>;
}

/** Break points of an identifier: after a separator, and inside long runs without one. */
const IDENT_SEPARATOR = /(?<=[._:/@,=#-])/;
const IDENT_RUN = 16;

export function identParts(text: string): string[] {
  return text
    .split(IDENT_SEPARATOR)
    .flatMap((part) => (part.length <= IDENT_RUN ? [part] : (part.match(new RegExp(`.{1,${IDENT_RUN}}`, "gu")) ?? [part])));
}

/**
 * Monospace identifier for table cells (allocation names, timepoints, config fields): it wraps
 * at its separators (`.`, `_`, `:` …), never at an arbitrary character. `Mono` may break
 * anywhere, and inside an auto-layout table that lets the browser squeeze the column to one
 * character per line.
 */
export function Ident({ children, className }: { children: string; className?: string }) {
  return (
    <span className={cn("font-mono text-[13px] wrap-break-word", className)}>
      {identParts(children).map((part, i) => (
        <Fragment key={i}>
          {i > 0 ? <wbr /> : null}
          {part}
        </Fragment>
      ))}
    </span>
  );
}

export function InfoTip({ label, children, className }: { label: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <span tabIndex={0} className={cn("cursor-help underline decoration-dotted decoration-1 underline-offset-4", className)}>
          {label}
        </span>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-lg border border-line bg-surface px-3 py-2 text-[12px] text-ink-2 shadow-sm"
        >
          {children}
          <Tooltip.Arrow className="fill-surface" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function useFieldId(prefix: string): string {
  const id = useId();
  return `${prefix}-${id.replace(/[^a-zA-Z0-9_-]/g, "")}`;
}
