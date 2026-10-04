"use client";

import { Switch as RadixSwitch, ToggleGroup } from "radix-ui";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";

/** Labelled switch (Radix Switch: role="switch", Space/Enter toggles). */
export function SwitchField({
  id,
  label,
  checked,
  onCheckedChange,
  disabled,
  description,
  trailing,
}: {
  id: string;
  label: ReactNode;
  checked: boolean;
  onCheckedChange: (value: boolean) => void;
  disabled?: boolean;
  description?: ReactNode;
  trailing?: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-start gap-3">
      <RadixSwitch.Root
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
        aria-describedby={description ? `${id}-desc` : undefined}
        className={cn(
          "relative mt-0.5 inline-flex h-6 w-10 shrink-0 items-center rounded-full border border-line-strong bg-neutral-soft transition-colors",
          "data-[state=checked]:border-accent data-[state=checked]:bg-accent disabled:cursor-not-allowed disabled:opacity-50",
        )}
      >
        <RadixSwitch.Thumb className="block size-4.5 translate-x-0.5 rounded-full bg-surface shadow-sm transition-transform data-[state=checked]:translate-x-[18px]" />
      </RadixSwitch.Root>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor={id} className={cn("text-sm font-medium text-ink", disabled && "text-muted")}>
            {label}
          </label>
          {trailing}
        </div>
        {description ? (
          <p id={`${id}-desc`} className="mt-0.5 text-[12px] leading-snug text-muted">
            {description}
          </p>
        ) : null}
      </div>
    </div>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  description?: string;
}

/** Segmented control (Radix ToggleGroup type="single": roving focus with arrow keys). */
export function Segmented<T extends string>({
  label,
  labelId,
  value,
  onValueChange,
  options,
  className,
  size = "md",
}: {
  label: string;
  labelId?: string;
  value: T;
  onValueChange: (value: T) => void;
  options: SegmentOption<T>[];
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(next) => {
        if (next) onValueChange(next as T);
      }}
      aria-label={labelId ? undefined : label}
      aria-labelledby={labelId}
      className={cn("inline-flex max-w-full flex-wrap rounded-lg border border-line-strong bg-surface-2 p-0.5", className)}
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.value}
          value={option.value}
          disabled={option.disabled}
          title={option.description}
          className={cn(
            "rounded-md font-medium text-ink-2 transition-colors",
            size === "md" ? "h-9 min-w-16 px-4 text-sm" : "h-7 px-3 text-[13px]",
            "hover:text-ink data-[state=on]:bg-surface data-[state=on]:text-accent data-[state=on]:shadow-[0_0_0_1px_var(--vf-line-strong)]",
            "disabled:cursor-not-allowed disabled:opacity-45",
          )}
        >
          {option.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

/**
 * Puts text on the clipboard. The Clipboard API exists only in secure contexts, so plain-HTTP LAN
 * deployments fall back to copying a temporary selection; false when neither works.
 */
export async function copyText(value: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && typeof navigator.clipboard?.writeText === "function") {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Denied or unavailable: try the selection fallback below.
  }
  if (typeof document === "undefined" || typeof document.execCommand !== "function") return false;
  const area = document.createElement("textarea");
  area.value = value;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  area.remove();
  return copied;
}

/**
 * Copies a value (ids, digests) so long strings never need horizontal scrolling. When the browser
 * cannot copy, the value is offered selected in a read-only field to copy by hand.
 */
export function CopyButton({ value, label = "복사" }: { value: string; label?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  if (state === "failed") {
    return (
      <span className="inline-flex min-w-0 items-center gap-1">
        <input
          readOnly
          autoFocus
          value={value}
          aria-label={`직접 복사할 값: ${value}`}
          onFocus={(event) => event.currentTarget.select()}
          onBlur={() => setState("idle")}
          className="h-6 w-40 min-w-0 rounded-md border border-line bg-surface px-1 font-mono text-[11px] text-ink"
        />
        <span role="status" className="shrink-0 text-[11px] text-warn">
          자동 복사 불가 · 직접 복사하세요
        </span>
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={async () => {
        const copied = await copyText(value);
        if (timer.current) clearTimeout(timer.current);
        if (!copied) {
          setState("failed");
          return;
        }
        setState("copied");
        timer.current = setTimeout(() => setState("idle"), 1500);
      }}
      className="inline-flex h-6 shrink-0 items-center rounded-md border border-line px-2 text-[11px] text-muted hover:bg-surface-2 hover:text-ink"
      aria-label={`${label}: ${value}`}
    >
      {state === "copied" ? "복사됨" : label}
    </button>
  );
}
