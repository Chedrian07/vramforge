"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { TooltipContentProps, TooltipValueType } from "recharts";

import { cn } from "@/lib/cn";

/** Props passed to custom Recharts tooltip renderers. */
export type ChartTooltipProps = TooltipContentProps<TooltipValueType, number | string>;

/** Chart chrome: hairline solid grid/axes in recessive tokens (dataviz marks & anatomy). */
export const AXIS_PROPS = {
  stroke: "var(--vf-chart-axis)",
  tick: { fill: "var(--vf-muted)", fontSize: 11 },
  tickLine: false,
} as const;

export const GRID_STROKE = "var(--vf-chart-grid)";

/** Width used before the container is measured (SSR, jsdom); the box clips until measured. */
export const INITIAL_DIMENSION = { width: 640, height: 220 };

/**
 * Measures its own width and renders the chart at that width. Unlike ResponsiveContainer it never
 * renders a 0-width chart (jsdom/SSR) and it clips instead of widening the page before measuring.
 */
export function ChartBox({ height, children }: { height: number; children: (width: number) => ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(INITIAL_DIMENSION.width);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => {
      const measured = Math.floor(element.getBoundingClientRect().width);
      if (measured > 0) setWidth(measured);
    };
    const frame = requestAnimationFrame(update);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(element);
    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
    };
  }, []);
  return (
    <div ref={ref} className="w-full min-w-0 overflow-hidden" style={{ height }}>
      {children(width)}
    </div>
  );
}

/** A chart with its caption and an always-available table alternative (plan.md §3.2). */
export function ChartFigure({
  title,
  description,
  chart,
  table,
  label,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  chart: ReactNode;
  table: ReactNode;
  label: string;
  className?: string;
}) {
  return (
    <figure className={cn("flex min-w-0 flex-col gap-3", className)}>
      <figcaption className="flex flex-col gap-0.5">
        <span className="text-[14px] font-semibold text-ink">{title}</span>
        {description ? <span className="text-[12px] leading-snug text-muted">{description}</span> : null}
      </figcaption>
      <div role="img" aria-label={label} className="min-w-0">
        {chart}
      </div>
      <div className="min-w-0">{table}</div>
    </figure>
  );
}

export function TooltipBox({ value, label, detail }: { value: ReactNode; label: ReactNode; detail?: ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-[12px] shadow-sm">
      <div className="num text-[13px] font-semibold text-ink">{value}</div>
      <div className="text-ink-2">{label}</div>
      {detail ? <div className="num text-muted">{detail}</div> : null}
    </div>
  );
}

export function LegendSwatch({ color, shape = "rect" }: { color: string; shape?: "rect" | "line" }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block shrink-0", shape === "rect" ? "size-3 rounded-[3px]" : "h-0.5 w-4 rounded-full")}
      style={{ background: color }}
    />
  );
}

/** Simple accessible data table shell with a caption and horizontal scroll inside the card only. */
export function DataTable({ caption, head, children, captionHidden = false }: { caption: string; head: ReactNode; children: ReactNode; captionHidden?: boolean }) {
  return (
    <div className="max-w-full overflow-x-auto rounded-lg border border-line">
      <table className="w-full min-w-[28rem] border-collapse text-[13px]">
        <caption className={cn("px-3 py-2 text-left text-[12px] font-medium text-muted", captionHidden && "sr-only")}>{caption}</caption>
        <thead className="bg-surface-2 text-left text-[12px] text-muted">{head}</thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Th({ children, className, align = "left" }: { children: ReactNode; className?: string; align?: "left" | "right" }) {
  return (
    <th scope="col" className={cn("px-3 py-2 font-medium", align === "right" && "text-right", className)}>
      {children}
    </th>
  );
}

export function Td({ children, className, align = "left" }: { children: ReactNode; className?: string; align?: "left" | "right" }) {
  return <td className={cn("border-t border-line/70 px-3 py-2 align-top", align === "right" && "text-right", className)}>{children}</td>;
}
