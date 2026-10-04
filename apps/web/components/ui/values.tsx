"use client";

import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { exactBytes, exactBytesRange, formatGiBRange, formatSize, formatSizeRange } from "@/lib/format/bytes";
import { SEVERITY_LABEL } from "@/lib/format/labels";
import { SEVERITY_TONE } from "@/lib/result/status";

import { Badge, InfoTip, Mono } from "./primitives";

/** "산정 불가" with its reason; used wherever a number is unknown (never rendered as 0). */
export function NotComputed({ reason, compact = false }: { reason?: string | null; compact?: boolean }) {
  return (
    <span className="inline-flex flex-col">
      <span className="font-medium text-warn">산정 불가</span>
      {reason && !compact ? <span className="text-[12px] leading-snug text-muted">{reason}</span> : null}
    </span>
  );
}

/** One size with the exact byte count in a tooltip. */
export function Bytes({ value, reason, className }: { value: number | null | undefined; reason?: string | null; className?: string }) {
  if (value == null) return <NotComputed reason={reason} compact />;
  return (
    <InfoTip label={<span className={cn("num", className)}>{formatSize(value)}</span>}>
      <span className="num">{exactBytes(value)}</span>
    </InfoTip>
  );
}

/** low–high range; GiB for headline numbers, adaptive units in tables. */
export function BytesRange({
  low,
  high,
  unit = "adaptive",
  reason,
  className,
}: {
  low: number | null | undefined;
  high: number | null | undefined;
  unit?: "gib" | "adaptive";
  reason?: string | null;
  className?: string;
}) {
  const text = unit === "gib" ? formatGiBRange(low, high) : formatSizeRange(low, high);
  if (text == null) return <NotComputed reason={reason} compact />;
  return (
    <InfoTip label={<span className={cn("num", className)}>{text}</span>}>
      <span className="num">{exactBytesRange(low, high)}</span>
    </InfoTip>
  );
}

export interface DisplayIssue {
  code: string;
  severity: "info" | "warning" | "error";
  user_message: string;
  affected_component?: string | null;
}

export function IssueList({ issues, empty, limit }: { issues: DisplayIssue[]; empty?: ReactNode; limit?: number }) {
  if (issues.length === 0) return empty ? <p className="text-[13px] text-muted">{empty}</p> : null;
  const shown = limit ? issues.slice(0, limit) : issues;
  return (
    <ul className="flex flex-col gap-2">
      {shown.map((issue, index) => (
        <li key={`${issue.code}-${index}`} className="flex min-w-0 flex-col gap-1 sm:flex-row sm:items-start sm:gap-2">
          <Badge tone={SEVERITY_TONE[issue.severity]}>{SEVERITY_LABEL[issue.severity]}</Badge>
          <div className="min-w-0 flex-1 text-[13px] leading-snug">
            <span className="text-ink">{issue.user_message}</span>{" "}
            <Mono className="text-[11px] text-muted">{issue.code}</Mono>
          </div>
        </li>
      ))}
      {limit && issues.length > limit ? (
        <li className="text-[12px] text-muted">외 {issues.length - limit}건은 &lsquo;적용 설정·근거&rsquo; 탭에서 확인하세요.</li>
      ) : null}
    </ul>
  );
}
