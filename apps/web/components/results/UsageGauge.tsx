"use client";

import type { HardwareFitResult } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { formatGiB, formatPercent } from "@/lib/format/bytes";
import { FIT_REASON_LABEL, HARDWARE_FIT_LABEL } from "@/lib/format/labels";
import { FIT_TONE } from "@/lib/result/status";

const FILL: Record<string, string> = {
  ok: "bg-accent",
  warn: "bg-warn",
  err: "bg-err",
  info: "bg-accent",
  neutral: "bg-line-strong",
};

/**
 * Meter of (scenario high + planning margin) / capacity as reported by the server.
 * The bar never grows past the track; the exact percentage stays in the text (plan.md §3.4).
 */
export function UsageGauge({ fit, stale = false }: { fit: HardwareFitResult; stale?: boolean }) {
  const tone = FIT_TONE[fit.status];
  const ratio = fit.utilization_ratio ?? null;
  const percent = formatPercent(ratio);
  const width = ratio == null ? 0 : Math.min(Math.max(ratio, 0), 1) * 100;
  const label = `GPU 사용량 ${percent ? `예상 ${percent}` : "판정 보류"} · ${FIT_REASON_LABEL[fit.reason]}`;
  return (
    <div className={cn("flex flex-col gap-1.5", stale && "opacity-60")}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
        <span className="font-medium text-ink">GPU 사용량</span>
        <span className="num font-semibold text-ink">{percent ? `예상 ${percent}` : HARDWARE_FIT_LABEL[fit.status]}</span>
      </div>
      {ratio != null ? (
        <div role="img" aria-label={label} className="relative h-2.5 w-full overflow-hidden rounded-full bg-neutral-soft">
          <div className={cn("h-full rounded-full", FILL[tone])} style={{ width: `${width}%` }} />
        </div>
      ) : null}
      <p className="text-[12px] leading-snug text-ink-2">
        <strong className="font-medium">{FIT_REASON_LABEL[fit.reason]}</strong>
        {fit.capacity_bytes != null ? <span className="text-muted"> · 기준 용량 {formatGiB(fit.capacity_bytes)}</span> : null}
      </p>
      {fit.message ? <p className="text-[12px] leading-snug text-muted">{fit.message}</p> : null}
    </div>
  );
}
