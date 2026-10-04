"use client";

import { BytesRange, Bytes } from "@/components/ui/values";
import { Badge } from "@/components/ui/primitives";
import { formatCount, formatPercent } from "@/lib/format/bytes";
import { HARDWARE_FIT_LABEL } from "@/lib/format/labels";
import type { BudgetRow } from "@/lib/result/summary";
import { FIT_TONE } from "@/lib/result/status";

/** GRPO without an explicit completion budget: one compact row per budget scenario (plan.md §5.4). */
export function BudgetList({ rows, showFit }: { rows: BudgetRow[]; showFit: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[18rem] text-[13px]">
        <caption className="sr-only">completion budget별 예상 피크와 권장 용량</caption>
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-muted">
            <th scope="col" className="py-1.5 pr-2 font-medium">
              budget (token)
            </th>
            <th scope="col" className="py-1.5 pr-2 font-medium">
              예상 피크
            </th>
            <th scope="col" className="py-1.5 pr-2 font-medium">
              권장 용량
            </th>
            {showFit ? (
              <th scope="col" className="py-1.5 font-medium">
                GPU 적합
              </th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.scenarioId} className="border-b border-line/60 last:border-b-0">
              <th scope="row" className="num py-1.5 pr-2 text-left font-medium text-ink">
                {row.budget != null ? formatCount(row.budget) : row.label}
              </th>
              <td className="py-1.5 pr-2">
                <BytesRange low={row.low} high={row.high} unit="gib" reason={row.nullReason} />
              </td>
              <td className="py-1.5 pr-2">
                <Bytes value={row.recommended} reason={row.nullReason} />
              </td>
              {showFit ? (
                <td className="py-1.5">
                  {row.fit ? (
                    <Badge tone={FIT_TONE[row.fit.status]}>
                      {HARDWARE_FIT_LABEL[row.fit.status]}
                      {row.fit.utilization_ratio != null ? ` · 예상 ${formatPercent(row.fit.utilization_ratio)}` : ""}
                    </Badge>
                  ) : (
                    "—"
                  )}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
