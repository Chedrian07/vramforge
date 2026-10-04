"use client";

import { Bar, BarChart, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Mono } from "@/components/ui/primitives";
import { BytesRange, NotComputed } from "@/components/ui/values";
import type { ScenarioEstimate } from "@/lib/api/types";
import { exactBytes, formatSize, gibNumber } from "@/lib/format/bytes";
import { CATEGORY_LABEL, EVIDENCE_LABEL, PHASE_LABEL } from "@/lib/format/labels";
import { groupOf, groupTotals } from "@/lib/result/groups";

import {
  AXIS_PROPS,
  ChartBox,
  ChartFigure,
  DataTable,
  LegendSwatch,
  Td,
  Th,
  TooltipBox,
  type ChartTooltipProps,
} from "../chart-kit";

function PeakTooltip({ active, payload }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="flex flex-col gap-1">
      {payload.map((entry) => (
        <TooltipBox
          key={String(entry.dataKey)}
          value={formatSize(Number(entry.value))}
          label={String(entry.name)}
          detail={exactBytes(Number(entry.value))}
        />
      ))}
    </div>
  );
}

/** Peak timepoint composition: every segment belongs to the SAME timepoint (plan.md §12.2). */
export function MemoryTab({ scenario }: { scenario: ScenarioEstimate | null }) {
  const device = scenario?.devices[0] ?? null;
  const breakdown = device?.peak_breakdown ?? null;
  if (!scenario || !device) return <p className="text-[13px] text-muted">메모리 산정 결과가 없습니다.</p>;
  if (!breakdown) {
    return <NotComputed reason="피크 시점 구성(breakdown)이 결과에 없습니다." />;
  }
  const { totals, unknown } = groupTotals(breakdown.items);
  const row: Record<string, number | string> = { name: breakdown.timepoint };
  for (const t of totals) row[t.group.key] = t.high;
  const knownHigh = totals.reduce((acc, t) => acc + t.high, 0);
  const phaseLabel = PHASE_LABEL[breakdown.phase];

  const chart = (
    <ChartBox height={84}>
      {(width) => (
        <BarChart
          width={width}
          height={84}
          layout="vertical"
          data={[row]}
          margin={{ top: 4, right: 12, bottom: 4, left: 12 }}
          barSize={24}
        >
          <XAxis
            type="number"
            domain={[0, "dataMax"]}
            tickFormatter={(v: number) => `${gibNumber(v)}`}
            unit=" GiB"
            {...AXIS_PROPS}
          />
          <YAxis type="category" dataKey="name" hide />
          <Tooltip content={PeakTooltip} cursor={false} isAnimationActive={false} />
          {totals.map((t, i) => (
            <Bar
              key={t.group.key}
              dataKey={t.group.key}
              name={t.group.label}
              stackId="peak"
              fill={t.group.color}
              stroke="var(--vf-surface)"
              strokeWidth={2}
              radius={i === totals.length - 1 ? [0, 4, 4, 0] : 0}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      )}
    </ChartBox>
  );

  const legend = (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5" aria-label="구성 범례">
      {totals.map((t) => (
        <li key={t.group.key} className="flex items-center gap-1.5 text-[12px]">
          <LegendSwatch color={t.group.color} />
          <span className="text-ink-2">{t.group.label}</span>
          <span className="num text-muted">{formatSize(t.high)}</span>
        </li>
      ))}
    </ul>
  );

  const table = (
    <DataTable
      caption={`피크 시점 ${breakdown.timepoint}에 살아 있는 allocation (상한 합 ${formatSize(breakdown.total_high) ?? "산정 불가"})`}
      head={
        <tr>
          <Th>allocation</Th>
          <Th>분류</Th>
          <Th align="right">하한 – 상한</Th>
          <Th>근거</Th>
          <Th>메모</Th>
        </tr>
      }
    >
      {breakdown.items.map((item, i) => (
        <tr key={`${item.name}-${i}`}>
          <Td>
            <span className="flex items-center gap-2">
              <LegendSwatch color={groupOf(item.category).color} />
              <span className="wrap-anywhere">{item.name}</span>
            </span>
          </Td>
          <Td>{CATEGORY_LABEL[item.category]}</Td>
          <Td align="right">
            <BytesRange low={item.bytes_low} high={item.bytes_high} />
          </Td>
          <Td>
            <Badge tone={item.evidence === "unknown" ? "warn" : item.evidence === "assumption" ? "neutral" : "info"}>
              {EVIDENCE_LABEL[item.evidence]}
            </Badge>
          </Td>
          <Td className="text-[12px] text-muted">{item.note ?? ""}</Td>
        </tr>
      ))}
      <tr className="font-medium">
        <Td>합계 (같은 시점)</Td>
        <Td>—</Td>
        <Td align="right">
          <BytesRange
            low={breakdown.total_low}
            high={breakdown.total_high}
            reason="크기 미상 항목이 있어 합계를 만들지 않습니다."
          />
        </Td>
        <Td>—</Td>
        <Td className="text-[12px] text-muted">{unknown.length > 0 ? `산정 불가 ${unknown.length}건 포함` : ""}</Td>
      </tr>
    </DataTable>
  );

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[13px] text-ink-2">
        피크 시점 <Mono>{breakdown.timepoint}</Mono> ({phaseLabel}). 막대의 합은 서로 다른 시점의 최대값을 쌓은 것이
        아니라 이 한 시점에 동시에 살아 있는 allocation의 상한 합입니다.
      </p>
      <ChartFigure
        title="피크 시점 메모리 구성"
        description="상한(high) 기준. 정확한 bytes와 하한은 표에 있습니다."
        label={`피크 시점 ${breakdown.timepoint}의 구성: ${totals.map((t) => `${t.group.label} ${formatSize(t.high)}`).join(", ")}`}
        chart={
          <div className="flex flex-col gap-2">
            {chart}
            {legend}
            {unknown.length > 0 ? (
              <p className="text-[12px] text-warn">
                크기 미상 {unknown.length}건은 막대에 그리지 않았습니다 ({unknown.map((u) => u.name).join(", ")}).
                표시된 합 {formatSize(knownHigh)}은 확정된 항목만의 합입니다.
              </p>
            ) : null}
          </div>
        }
        table={table}
      />
    </div>
  );
}
