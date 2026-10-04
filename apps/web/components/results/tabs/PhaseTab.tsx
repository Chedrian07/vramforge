"use client";

import { Bar, BarChart, CartesianGrid, Cell, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Mono } from "@/components/ui/primitives";
import { BytesRange, Bytes } from "@/components/ui/values";
import type { ScenarioEstimate } from "@/lib/api/types";
import { exactBytesRange, formatSizeRange } from "@/lib/format/bytes";
import { PHASE_LABEL } from "@/lib/format/labels";

import {
  AXIS_PROPS,
  ChartBox,
  ChartFigure,
  DataTable,
  GRID_STROKE,
  LegendSwatch,
  Td,
  Th,
  TooltipBox,
  gibTickLabel,
  gibTicks,
  type ChartTooltipProps,
} from "../chart-kit";

interface PhaseRow {
  phase: string;
  label: string;
  low: number;
  extra: number;
  high: number;
  isPeak: boolean;
}

function PhaseTooltip({ active, payload }: ChartTooltipProps) {
  const entry = payload?.[0];
  if (!active || !entry) return null;
  const row = entry.payload as PhaseRow;
  return (
    <TooltipBox
      value={formatSizeRange(row.low, row.high)}
      label={`${row.label}${row.isPeak ? " · 피크 단계" : ""}`}
      detail={exactBytesRange(row.low, row.high)}
    />
  );
}

export function PhaseTab({ scenario }: { scenario: ScenarioEstimate | null }) {
  const device = scenario?.devices[0] ?? null;
  if (!scenario || !device) return <p className="text-[13px] text-muted">단계별 산정 결과가 없습니다.</p>;
  const rows: PhaseRow[] = device.phases
    .filter((p) => p.included && p.bytes_low != null && p.bytes_high != null)
    .map((p) => ({
      phase: p.phase,
      label: PHASE_LABEL[p.phase],
      low: p.bytes_low ?? 0,
      extra: (p.bytes_high ?? 0) - (p.bytes_low ?? 0),
      high: p.bytes_high ?? 0,
      isPeak: p.phase === device.peak_phase,
    }));
  const height = Math.max(120, rows.length * 40 + 48);
  const axis = gibTicks(Math.max(0, ...rows.map((r) => r.high)));

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[13px] text-ink-2">
        단계마다 동시에 살아 있는 메모리의 최대값입니다. 서로 겹치지 않는 단계의 최대값을 더하지 않으며, 전체 피크는
        가장 큰 단계의 값입니다.
        {device.peak_timepoint ? (
          <>
            {" "}
            피크 시점: <Mono>{device.peak_timepoint}</Mono>
          </>
        ) : null}
      </p>
      <ChartFigure
        title="단계별 피크"
        description="진한 부분은 하한, 연한 부분은 상한까지의 범위입니다. 피크 단계만 강조색으로 표시합니다."
        label={`단계별 피크: ${rows.map((r) => `${r.label} ${formatSizeRange(r.low, r.high)}`).join(", ")}`}
        chart={
          rows.length === 0 ? (
            <p className="text-[13px] text-muted">크기를 산정한 단계가 없습니다.</p>
          ) : (
            <div className="flex flex-col gap-2">
              <ChartBox height={height}>
                {(width) => (
                  <BarChart
                    width={width}
                    height={height}
                    layout="vertical"
                    data={rows}
                    margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
                    barSize={20}
                  >
                    <CartesianGrid horizontal={false} stroke={GRID_STROKE} />
                    <XAxis
                      type="number"
                      domain={[0, axis.max]}
                      ticks={axis.ticks}
                      tickFormatter={gibTickLabel}
                      unit=" GiB"
                      {...AXIS_PROPS}
                    />
                    <YAxis type="category" dataKey="label" width={132} {...AXIS_PROPS} />
                    <Tooltip
                      content={PhaseTooltip}
                      cursor={{ fill: "var(--vf-accent-soft)" }}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="low"
                      name="하한"
                      stackId="range"
                      stroke="var(--vf-surface)"
                      strokeWidth={2}
                      isAnimationActive={false}
                    >
                      {rows.map((r) => (
                        <Cell key={r.phase} fill={r.isPeak ? "var(--vf-accent)" : "var(--vf-chart-muted)"} />
                      ))}
                    </Bar>
                    <Bar
                      dataKey="extra"
                      name="상한까지"
                      stackId="range"
                      stroke="var(--vf-surface)"
                      strokeWidth={2}
                      radius={[0, 4, 4, 0]}
                      isAnimationActive={false}
                    >
                      {rows.map((r) => (
                        <Cell
                          key={r.phase}
                          fill={r.isPeak ? "var(--vf-chart-accent-wash)" : "var(--vf-chart-muted-wash)"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                )}
              </ChartBox>
              <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink-2" aria-label="단계 차트 범례">
                <li className="flex items-center gap-1.5">
                  <LegendSwatch color="var(--vf-accent)" /> 피크 단계 하한
                </li>
                <li className="flex items-center gap-1.5">
                  <LegendSwatch color="var(--vf-chart-accent-wash)" /> 피크 단계 상한까지
                </li>
                <li className="flex items-center gap-1.5">
                  <LegendSwatch color="var(--vf-chart-muted)" /> 다른 단계
                </li>
              </ul>
            </div>
          )
        }
        table={
          <DataTable
            caption="단계별 피크 (제외한 단계 포함)"
            head={
              <tr>
                <Th>단계</Th>
                <Th>포함</Th>
                <Th>피크 시점</Th>
                <Th align="right">하한 – 상한</Th>
                <Th align="right">확정 상주량</Th>
                <Th>비고</Th>
              </tr>
            }
          >
            {device.phases.map((p) => (
              <tr key={p.phase}>
                <Td>
                  <span className="font-medium text-ink">{PHASE_LABEL[p.phase]}</span>
                  {p.phase === device.peak_phase ? (
                    <Badge tone="info" className="ml-2">
                      피크
                    </Badge>
                  ) : null}
                </Td>
                <Td>{p.included ? "포함" : <Badge tone="neutral">제외</Badge>}</Td>
                <Td>{p.peak_timepoint ? <Mono className="text-[12px]">{p.peak_timepoint}</Mono> : "—"}</Td>
                <Td align="right">
                  {p.included ? <BytesRange low={p.bytes_low} high={p.bytes_high} reason="미상 항목 포함" /> : "—"}
                </Td>
                <Td align="right">
                  {p.included && p.known_floor_bytes != null ? <Bytes value={p.known_floor_bytes} /> : "—"}
                </Td>
                <Td className="text-[12px] text-muted">
                  {p.excluded_reason ?? ""}
                  {(p.unknown_components ?? []).length > 0
                    ? ` 산정 불가: ${(p.unknown_components ?? []).join(", ")}`
                    : ""}
                </Td>
              </tr>
            ))}
          </DataTable>
        }
      />
      {device.timepoints.length > 0 ? (
        <details>
          <summary className="cursor-pointer text-[13px] text-ink-2 select-none">
            모든 시점의 합계 보기 ({device.timepoints.length}개)
          </summary>
          <div className="mt-2">
            <DataTable
              caption="시점별 동시 상주량"
              captionHidden
              head={
                <tr>
                  <Th>시점</Th>
                  <Th>단계</Th>
                  <Th align="right">하한 – 상한</Th>
                  <Th align="right">확정 상주량</Th>
                  <Th>미상</Th>
                </tr>
              }
            >
              {device.timepoints.map((t) => (
                <tr key={t.timepoint}>
                  <Td>
                    <Mono className="text-[12px]">{t.timepoint}</Mono>
                  </Td>
                  <Td>{PHASE_LABEL[t.phase]}</Td>
                  <Td align="right">
                    <BytesRange low={t.bytes_low} high={t.bytes_high} reason="미상 항목 포함" />
                  </Td>
                  <Td align="right">
                    <Bytes value={t.known_floor_bytes} />
                  </Td>
                  <Td className="text-[12px] text-muted">{(t.unknown ?? []).join(", ")}</Td>
                </tr>
              ))}
            </DataTable>
          </div>
        </details>
      ) : null}
    </div>
  );
}
