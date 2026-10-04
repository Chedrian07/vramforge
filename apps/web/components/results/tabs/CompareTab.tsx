"use client";

import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/primitives";
import { Bytes, BytesRange } from "@/components/ui/values";
import type { AnalysisResult } from "@/lib/api/types";
import { exactBytes, formatCount, formatPercent, formatSize } from "@/lib/format/bytes";
import { HARDWARE_FIT_LABEL } from "@/lib/format/labels";
import type { ScenarioHistoryEntry } from "@/lib/hooks/useRecompute";
import { FIT_TONE } from "@/lib/result/status";
import { summarize } from "@/lib/result/summary";

import {
  AXIS_PROPS,
  ChartBox,
  ChartFigure,
  DataTable,
  GRID_STROKE,
  Td,
  Th,
  TooltipBox,
  gibTickLabel,
  gibTicks,
  type ChartTooltipProps,
} from "../chart-kit";

function signedSize(delta: number | null): string {
  if (delta == null) return "—";
  if (delta === 0) return "변화 없음";
  return `${delta > 0 ? "+" : "−"}${formatSize(Math.abs(delta))}`;
}

function ScenarioTooltip({ active, payload }: ChartTooltipProps) {
  const entry = payload?.[0];
  if (!active || !entry) return null;
  const row = entry.payload as { label: string; high: number };
  return (
    <TooltipBox value={formatSize(row.high)} label={`${row.label} · 예상 피크 상한`} detail={exactBytes(row.high)} />
  );
}

/** Highest peak of a result (primary scenario, or the largest budget scenario). */
function peakOf(result: AnalysisResult): number | null {
  const s = summarize(result);
  if (s.kind === "budgets") return s.budgetRange?.high ?? null;
  return s.peakHigh;
}

export function CompareTab({
  result,
  base,
  history,
}: {
  result: AnalysisResult | null;
  base: AnalysisResult | null;
  history: ScenarioHistoryEntry[];
}) {
  const scenarios = result?.memory?.scenarios ?? [];
  const rows = scenarios.map((s) => ({
    id: s.scenario_id,
    label: s.label,
    shape: s.batch_shape,
    low: s.devices[0]?.scenario_low_bytes ?? null,
    high: s.devices[0]?.scenario_high_bytes ?? null,
    recommended: s.recommendation?.recommended_application_capacity_bytes ?? null,
    fit: s.hardware_fit,
  }));
  const chartRows = rows.filter((r): r is typeof r & { high: number } => r.high != null);
  const unknownRows = rows.filter((r) => r.high == null);
  const first = rows[0]?.high ?? null;
  const chartHeight = Math.max(140, chartRows.length * 40 + 48);
  const axis = gibTicks(Math.max(0, ...chartRows.map((r) => r.high)));
  const basePeak = base ? peakOf(base) : null;

  return (
    <div className="flex flex-col gap-5">
      {rows.length > 1 ? (
        <ChartFigure
          title="시나리오별 예상 피크"
          description="같은 토큰화 결과에서 조건만 바꾼 시나리오입니다 (예: GRPO completion budget)."
          label={`시나리오별 예상 피크 상한: ${chartRows.map((r) => `${r.label} ${formatSize(r.high)}`).join(", ")}`}
          chart={
            <div className="flex flex-col gap-2">
              <ChartBox height={chartHeight}>
                {(width) => (
                  <BarChart
                    width={width}
                    height={chartHeight}
                    layout="vertical"
                    data={chartRows}
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
                    <YAxis type="category" dataKey="label" width={160} {...AXIS_PROPS} />
                    <Tooltip
                      content={ScenarioTooltip}
                      cursor={{ fill: "var(--vf-accent-soft)" }}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="high"
                      name="예상 피크 상한"
                      fill="var(--vf-accent)"
                      radius={[0, 4, 4, 0]}
                      isAnimationActive={false}
                    />
                  </BarChart>
                )}
              </ChartBox>
              {unknownRows.length > 0 ? (
                <p className="text-[12px] text-warn">
                  예상 피크를 산정하지 못한 시나리오 {unknownRows.length}개({unknownRows.map((r) => r.label).join(", ")})는 막대에
                  그리지 않았습니다. 사유는 표와 &lsquo;적용 설정·근거&rsquo; 탭에 있습니다.
                </p>
              ) : null}
            </div>
          }
          table={
            <DataTable
              caption="시나리오 비교"
              head={
                <tr>
                  <Th>시나리오</Th>
                  <Th align="right">padded 길이</Th>
                  <Th align="right">token slot</Th>
                  <Th align="right">예상 피크</Th>
                  <Th align="right">권장 용량</Th>
                  <Th align="right">첫 시나리오 대비</Th>
                  <Th>GPU 적합</Th>
                </tr>
              }
            >
              {rows.map((r) => (
                <tr key={r.id}>
                  <Td className="font-medium text-ink">{r.label}</Td>
                  <Td align="right" className="num">
                    {formatCount(r.shape.padded_length)}
                  </Td>
                  <Td align="right" className="num">
                    {formatCount(r.shape.token_slots)}
                  </Td>
                  <Td align="right">
                    <BytesRange low={r.low} high={r.high} reason="미상 항목 포함" />
                  </Td>
                  <Td align="right">
                    <Bytes value={r.recommended} />
                  </Td>
                  <Td align="right" className="num">
                    {r.high != null && first != null ? signedSize(r.high - first) : "—"}
                  </Td>
                  <Td>
                    <Badge tone={FIT_TONE[r.fit.status]}>
                      {HARDWARE_FIT_LABEL[r.fit.status]}
                      {r.fit.utilization_ratio != null ? ` · 예상 ${formatPercent(r.fit.utilization_ratio)}` : ""}
                    </Badge>
                  </Td>
                </tr>
              ))}
            </DataTable>
          }
        />
      ) : null}

      <section aria-label="재계산 비교" className="flex flex-col gap-2">
        <h3 className="text-[15px] font-semibold text-ink">재계산 비교</h3>
        {history.length === 0 ? (
          <p className="text-[13px] text-muted">
            Advanced 설정이나 하드웨어를 바꾸면 토큰화 결과를 재사용해 다시 계산하고, 기준 분석과의 차이를 여기에
            표시합니다.
          </p>
        ) : (
          <DataTable
            caption="기준 분석 대비 변경별 차이"
            head={
              <tr>
                <Th>변경</Th>
                <Th align="right">예상 피크 상한</Th>
                <Th align="right">기준 대비</Th>
              </tr>
            }
          >
            <tr>
              <Td className="text-muted">기준 분석</Td>
              <Td align="right">
                <Bytes value={basePeak} reason="산정 불가" />
              </Td>
              <Td align="right">—</Td>
            </tr>
            {history.map((entry) => {
              const peak = peakOf(entry.result);
              return (
                <tr key={entry.key}>
                  <Td>{entry.changes.length ? entry.changes.join(", ") : "변경 없음"}</Td>
                  <Td align="right">
                    <Bytes value={peak} reason="산정 불가" />
                  </Td>
                  <Td align="right" className="num">
                    {peak != null && basePeak != null ? signedSize(peak - basePeak) : "—"}
                  </Td>
                </tr>
              );
            })}
          </DataTable>
        )}
      </section>
    </div>
  );
}
