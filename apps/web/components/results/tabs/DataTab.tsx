"use client";

import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Mono } from "@/components/ui/primitives";
import type { AnalysisResult, BranchStats } from "@/lib/api/types";
import { formatCount, formatNumber, shortDigest } from "@/lib/format/bytes";
import { BRANCH_LABEL, DATA_PRESERVATION_LABEL, PRESERVATION_CHECK_LABEL } from "@/lib/format/labels";
import { exceededRowsText, scanCoverageDisplay } from "@/lib/result/scan";
import { PRESERVATION_TONE } from "@/lib/result/status";

import {
  AXIS_PROPS,
  ChartBox,
  ChartFigure,
  DataTable,
  GRID_STROKE,
  Td,
  Th,
  TooltipBox,
  type ChartTooltipProps,
} from "../chart-kit";

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col">
      <dt className="text-[12px] text-muted">{label}</dt>
      <dd className="num text-[14px] text-ink">{value ?? "—"}</dd>
    </div>
  );
}

function HistogramTooltip({ active, payload }: ChartTooltipProps) {
  const entry = payload?.[0];
  if (!active || !entry) return null;
  const bin = entry.payload as { lo: number; hi: number; count: number };
  return (
    <TooltipBox
      value={`${formatCount(bin.count)} rows`}
      label={`길이 ${formatCount(bin.lo)}–${formatCount(bin.hi - 1)} token`}
    />
  );
}

function BranchSection({ branch }: { branch: BranchStats }) {
  const s = branch.stats;
  const bins = (s.histogram ?? []).map((b) => ({ ...b, range: `${b.lo}–${b.hi - 1}` }));
  const label = BRANCH_LABEL[branch.branch];
  return (
    <section aria-label={`${label} 길이`} className="flex flex-col gap-3 rounded-lg border border-line p-3 sm:p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[15px] font-semibold text-ink">{label}</h3>
        <Mono className="text-muted">{branch.branch}</Mono>
        {s.quantiles_exact === false ? <Badge tone="warn">근사 분위수</Badge> : null}
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-5">
        <Stat label="row 수" value={formatCount(s.count)} />
        <Stat label="최소" value={formatCount(s.min)} />
        <Stat
          label="최대"
          value={
            <>
              {formatCount(s.max)}
              {s.max_row_id ? <Mono className="ml-1 text-[11px] text-muted">({s.max_row_id})</Mono> : null}
            </>
          }
        />
        <Stat label="평균" value={formatNumber(s.mean, 1)} />
        <Stat label="총 token" value={formatCount(s.total_tokens)} />
        <Stat label="P50" value={formatCount(s.p50)} />
        <Stat label="P90" value={formatCount(s.p90)} />
        <Stat label="P95" value={formatCount(s.p95)} />
        <Stat label="P99" value={formatCount(s.p99)} />
      </dl>
      {bins.length > 0 ? (
        <ChartFigure
          title={`${label} 길이 분포`}
          description="token 길이 구간별 row 수. 메모리 기준은 분위수가 아니라 최악 batch shape입니다."
          label={`${label} 길이 히스토그램, 구간 ${bins.length}개, 최대 ${formatCount(s.max)} token`}
          chart={
            <ChartBox height={200}>
              {(width) => (
                <BarChart width={width} height={200} data={bins} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid vertical={false} stroke={GRID_STROKE} />
                  <XAxis dataKey="range" interval="preserveStartEnd" {...AXIS_PROPS} />
                  <YAxis
                    allowDecimals={false}
                    width={44}
                    tickFormatter={(v: number) => formatCount(v) ?? ""}
                    {...AXIS_PROPS}
                  />
                  <Tooltip
                    content={HistogramTooltip}
                    cursor={{ fill: "var(--vf-accent-soft)" }}
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="count"
                    name="row 수"
                    fill="var(--vf-accent)"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={24}
                    isAnimationActive={false}
                  />
                </BarChart>
              )}
            </ChartBox>
          }
          table={
            <details>
              <summary className="cursor-pointer text-[12px] text-ink-2 select-none">구간별 표 보기</summary>
              <div className="mt-2">
                <DataTable
                  caption={`${label} 길이 구간별 row 수`}
                  captionHidden
                  head={
                    <tr>
                      <Th>길이 구간 (token)</Th>
                      <Th align="right">row 수</Th>
                    </tr>
                  }
                >
                  {bins.map((b) => (
                    <tr key={b.lo}>
                      <Td className="num">
                        {formatCount(b.lo)}–{formatCount(b.hi - 1)}
                      </Td>
                      <Td align="right" className="num">
                        {formatCount(b.count)}
                      </Td>
                    </tr>
                  ))}
                </DataTable>
              </div>
            </details>
          }
        />
      ) : null}
      {(branch.top_rows ?? []).length > 0 ? (
        <DataTable
          caption="가장 긴 row (ID와 길이만 표시)"
          head={
            <tr>
              <Th>row id</Th>
              <Th align="right">길이 (token)</Th>
            </tr>
          }
        >
          {(branch.top_rows ?? []).map((row) => (
            <tr key={row.row_id}>
              <Td>
                <Mono>{row.row_id}</Mono>
              </Td>
              <Td align="right" className="num">
                {formatCount(row.length)}
              </Td>
            </tr>
          ))}
        </DataTable>
      ) : null}
    </section>
  );
}

export function DataTab({ result, partial }: { result: AnalysisResult | null; partial: boolean }) {
  const scan = result?.dataset_scan ?? null;
  if (!scan) return <p className="text-[13px] text-muted">데이터 스캔 결과가 아직 없습니다.</p>;
  const mapping = scan.mapping_applied;
  const audit = result?.preservation_audit ?? null;
  const context = result?.context_validation ?? null;
  const coverage = scanCoverageDisplay(scan.coverage, { rowsSeen: scan.rows_seen, rowsFailed: scan.rows_failed });
  return (
    <div className="flex flex-col gap-5">
      {partial || scan.coverage !== "complete" ? (
        <p className="rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">
          현재까지 확인한 데이터 기준입니다. 전체 최대 길이로 간주하지 마세요.
        </p>
      ) : null}
      {scan.rows_failed > 0 ? (
        <p className="rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">
          읽거나 토큰화하지 못한 row {formatCount(scan.rows_failed)}개는 길이 통계에 없습니다. 아래 실패 row 표본에서 사유를 확인하세요.
        </p>
      ) : null}
      <section aria-label="스캔 범위" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-[15px] font-semibold text-ink">스캔 범위</h3>
          <Badge tone={coverage.tone}>{coverage.text}</Badge>
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
          <Stat label="예상 row" value={formatCount(scan.rows_expected) ?? "미확인"} />
          <Stat label="읽은 row" value={formatCount(scan.rows_seen)} />
          <Stat label="성공 row" value={formatCount(scan.rows_ok)} />
          <Stat label="실패 row" value={formatCount(scan.rows_failed)} />
          <Stat
            label="미처리 row"
            value={scan.rows_unprocessed == null ? "미확인" : formatCount(scan.rows_unprocessed)}
          />
          <Stat
            label="shard"
            value={`${formatCount(scan.shards_completed)} / ${scan.shards_total == null ? "?" : formatCount(scan.shards_total)}`}
          />
          <Stat label="중복 row" value={scan.duplicate_rows == null ? "미확인" : formatCount(scan.duplicate_rows)} />
          <Stat label="context 초과 row" value={exceededRowsText(scan.context_exceeded_rows)} />
          {scan.omitted_system_messages != null ? (
            <Stat label="생략한 빈 system 메시지" value={formatCount(scan.omitted_system_messages)} />
          ) : null}
        </dl>
        <p className="text-[13px] text-ink-2">{scan.transformation_note}</p>
        <p className="text-[12px] text-muted">
          split <Mono>{scan.split ?? "—"}</Mono>
          {scan.split_auto_selected ? " (자동 선택)" : ""} · config <Mono>{scan.config ?? "—"}</Mono> · 전처리{" "}
          <Mono>
            {scan.preprocessing_adapter}@{scan.preprocessing_adapter_version}
          </Mono>{" "}
          · tokenizer <Mono>{shortDigest(scan.tokenizer_fingerprint, 12)}</Mono>
          {scan.template_fingerprint ? (
            <>
              {" "}
              · template <Mono>{shortDigest(scan.template_fingerprint, 12)}</Mono>
            </>
          ) : null}
        </p>
        {mapping ? (
          <p className="text-[12px] text-ink-2">
            적용 매핑:{" "}
            {(["system", "prompt", "chosen", "rejected", "completion", "messages", "text"] as const)
              .filter((role) => mapping[role])
              .map((role) => `${mapping[role]} → ${role}`)
              .join(", ")}{" "}
            · 빈 system {mapping.empty_system_policy === "omit" ? "생략" : "유지"}
          </p>
        ) : null}
        <p className="text-[12px] text-muted">원문은 기본으로 표시하지 않습니다. 긴 row는 ID와 길이만 보여 줍니다.</p>
      </section>

      {(scan.branches ?? []).map((branch) => (
        <BranchSection key={branch.branch} branch={branch} />
      ))}

      {(scan.failed_rows_sample ?? []).length > 0 ? (
        <DataTable
          caption={`실패 row (표본, 전체 ${formatCount(scan.rows_failed)}건)`}
          head={
            <tr>
              <Th>row id</Th>
              <Th>shard</Th>
              <Th>코드</Th>
              <Th>사유</Th>
            </tr>
          }
        >
          {(scan.failed_rows_sample ?? []).map((row) => (
            <tr key={row.row_id}>
              <Td>
                <Mono>{row.row_id}</Mono>
              </Td>
              <Td>
                <Mono>{row.shard_id ?? "—"}</Mono>
              </Td>
              <Td>
                <Mono>{row.error_code}</Mono>
              </Td>
              <Td>{row.message}</Td>
            </tr>
          ))}
        </DataTable>
      ) : null}

      {context ? (
        <section aria-label="context 검증" className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[15px] font-semibold text-ink">context 검증</h3>
            <Badge tone={context.status === "ok" ? "ok" : context.status === "exceeded" ? "err" : "warn"}>
              {context.status === "ok"
                ? "상한 이내"
                : context.status === "exceeded"
                  ? "상한 초과 (자동 절단 없음)"
                  : "미확인"}
            </Badge>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
            <Stat label="모델 선언 상한" value={formatCount(context.model_declared_max) ?? "미확인"} />
            <Stat
              label="tokenizer 상한"
              value={
                context.tokenizer_limit_is_sentinel
                  ? "sentinel (무시)"
                  : (formatCount(context.tokenizer_model_max_length) ?? "미확인")
              }
            />
            <Stat label="backend 검증 상한" value={formatCount(context.backend_verified_max) ?? "미검증"} />
            <Stat label="적용 상한" value={formatCount(context.effective_limit) ?? "미확인"} />
            <Stat label="관측 최대" value={formatCount(context.max_observed_length)} />
            <Stat label="초과 row" value={exceededRowsText(context.exceeded_rows, context.exceeded_rows_exact !== false)} />
          </dl>
          {context.limit_source ? (
            <p className="text-[12px] text-muted">
              상한 근거: <Mono>{context.limit_source}</Mono>
            </p>
          ) : null}
        </section>
      ) : null}

      {audit ? (
        <section aria-label="데이터 보존 검사" className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[15px] font-semibold text-ink">데이터 보존 검사</h3>
            <Badge tone={PRESERVATION_TONE[audit.status]}>{DATA_PRESERVATION_LABEL[audit.status]}</Badge>
          </div>
          <ul className="flex flex-col gap-1.5">
            {(audit.checks ?? []).map((check) => (
              <li key={check.name} className="flex flex-col gap-0.5 text-[13px] sm:flex-row sm:items-start sm:gap-2">
                <Badge className="self-start" tone={check.passed === true ? "ok" : check.passed === false ? "err" : "neutral"}>
                  {check.passed === true ? "통과" : check.passed === false ? "위반" : "해당 없음"}
                </Badge>
                <span className="font-medium text-ink">{PRESERVATION_CHECK_LABEL[check.name]}</span>
                <span className="text-muted">{check.detail}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
