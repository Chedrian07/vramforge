"use client";

import type { ReactNode } from "react";

import { CopyButton } from "@/components/ui/controls";
import { Badge, Ident, Mono } from "@/components/ui/primitives";
import { IssueList } from "@/components/ui/values";
import type { AnalysisResult, ScenarioEstimate } from "@/lib/api/types";
import { shortDigest } from "@/lib/format/bytes";
import { EVIDENCE_LABEL, EVIDENCE_LEVEL_LABEL, PHASE_LABEL, READINESS_LABEL } from "@/lib/format/labels";
import { READINESS_TONE } from "@/lib/result/status";

import { DataTable, Td, Th } from "../chart-kit";

function show(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "켬" : "끔";
  if (Array.isArray(value)) return value.length ? value.map((v) => show(v)).join(", ") : "없음";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section aria-label={title} className="flex flex-col gap-2">
      <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
      {children}
    </section>
  );
}

function KeyValue({ label, value, copy }: { label: string; value: string | null | undefined; copy?: boolean }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 sm:flex-row sm:items-center sm:gap-2">
      <dt className="shrink-0 text-[12px] text-muted sm:w-40">{label}</dt>
      <dd className="flex min-w-0 items-center gap-2">
        <Mono className="text-ink">{value ?? "—"}</Mono>
        {copy && value ? <CopyButton value={value} /> : null}
      </dd>
    </div>
  );
}

export function EvidenceTab({ result, scenario }: { result: AnalysisResult | null; scenario: ScenarioEstimate | null }) {
  if (!result) return <p className="text-[13px] text-muted">분석 후 적용 설정과 근거를 표시합니다.</p>;
  const req = result.requested_config;
  const res = result.resolved_config ?? null;
  const observed = result.observed_config ?? null;
  const t = req.training;
  const observedCell = (key: string) => (observed ? show(observed[key]) : "미검증");

  const comparison: Array<{ label: string; requested: unknown; resolved: unknown; key: string }> = [
    { key: "objective", label: "학습 방식", requested: t.objective, resolved: res?.objective },
    { key: "strategy", label: "전략", requested: t.strategy, resolved: res?.strategy },
    { key: "quantization", label: "4-bit 로딩", requested: t.quantization?.enabled, resolved: res ? (res.quantization.enabled ? `${res.quantization.method ?? "4-bit"}${res.quantization.double_quant ? " + double quant" : ""}` : false) : undefined },
    { key: "load_dtype", label: "load dtype", requested: t.load_dtype, resolved: res?.load_dtype },
    { key: "loading_scope", label: "loading scope", requested: req.model.loading_scope, resolved: res?.loading_scope },
    { key: "lora", label: "LoRA r / alpha / dropout", requested: t.lora ? `${t.lora.r} / ${t.lora.alpha} / ${t.lora.dropout}` : undefined, resolved: res?.lora ? `${res.lora.r} / ${res.lora.alpha} / ${res.lora.dropout}` : res ? "사용 안 함" : undefined },
    { key: "target_modules", label: "LoRA 대상", requested: t.lora?.target_modules, resolved: res?.lora ? `${res.lora.target_module_patterns.join(", ")} (모듈 ${res.lora.target_modules.length}개)` : undefined },
    { key: "optimizer", label: "optimizer", requested: t.optimizer, resolved: res ? `${res.optimizer.name} (state ${res.optimizer.state_dtype})` : undefined },
    { key: "microbatch", label: "microbatch", requested: t.microbatch_per_device ?? "자동", resolved: res?.microbatch },
    { key: "accumulation", label: "gradient accumulation", requested: t.gradient_accumulation_steps ?? "자동", resolved: res?.accumulation },
    { key: "gradient_checkpointing", label: "gradient checkpointing", requested: t.gradient_checkpointing, resolved: res ? `${show(res.gradient_checkpointing)} (${res.checkpointing_granularity})` : undefined },
    { key: "attention", label: "attention 경로", requested: `${t.attention_backend ?? "auto"} / linear ${t.linear_attention_kernel ?? "auto"}`, resolved: res ? Object.entries(res.attention_path_by_layer_type ?? {}).map(([k, v]) => `${k}: ${v}`).join(", ") : undefined },
    { key: "loss", label: "loss 경로", requested: t.loss_kernel, resolved: res?.loss_path },
    { key: "mixed_precision", label: "mixed precision", requested: t.precision, resolved: res?.mixed_precision },
    { key: "packing", label: "packing", requested: t.packing, resolved: res?.packing },
    {
      key: "loss_mask",
      label: "loss 대상 token",
      requested: undefined,
      resolved: res
        ? `assistant_only_loss ${show(res.assistant_only_loss)} · completion_only_loss ${res.completion_only_loss == null ? "기본값(None)" : show(res.completion_only_loss)}`
        : undefined,
    },
  ];
  if (t.objective === "dpo") {
    comparison.push({
      key: "dpo",
      label: "DPO reference / beta",
      requested: `${req.dpo?.reference_strategy ?? "auto"}${req.dpo?.reference_model ? ` (${req.dpo.reference_model})` : ""} / ${req.dpo?.beta ?? "—"}`,
      resolved: res?.dpo ? `${res.dpo.reference_strategy}${res.dpo.reference_model ? ` (${res.dpo.reference_model})` : ""} / ${res.dpo.beta}` : undefined,
    });
  }
  if (t.objective === "grpo") {
    comparison.push({
      key: "grpo",
      label: "GRPO G / gbs / spg / budget",
      requested: `${req.grpo?.num_generations ?? "—"} / ${req.grpo?.generation_batch_size ?? "자동"} / ${req.grpo?.steps_per_generation ?? "자동"} / ${req.grpo?.completion_budget ?? (req.grpo?.completion_budget_candidates ?? []).join(", ")}`,
      resolved: res?.grpo ? `${res.grpo.num_generations} / ${res.grpo.generation_batch_size} / ${res.grpo.steps_per_generation} / ${res.grpo.completion_budgets.join(", ")}` : undefined,
    });
    comparison.push({
      key: "reward",
      label: "reward / 학습 GPU 상주",
      requested: `${req.grpo?.reward?.kind ?? "unspecified"} / ${show(req.grpo?.reward?.on_training_gpu ?? true)}`,
      resolved: res?.grpo ? `${res.grpo.reward_kind} / ${show(res.grpo.reward_on_training_gpu)}` : undefined,
    });
  }

  const assumptions = [...(result.assumptions ?? []), ...(result.memory?.assumptions ?? [])].filter(
    (a, i, all) => all.findIndex((b) => b.id === a.id) === i,
  );
  const allocations = scenario?.allocations ?? [];
  const compat = result.compatibility_report ?? null;
  const manifests = result.source_manifests;

  return (
    <div className="flex flex-col gap-6">
      <Section title="프로필과 버전">
        <dl className="flex flex-col gap-1.5">
          <KeyValue label="profile" value={res?.profile_id ?? result.profile_id} />
          <KeyValue label="profile version" value={res?.profile_version} />
          <KeyValue label="environment" value={res?.environment_id} />
          <KeyValue label="dependency lock" value={result.dependency_lock_digest ?? res?.dependency_lock_digest} copy />
          <KeyValue label="estimator" value={`${result.estimator_version} · schema ${result.schema_version ?? "1.0"}`} />
          <KeyValue label="adapter" value={res ? `${res.architecture_adapter ?? "—"} · ${res.trainer_adapter ?? "—"} · ${res.preprocessing_adapter ?? "—"}` : null} />
          <KeyValue label="analysis fingerprint" value={result.analysis_fingerprint} copy />
        </dl>
      </Section>

      <Section title="소스 revision">
        <dl className="flex flex-col gap-1.5">
          <KeyValue label="모델" value={manifests?.model ? `${manifests.model.reference} @ ${manifests.model.resolved_revision}` : null} copy />
          <KeyValue label="데이터셋" value={manifests?.dataset ? `${manifests.dataset.reference} @ ${manifests.dataset.resolved_revision}` : null} copy />
          <KeyValue
            label="tokenizer"
            value={result.tokenizer_manifest ? `${result.tokenizer_manifest.tokenizer_class} · template ${shortDigest(result.tokenizer_manifest.chat_template_sha256, 12) ?? "없음"}` : null}
          />
        </dl>
        {result.tokenizer_manifest?.template_parse_error ? (
          <IssueList
            issues={[{ code: "", severity: "warning", user_message: `chat template을 해석하지 못했습니다: ${result.tokenizer_manifest.template_parse_error}` }]}
          />
        ) : null}
      </Section>

      <Section title="요청 · 적용 · 관측 설정">
        <p className="text-[12px] text-muted">관측(observed)은 GPU 검증에서 실제 확인한 실행 경로입니다. GPU 검증이 연결되지 않아 모두 미검증입니다.</p>
        <DataTable caption="requested / resolved / observed" captionHidden head={<tr><Th>항목</Th><Th>요청</Th><Th>적용</Th><Th>관측</Th></tr>}>
          {comparison.map((row) => (
            <tr key={row.key}>
              <Td className="font-medium text-ink">{row.label}</Td>
              <Td><Ident className="text-[12px]">{row.requested === undefined ? "—" : show(row.requested)}</Ident></Td>
              <Td><Ident className="text-[12px]">{res ? show(row.resolved) : "확정 전"}</Ident></Td>
              <Td className="text-[12px] text-muted">{observedCell(row.key)}</Td>
            </tr>
          ))}
        </DataTable>
        {res ? (
          <p className="text-[12px] text-ink-2">
            실효 dtype: compute <Mono>{res.effective_dtypes.compute}</Mono> · adapter <Mono>{res.effective_dtypes.adapter}</Mono> · gradient <Mono>{res.effective_dtypes.gradient}</Mono> · optimizer state <Mono>{res.effective_dtypes.optimizer_state}</Mono> · logits <Mono>{res.effective_dtypes.logits}</Mono> · KV <Mono>{res.effective_dtypes.kv_cache}</Mono> · recurrent <Mono>{res.effective_dtypes.recurrent_state}</Mono>
            {res.effective_dtypes.conv_state ? (
              <>
                {" "}
                · conv state <Mono>{res.effective_dtypes.conv_state}</Mono>
              </>
            ) : null}
          </p>
        ) : null}
      </Section>

      {res && (res.resolutions ?? []).length > 0 ? (
        <Section title="설정 해석 근거">
          <DataTable caption="요청과 다르게 해석한 설정" captionHidden head={<tr><Th>필드</Th><Th>요청</Th><Th>적용</Th><Th>이유</Th></tr>}>
            {(res.resolutions ?? []).map((r) => (
              <tr key={r.field}>
                <Td><Ident className="text-[12px]">{r.field}</Ident></Td>
                <Td><Ident className="text-[12px]">{show(r.requested)}</Ident></Td>
                <Td><Ident className="text-[12px]">{show(r.resolved)}</Ident></Td>
                <Td prose>{r.reason}</Td>
              </tr>
            ))}
          </DataTable>
        </Section>
      ) : null}

      {compat ? (
        <Section title="호환성">
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <span className="text-muted">지원 등급</span>
            <Badge tone={compat.support_grade ? "info" : "err"}>{compat.support_grade ? EVIDENCE_LEVEL_LABEL[compat.support_grade] : "미지원 조합"}</Badge>
            <span className="text-muted">학습 준비</span>
            <Badge tone={READINESS_TONE[compat.readiness]}>{READINESS_LABEL[compat.readiness]}</Badge>
          </div>
          <IssueList issues={[...(compat.blockers ?? []), ...(compat.warnings ?? [])]} />
          {(compat.not_effective ?? []).length > 0 ? (
            <DataTable caption="요청했지만 적용되지 않은 옵션" head={<tr><Th>필드</Th><Th>요청</Th><Th>적용</Th><Th>이유</Th></tr>}>
              {(compat.not_effective ?? []).map((r) => (
                <tr key={r.field}>
                  <Td><Ident className="text-[12px]">{r.field}</Ident></Td>
                  <Td><Ident className="text-[12px]">{show(r.requested)}</Ident></Td>
                  <Td><Ident className="text-[12px]">{show(r.resolved)}</Ident></Td>
                  <Td prose>{r.reason}</Td>
                </tr>
              ))}
            </DataTable>
          ) : null}
        </Section>
      ) : null}

      <Section title="가정">
        {assumptions.length === 0 ? (
          <p className="text-[13px] text-muted">기록된 가정이 없습니다.</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {assumptions.map((a) => (
              <li key={a.id} className="flex flex-col gap-0.5 text-[13px] sm:flex-row sm:items-start sm:gap-2">
                <Badge className="self-start" tone="neutral">{EVIDENCE_LABEL[a.evidence ?? "assumption"]}</Badge>
                <span className="text-ink">{a.text}</span>
                {a.source ? <Mono className="text-[11px] text-muted">{a.source}</Mono> : null}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="수식과 shape">
        {allocations.length === 0 ? (
          <p className="text-[13px] text-muted">이 결과에는 allocation별 수식 정보가 포함되지 않았습니다. 피크 시점 항목은 &lsquo;메모리 구성&rsquo; 탭에 있습니다.</p>
        ) : (
          <DataTable caption="allocation 수식" captionHidden head={<tr><Th>allocation</Th><Th>shape</Th><Th>dtype</Th><Th>수식 근거</Th><Th>근거</Th></tr>}>
            {allocations.map((a, i) => (
              <tr key={`${a.name}-${i}`}>
                <Td><Ident className="text-[12px]">{a.name}</Ident></Td>
                <Td><Ident className="text-[12px]">{a.shape_expression ?? "—"}</Ident></Td>
                <Td><Ident className="text-[12px]">{a.dtype ?? "—"}</Ident></Td>
                <Td><Ident className="text-[12px]">{a.formula_ref ?? "—"}</Ident></Td>
                <Td>{EVIDENCE_LABEL[a.evidence]}</Td>
              </tr>
            ))}
          </DataTable>
        )}
      </Section>

      <Section title="미확정 · 제외 항목">
        <IssueList
          issues={[
            ...(result.unknown_components ?? []).map((u) => ({ code: "UNKNOWN_MEMORY_COMPONENT", severity: "warning" as const, user_message: `산정 불가: ${u.name}${u.phase ? ` (${u.phase})` : ""} — ${u.reason}` })),
            ...(result.excluded_components ?? []).map((e) => ({ code: e.code ?? "", severity: "info" as const, user_message: `제외: ${e.name} — ${e.reason}` })),
          ]}
          empty="미확정이거나 범위에서 제외한 항목이 없습니다."
        />
      </Section>

      <Section title="측정 범위">
        <p className="text-[13px] text-ink-2">
          {result.measurement_scope?.measured ? "GPU에서 측정한 단계가 있습니다." : "GPU 실측이 없는 정적 추정입니다."} {result.measurement_scope?.note}
        </p>
        <p className="text-[12px] text-muted">
          포함: {(result.measurement_scope?.phases_included ?? []).map((p) => PHASE_LABEL[p]).join(", ") || "—"} · 제외: {(result.measurement_scope?.phases_excluded ?? []).map((p) => PHASE_LABEL[p]).join(", ") || "—"}
        </p>
      </Section>

      <Section title="경고와 오류">
        <IssueList issues={[...(result.errors ?? []), ...(result.warnings ?? []), ...(result.memory?.issues ?? [])]} empty="경고나 오류가 없습니다." />
      </Section>
    </div>
  );
}
