"use client";

import { useFormContext, useWatch } from "react-hook-form";

import { Badge, Button, Field, Mono, TextInput, describedBy } from "@/components/ui/primitives";
import { CopyButton } from "@/components/ui/controls";
import { IssueList } from "@/components/ui/values";
import type { ModelInspection } from "@/lib/api/types";
import { formatCount, shortDigest } from "@/lib/format/bytes";
import { EVIDENCE_LEVEL_LABEL, OBJECTIVE_LABEL, READINESS_LABEL, STRATEGY_LABEL } from "@/lib/format/labels";
import { modelInspectCall, referenceOfKey } from "@/lib/form/inspect";
import type { FormValues } from "@/lib/form/values";
import type { useModelInspection } from "@/lib/hooks/useInspection";
import { READINESS_TONE } from "@/lib/result/status";

type Inspection = ReturnType<typeof useModelInspection>;

export function ModelSection({ inspection }: { inspection: Inspection }) {
  const { register, getValues, formState } = useFormContext<FormValues>();
  const reference = useWatch<FormValues, "modelReference">({ name: "modelReference" });
  const error = formState.errors.modelReference?.message;
  const id = "model-reference";
  const hint = "Hugging Face ID·URL 또는 서버에 등록된 local: 경로. 입력을 마치거나 확인을 누르면 메타데이터만 조회합니다.";

  const trigger = (force: boolean) => {
    const call = modelInspectCall(getValues());
    if (!call) return;
    void (force ? inspection.reinspect(call.key, call.body) : inspection.inspect(call.key, call.body));
  };

  const field = register("modelReference", { onBlur: () => trigger(false) });
  const { state } = inspection;
  const inspectedRef = referenceOfKey(state.key);
  const outdated = inspectedRef != null && inspectedRef !== reference.trim();

  return (
    <div className="flex flex-col gap-3">
      <Field label="Model" htmlFor={id} hint={hint} error={error}>
        <div className="flex gap-2">
          <TextInput
            id={id}
            {...field}
            placeholder="org/model · https://huggingface.co/org/model · local:models/…"
            className="font-mono text-[14px]"
            autoComplete="off"
            spellCheck={false}
            aria-invalid={error ? true : undefined}
            aria-describedby={describedBy(id, { hint, error })}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                trigger(true);
              }
            }}
          />
          <Button onClick={() => trigger(true)} aria-label="모델 확인">
            확인
          </Button>
        </div>
      </Field>
      <div aria-live="polite" className="min-w-0">
        {state.status === "loading" ? <p className="text-[13px] text-muted">모델 구조와 tokenizer를 확인하는 중…</p> : null}
        {state.status === "error" && state.error ? <IssueList issues={[state.error.issue]} /> : null}
        {state.status === "done" && state.data ? (
          <ModelInspectionView data={state.data} outdated={outdated} />
        ) : null}
      </div>
    </div>
  );
}

function ModelInspectionView({ data, outdated }: { data: ModelInspection; outdated: boolean }) {
  const facts = data.summary?.facts ?? null;
  const tokenizer = data.tokenizer ?? null;
  // Watched (not read once) so the "(선택)" mark follows Method and strategy changes.
  const objective = useWatch<FormValues, "objective">({ name: "objective" });
  const strategy = useWatch<FormValues, "strategy">({ name: "strategy" });
  const layerCounts = facts?.layer_type_counts ?? [];
  const support = data.support ?? [];
  return (
    <div className={`flex flex-col gap-3 rounded-lg border border-line bg-surface-2 p-3 ${outdated ? "opacity-60" : ""}`}>
      {outdated ? (
        <p className="text-[12px] text-warn">입력한 모델이 바뀌었습니다. 확인을 눌러 다시 조회하세요.</p>
      ) : null}
      <dl className="grid grid-cols-1 gap-x-4 gap-y-2 text-[13px] sm:grid-cols-[auto_minmax(0,1fr)]">
        <dt className="text-muted">구조</dt>
        <dd className="min-w-0">
          {facts ? (
            <>
              <Mono>{facts.architectures[0] ?? facts.model_type}</Mono>
              <span className="text-muted"> · {facts.model_type}</span>
            </>
          ) : (
            <span className="text-warn">구조 정보를 확인하지 못했습니다</span>
          )}
        </dd>
        <dt className="text-muted">Architecture adapter</dt>
        <dd className="min-w-0">
          {data.architecture_adapter ? (
            <Mono>{data.architecture_adapter}</Mono>
          ) : (
            <Badge tone="err">등록된 adapter 없음 (메모리 산정 미지원)</Badge>
          )}
        </dd>
        {facts ? (
          <>
            <dt className="text-muted">레이어</dt>
            <dd className="num">
              {facts.num_hidden_layers}층
              {layerCounts.length > 0 ? ` · ${layerCounts.map((c) => `${c.layer_type} ${c.count}`).join(" / ")}` : ""}
              {` · hidden ${formatCount(facts.hidden_size)} · vocab ${formatCount(facts.vocab_size)}`}
            </dd>
          </>
        ) : null}
        {data.summary ? (
          <>
            <dt className="text-muted">파라미터</dt>
            <dd className="num">{formatCount(data.summary.params_total)}</dd>
          </>
        ) : null}
        {data.manifest ? (
          <>
            <dt className="text-muted">Revision</dt>
            <dd className="flex min-w-0 items-center gap-2">
              <Mono>{shortDigest(data.manifest.resolved_revision, 12)}</Mono>
              <CopyButton value={data.manifest.resolved_revision} />
            </dd>
          </>
        ) : null}
        <dt className="text-muted">Tokenizer</dt>
        <dd className="min-w-0">
          {tokenizer ? (
            <span className="flex flex-wrap items-center gap-2">
              <Mono>{tokenizer.tokenizer_class}</Mono>
              {tokenizer.chat_template_present ? (
                <Badge tone="ok">chat template · {tokenizer.chat_template_source}</Badge>
              ) : (
                <Badge tone="warn">chat template 없음</Badge>
              )}
              {tokenizer.chat_template_sha256 ? <Mono className="text-muted">{shortDigest(tokenizer.chat_template_sha256, 10)}</Mono> : null}
              {tokenizer.template_parse_error ? <Badge tone="warn">template 해석 오류</Badge> : null}
            </span>
          ) : (
            <Badge tone="err">tokenizer 없음: 데이터 길이를 계산할 수 없습니다</Badge>
          )}
        </dd>
      </dl>
      {tokenizer?.template_parse_error ? (
        <p className="text-[12px] leading-snug text-warn">chat template을 해석하지 못했습니다: {tokenizer.template_parse_error}</p>
      ) : null}
      {facts?.has_vision ? (
        <p className="text-[12px] leading-snug text-ink-2">
          <Badge tone="info">vision tower</Badge>{" "}
          비전 모듈이 포함된 checkpoint입니다. 텍스트 데이터만 학습해도 loader가 비전 모듈을 로딩하면 frozen 가중치로 VRAM에 상주합니다.
        </p>
      ) : null}
      {support.length > 0 ? (
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-muted">지원 조합</p>
          <ul className="flex flex-wrap gap-1.5">
            {support.map((entry) => {
              const selected = entry.objective === objective && entry.strategy === strategy;
              return (
                <li key={`${entry.objective}-${entry.strategy}`}>
                  <Badge tone={READINESS_TONE[entry.readiness]} className={selected ? "ring-2 ring-accent/40" : undefined} title={entry.note || undefined}>
                    {OBJECTIVE_LABEL[entry.objective]} · {STRATEGY_LABEL[entry.strategy]} —{" "}
                    {entry.grade ? EVIDENCE_LEVEL_LABEL[entry.grade] : "미지원"} · {READINESS_LABEL[entry.readiness]}
                    {selected ? " (선택)" : ""}
                  </Badge>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
      <IssueList issues={data.issues ?? []} />
    </div>
  );
}
