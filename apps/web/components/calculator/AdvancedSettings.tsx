"use client";

import { Accordion } from "radix-ui";
import type { ReactNode } from "react";
import { useFormContext, useWatch, type FieldErrors, type FieldPath } from "react-hook-form";

import { Segmented } from "@/components/ui/controls";
import { Field, Mono, NativeSelect } from "@/components/ui/primitives";
import type { DatasetInspection, ResolvedConfig } from "@/lib/api/types";
import {
  ATTENTION_BACKEND_LABEL,
  COMPUTE_DTYPE_LABEL,
  EMPTY_SYSTEM_POLICY_LABEL,
  LINEAR_ATTENTION_KERNEL_LABEL,
  LOADING_SCOPE_LABEL,
  LOAD_DTYPE_LABEL,
  LORA_BIAS_LABEL,
  LOSS_KERNEL_LABEL,
  MICROBATCH_UNIT,
  OBJECTIVE_LABEL,
  OPTIMIZER_LABEL,
  PRECISION_LABEL,
  QUANT_FORMAT_LABEL,
  REFERENCE_STRATEGY_LABEL,
  REWARD_KIND_LABEL,
  ROLLOUT_BACKEND_LABEL,
} from "@/lib/format/labels";
import { mappedColumns } from "@/lib/form/mapping";
import type { FormValues } from "@/lib/form/values";
import { useBackendProfiles } from "@/lib/hooks/useInspection";

import { ListField, SelectField, SwitchControl, TextField, UnsupportedSwitch, enumOptions } from "./advanced-fields";
import { resetConfigSelections } from "./dataset-selection";

export const ADVANCED_GROUPS = ["adapter", "batch", "runtime", "method", "dataset"] as const;
export type AdvancedGroup = (typeof ADVANCED_GROUPS)[number];

/** Form fields edited inside each group (a closed group unmounts its fields). */
const GROUP_FIELDS: Record<AdvancedGroup, readonly FieldPath<FormValues>[]> = {
  adapter: ["loraR", "loraAlpha", "loraDropout", "loraTargetMode", "loraTargetModules", "loraExcludeModules", "loraModulesToSave", "loraBias", "loraRankPattern"],
  batch: ["microbatch", "accumulation", "padToMultipleOf", "precision", "loadDtype", "optimizer", "quantFormat", "computeDtype"],
  runtime: ["attentionBackend", "linearAttentionKernel", "lossKernel", "backendProfile"],
  method: [
    "dpoReferenceStrategy",
    "dpoReferenceModel",
    "dpoBeta",
    "dpoLossType",
    "dpoPrecomputeBatchSize",
    "grpoNumGenerations",
    "grpoGenerationBatchSize",
    "grpoStepsPerGeneration",
    "grpoNumIterations",
    "grpoCompletionBudget",
    "grpoBudgetCandidates",
    "grpoBeta",
    "grpoRewardKind",
    "grpoRewardModelReference",
  ],
  dataset: ["datasetSplit", "datasetEvalSplit", "datasetConfig", "enableThinking", "loadingScope", "modelRevision", "datasetRevision", "seed"],
};

/** Groups holding a validation error: opened after a failed submit so the message is visible. */
export function groupsWithErrors(errors: FieldErrors<FormValues>): AdvancedGroup[] {
  return ADVANCED_GROUPS.filter((group) => GROUP_FIELDS[group].some((field) => errors[field as keyof FormValues] != null));
}

function Group({ value, title, summary, children }: { value: string; title: string; summary: ReactNode; children: ReactNode }) {
  return (
    <Accordion.Item value={value} className="border-b border-line last:border-b-0">
      <Accordion.Header className="m-0">
        <Accordion.Trigger className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left hover:bg-surface-2">
          <span className="flex min-w-0 flex-col">
            <span className="text-sm font-semibold text-ink">{title}</span>
            <span className="truncate text-[12px] text-muted">{summary}</span>
          </span>
          <span aria-hidden className="shrink-0 text-muted transition-transform group-data-[state=open]:rotate-90">
            ▸
          </span>
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="px-4 pt-1 pb-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">{children}</div>
      </Accordion.Content>
    </Accordion.Item>
  );
}

function Wide({ children }: { children: ReactNode }) {
  return <div className="sm:col-span-2">{children}</div>;
}

export function AdvancedSettings({
  resolved,
  datasetInspection,
  open,
  onOpenChange,
}: {
  resolved: ResolvedConfig | null;
  datasetInspection: DatasetInspection | null;
  /** Controlled open groups (uncontrolled when omitted). */
  open?: string[];
  onOpenChange?: (groups: string[]) => void;
}) {
  const values = useWatch<FormValues>() as FormValues;
  const { setValue } = useFormContext<FormValues>();
  const profiles = useBackendProfiles();
  const isFull = values.strategy === "full";
  const unit = MICROBATCH_UNIT[values.objective];
  const auto = (v: string) => (v === "" ? "자동" : v);

  return (
    <Accordion.Root type="multiple" value={open} onValueChange={onOpenChange} className="overflow-hidden rounded-xl border border-line bg-surface">
      <div className="border-b border-line px-4 py-3">
        <h2 className="text-lg font-semibold tracking-tight text-ink">Advanced</h2>
        <p className="text-[12px] text-muted">비워 둔 값은 공개된 프로필 preset으로 해석하고, 실제 적용값은 분석 후 표시합니다.</p>
      </div>

      <Group value="adapter" title="Adapter" summary={isFull ? "Full fine-tuning: adapter 미사용" : `r ${values.loraR} · alpha ${values.loraAlpha} · dropout ${values.loraDropout} · ${values.loraTargetMode === "custom" ? "직접 지정" : values.loraTargetMode}`}>
        {isFull ? (
          <Wide>
            <p className="text-[13px] text-muted">Full fine-tuning에서는 LoRA 설정을 사용하지 않습니다.</p>
          </Wide>
        ) : null}
        <TextField name="loraR" label="rank (r)" numeric disabled={isFull} hint="품질 최적값이 아닌 제품 초기값 16" applied={resolved?.lora?.r} />
        <TextField name="loraAlpha" label="alpha" numeric disabled={isFull} hint="scaling 값. alpha만 바뀌면 파라미터 수는 같습니다." applied={resolved?.lora?.alpha} />
        <TextField name="loraDropout" label="dropout" numeric disabled={isFull} applied={resolved?.lora?.dropout} />
        <SelectField
          name="loraTargetMode"
          label="target modules"
          disabled={isFull}
          options={[
            { value: "auto_verified", label: "검증된 구조별 preset" },
            { value: "all-linear", label: "all-linear (PEFT)" },
            { value: "custom", label: "직접 지정" },
          ]}
          hint="all-linear는 vision tower Linear에도 adapter를 붙일 수 있습니다."
        />
        {values.loraTargetMode === "custom" ? (
          <Wide>
            <ListField name="loraTargetModules" label="대상 모듈 (쉼표 또는 줄바꿈)" placeholder="q_proj, k_proj, v_proj, o_proj" disabled={isFull} />
          </Wide>
        ) : null}
        <ListField name="loraExcludeModules" label="제외 모듈" placeholder="visual" disabled={isFull} />
        <ListField name="loraModulesToSave" label="modules_to_save (추가 학습 모듈)" hint="원본은 그대로 두고 학습용 사본·gradient·state가 따로 집계됩니다." disabled={isFull} />
        <SelectField name="loraBias" label="bias" options={enumOptions(LORA_BIAS_LABEL)} disabled={isFull} />
        <ListField name="loraRankPattern" label="rank_pattern (패턴=rank)" placeholder="q_proj=8" disabled={isFull} />
        <SwitchControl name="loraUseDora" label="DoRA" disabled={isFull} />
        <SwitchControl name="loraUseRslora" label="rsLoRA" disabled={isFull} />
        {resolved?.lora ? (
          <Wide>
            <p className="text-[12px] text-ink-2">
              해석한 대상 모듈 {resolved.lora.target_modules.length}개 (패턴: <Mono>{resolved.lora.target_module_patterns.join(", ") || "—"}</Mono>)
            </p>
          </Wide>
        ) : null}
      </Group>

      <Group value="batch" title="Batch & precision" summary={`microbatch ${auto(values.microbatch)} · accumulation ${auto(values.accumulation)} · ${OPTIMIZER_LABEL[values.optimizer]} · load ${LOAD_DTYPE_LABEL[values.loadDtype]}`}>
        <TextField name="microbatch" label={`microbatch (${unit})`} numeric placeholder="자동 (프로필 preset)" applied={resolved?.microbatch} />
        <TextField name="accumulation" label="gradient accumulation" numeric placeholder="자동 (프로필 preset)" hint="activation에 곱하지 않습니다. gradient·optimizer state는 유지됩니다." applied={resolved?.accumulation} />
        <TextField name="padToMultipleOf" label="pad_to_multiple_of" numeric placeholder="없음" applied={resolved ? (resolved.pad_to_multiple_of ?? "없음") : undefined} />
        <SelectField name="precision" label="mixed precision" options={enumOptions(PRECISION_LABEL)} applied={resolved?.effective_dtypes.compute} />
        <SelectField
          name="loadDtype"
          label="load dtype"
          options={enumOptions(LOAD_DTYPE_LABEL)}
          hint="TRL 1.14.1은 dtype을 지정하지 않으면 float32로 로드합니다. 자동은 프로필 preset을 따릅니다."
          applied={resolved?.load_dtype}
        />
        <SelectField name="optimizer" label="optimizer" options={enumOptions(OPTIMIZER_LABEL)} applied={resolved?.optimizer.name} />
        <SelectField name="quantFormat" label="4-bit 형식" options={enumOptions(QUANT_FORMAT_LABEL)} disabled={!values.load4bit} />
        <SelectField name="computeDtype" label="4-bit compute dtype" options={enumOptions(COMPUTE_DTYPE_LABEL)} disabled={!values.load4bit} applied={resolved?.quantization.compute_dtype ?? undefined} />
        <SwitchControl name="doubleQuant" label="double quantization" disabled={!values.load4bit} description="4-bit 로딩을 켠 경우에만 적용됩니다." />
      </Group>

      <Group value="runtime" title="Runtime" summary={`checkpointing ${values.gradientCheckpointing ? "켬" : "끔"} · attention ${ATTENTION_BACKEND_LABEL[values.attentionBackend]} · loss ${LOSS_KERNEL_LABEL[values.lossKernel]}`}>
        <Wide>
          <SwitchControl name="gradientCheckpointing" label="gradient checkpointing" description="지원 경로에서 decoder layer 단위로 적용합니다." />
        </Wide>
        <SelectField name="attentionBackend" label="attention backend" options={enumOptions(ATTENTION_BACKEND_LABEL)} applied={resolved?.attention_path_by_layer_type?.full_attention} />
        <SelectField
          name="linearAttentionKernel"
          label="linear-attention kernel"
          options={enumOptions(LINEAR_ATTENTION_KERNEL_LABEL)}
          hint="Qwen3.5 같은 hybrid 구조에서만 사용합니다. 설치된 커널에 따라 메모리 경로가 달라집니다."
          applied={resolved?.attention_path_by_layer_type?.linear_attention}
        />
        <SelectField name="lossKernel" label="loss kernel" options={enumOptions(LOSS_KERNEL_LABEL)} hint="호환 프로필에서만 절감을 반영합니다." applied={resolved?.loss_path} />
        <SelectField
          name="backendProfile"
          label="backend profile"
          options={[{ value: "auto", label: "자동 (등록 프로필 중 선택)" }, ...(profiles.data?.profiles ?? []).map((p) => ({ value: p.profile_id, label: `${p.profile_id} · ${p.profile_version}` }))]}
          applied={resolved?.profile_id ?? undefined}
        />
        <Wide>
          <UnsupportedSwitch
            id="runtime-packing"
            label="packing"
            reason="엄격 무절단 모드에서는 packing을 끕니다. 샘플 분할·연결과 경계 누수를 막기 위해서이며 검증된 보존형 packing은 아직 없습니다."
          />
        </Wide>
        <UnsupportedSwitch id="runtime-offload-params" label="parameter offload" reason="이 릴리스는 offload 메모리 profile을 제공하지 않습니다." />
        <UnsupportedSwitch id="runtime-offload-optimizer" label="optimizer offload" reason="이 릴리스는 offload 메모리 profile을 제공하지 않습니다." />
        <UnsupportedSwitch id="runtime-offload-activations" label="activation offload" reason="이 릴리스는 offload 메모리 profile을 제공하지 않습니다." />
        <UnsupportedSwitch id="runtime-compile" label="torch.compile" reason="compile은 별도 메모리 profile이 필요해 지원하지 않습니다." />
      </Group>

      <Group value="method" title={`Method-specific · ${OBJECTIVE_LABEL[values.objective]}`} summary={methodSummary(values)}>
        <MethodSpecific values={values} setGenerationUnit={(u) => setValue("grpoGenerationUnit", u, { shouldDirty: true, shouldValidate: true })} setBudgetMode={(m) => setValue("grpoBudgetMode", m, { shouldDirty: true, shouldValidate: true })} />
      </Group>

      <Group value="dataset" title="Dataset & reproducibility" summary={`split ${values.datasetSplit || "자동"} · ${EMPTY_SYSTEM_POLICY_LABEL[values.mappingEnabled ? values.emptySystemPolicy : "omit"]} · thinking ${values.enableThinking === "default" ? "템플릿 기본" : values.enableThinking === "on" ? "켬" : "끔"}`}>
        <DatasetReproducibility inspection={datasetInspection} />
      </Group>
    </Accordion.Root>
  );
}

function methodSummary(values: FormValues): string {
  if (values.objective === "dpo") return `reference ${REFERENCE_STRATEGY_LABEL[values.dpoReferenceStrategy]} · beta ${values.dpoBeta}`;
  if (values.objective === "grpo") {
    const budget = values.grpoBudgetMode === "explicit" ? `budget ${values.grpoCompletionBudget || "?"}` : `budget 후보 ${values.grpoBudgetCandidates}`;
    return `G ${values.grpoNumGenerations} · ${budget} · reward ${REWARD_KIND_LABEL[values.grpoRewardKind]}`;
  }
  return "추가 설정 없음";
}

function MethodSpecific({
  values,
  setGenerationUnit,
  setBudgetMode,
}: {
  values: FormValues;
  setGenerationUnit: (unit: FormValues["grpoGenerationUnit"]) => void;
  setBudgetMode: (mode: FormValues["grpoBudgetMode"]) => void;
}) {
  if (values.objective === "sft") {
    return (
      <Wide>
        <p className="text-[13px] text-muted">SFT에는 방법별 추가 설정이 없습니다. 데이터 변환은 Dataset 영역의 매핑에서 확인하세요.</p>
      </Wide>
    );
  }
  if (values.objective === "dpo") {
    return (
      <>
        <SelectField name="dpoReferenceStrategy" label="reference 전략" options={enumOptions(REFERENCE_STRATEGY_LABEL)} hint="adapter off, 별도 모델, log-prob 사전 계산은 서로 다른 메모리 수명을 갖습니다." />
        {values.dpoReferenceStrategy === "standalone_model" ? (
          <TextField name="dpoReferenceModel" label="reference 모델 (비우면 policy와 같은 checkpoint)" placeholder="org/model" />
        ) : null}
        <TextField name="dpoBeta" label="beta" numeric />
        <TextField name="dpoLossType" label="loss type" placeholder="sigmoid" />
        {values.dpoReferenceStrategy === "precomputed_log_probs" ? (
          <TextField name="dpoPrecomputeBatchSize" label="precompute batch" numeric placeholder="자동 (학습 microbatch)" />
        ) : null}
        <Wide>
          <SwitchControl name="dpoSyncRefModel" label="reference sync" description="PEFT 또는 사전 계산과 함께 쓸 수 없습니다. 충돌하면 서버가 오류로 보고합니다." />
        </Wide>
      </>
    );
  }
  return (
    <>
      <TextField name="grpoNumGenerations" label="num_generations (G)" numeric hint="TRL GRPO는 G ≥ 2와 batch 배수 조건을 검사합니다." />
      <TextField name="grpoNumIterations" label="num_iterations" numeric />
      <Wide>
        <div className="flex flex-col gap-2">
          <span id="gen-unit-label" className="text-[13px] font-medium text-ink-2">
            생성 단위 (둘 중 하나만 지정)
          </span>
          <Segmented
            labelId="gen-unit-label"
            label="생성 단위"
            size="sm"
            value={values.grpoGenerationUnit}
            onValueChange={setGenerationUnit}
            options={[
              { value: "generation_batch_size", label: "generation_batch_size" },
              { value: "steps_per_generation", label: "steps_per_generation" },
            ]}
          />
        </div>
      </Wide>
      {values.grpoGenerationUnit === "generation_batch_size" ? (
        <TextField name="grpoGenerationBatchSize" label="generation_batch_size" numeric placeholder="자동" hint="한 번 생성할 completion 수 (프로세스 합)" />
      ) : (
        <TextField name="grpoStepsPerGeneration" label="steps_per_generation" numeric placeholder="자동" />
      )}
      <TextField name="grpoBeta" label="beta (KL)" numeric hint="메모리를 맞추려고 자동으로 0으로 바꾸지 않습니다." />
      <Wide>
        <div className="flex flex-col gap-2">
          <span id="budget-mode-label" className="text-[13px] font-medium text-ink-2">
            completion budget
          </span>
          <Segmented
            labelId="budget-mode-label"
            label="completion budget"
            size="sm"
            value={values.grpoBudgetMode}
            onValueChange={setBudgetMode}
            options={[
              { value: "candidates", label: "예산별 시나리오" },
              { value: "explicit", label: "지정" },
            ]}
          />
        </div>
      </Wide>
      {values.grpoBudgetMode === "explicit" ? (
        <TextField name="grpoCompletionBudget" label="completion budget (token)" numeric placeholder="2048" />
      ) : (
        <TextField name="grpoBudgetCandidates" label="budget 후보 (쉼표 구분)" hint="각 예산마다 별도 시나리오를 계산합니다." />
      )}
      <UnsupportedSwitch id="grpo-max-live" label="max_live_sequences 제한" reason="Transformers 공유 policy rollout은 동시 생성 수를 별도로 제한하지 않습니다 (C = update microbatch × steps_per_generation)." />
      <SelectField name="grpoRewardKind" label="reward" options={enumOptions(REWARD_KIND_LABEL)} hint="미지정이면 reward footprint를 제외한 조건부 결과입니다." />
      {values.grpoRewardKind === "local_model" ? (
        <>
          <TextField name="grpoRewardModelReference" label="reward 모델" placeholder="org/reward-model" />
          <SwitchControl name="grpoRewardOnTrainingGpu" label="학습 GPU에 reward 모델 상주" />
        </>
      ) : null}
      <Wide>
        <label htmlFor="grpo-rollout" className="text-[13px] font-medium text-ink-2">
          rollout backend
        </label>
        <select id="grpo-rollout" className="mt-1 h-10 w-full rounded-lg border border-line-strong bg-surface px-3 text-[15px]" value="transformers_shared_policy" onChange={() => {}} aria-describedby="grpo-rollout-hint">
          <option value="transformers_shared_policy">{ROLLOUT_BACKEND_LABEL.transformers_shared_policy}</option>
          <option value="vllm_colocate" disabled>
            {ROLLOUT_BACKEND_LABEL.vllm_colocate} (미지원)
          </option>
          <option value="vllm_server" disabled>
            {ROLLOUT_BACKEND_LABEL.vllm_server} (미지원)
          </option>
        </select>
        <p id="grpo-rollout-hint" className="mt-1 text-[12px] text-muted">1차 범위는 학습 policy를 그대로 쓰는 Transformers 생성 경로입니다.</p>
      </Wide>
    </>
  );
}

/** Read-only mapping summary; the editor itself lives next to the dataset input. Only the roles of
 * the chosen format are listed (roles left from another format are never sent). */
function MappingSummary() {
  const values = useWatch<FormValues>() as FormValues;
  const pairs = Object.entries(mappedColumns(values)).map(([role, column]) => `${column} → ${role}`);
  const jump = () => {
    const target = document.getElementById("mapping-format") ?? document.getElementById("dataset-reference");
    target?.focus();
  };
  return (
    <Wide>
      <div className="flex flex-col gap-1 rounded-lg border border-line bg-surface-2 px-3 py-2 text-[13px]">
        <span className="font-medium text-ink-2">컬럼 매핑</span>
        <span className="text-ink">
          {values.mappingEnabled && pairs.length > 0 ? pairs.join(", ") : "자동 감지 (분석 시 서버가 적용한 매핑을 결과에 표시)"}
        </span>
        <span>
          <button type="button" onClick={jump} className="text-[12px] text-accent underline underline-offset-2">
            Dataset 영역의 매핑 편집으로 이동
          </button>
        </span>
      </div>
    </Wide>
  );
}

function DatasetReproducibility({ inspection }: { inspection: DatasetInspection | null }) {
  const splits = (inspection?.splits ?? []).map((s) => s.name);
  const mappingEnabled = useWatch<FormValues, "mappingEnabled">({ name: "mappingEnabled" });
  const { setValue, getValues } = useFormContext<FormValues>();
  return (
    <>
      <MappingSummary />
      {splits.length > 0 ? (
        <SelectField name="datasetSplit" label="학습 split" options={[{ value: "", label: "자동 (train)" }, ...splits.map((s) => ({ value: s, label: s }))]} />
      ) : (
        <TextField name="datasetSplit" label="학습 split" placeholder="자동 (train)" />
      )}
      {splits.length > 0 ? (
        <SelectField name="datasetEvalSplit" label="평가 split" options={[{ value: "", label: "없음" }, ...splits.map((s) => ({ value: s, label: s }))]} hint="선택하면 평가 데이터도 같은 무절단 검사를 거칩니다." />
      ) : (
        <TextField name="datasetEvalSplit" label="평가 split" placeholder="없음" />
      )}
      <TextField
        name="datasetConfig"
        label="데이터셋 config"
        placeholder="자동"
        registerOptions={{ onChange: () => resetConfigSelections(setValue, getValues) }}
      />
      {mappingEnabled ? (
        <SelectField name="emptySystemPolicy" label="빈 system 메시지" options={enumOptions(EMPTY_SYSTEM_POLICY_LABEL)} hint="유지하면 템플릿이 빈 system 블록을 렌더링해 길이가 늘 수 있습니다." />
      ) : (
        // The policy is part of an explicit ColumnMapping (schemas/request.py); with auto-detection
        // the server applies "omit", so the control shows that and cannot be changed here.
        <Field label="빈 system 메시지" htmlFor="f-emptySystemPolicy" hint="자동 감지 매핑에는 기본값(빈 system 생략)이 적용됩니다. 컬럼 매핑을 지정하면 바꿀 수 있습니다.">
          <NativeSelect id="f-emptySystemPolicy" value="omit" disabled aria-describedby="f-emptySystemPolicy-hint" onChange={() => {}}>
            <option value="omit">{EMPTY_SYSTEM_POLICY_LABEL.omit}</option>
          </NativeSelect>
        </Field>
      )}
      <SelectField
        name="enableThinking"
        label="enable_thinking (chat template)"
        options={[
          { value: "default", label: "템플릿 기본값" },
          { value: "on", label: "true" },
          { value: "off", label: "false" },
        ]}
        hint="템플릿이 이 인자를 지원할 때만 의미가 있습니다. 바꾸면 토큰화를 다시 합니다."
      />
      <SelectField name="loadingScope" label="loading scope" options={enumOptions(LOADING_SCOPE_LABEL)} />
      <TextField name="modelRevision" label="모델 revision" placeholder="최신 (분석 시작 시 commit으로 고정)" />
      <TextField name="datasetRevision" label="데이터셋 revision" placeholder="최신 (분석 시작 시 commit으로 고정)" />
      <TextField name="seed" label="seed" numeric />
      <Wide>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <SwitchControl name="includeEvaluation" label="평가 단계 포함" description="끄면 평가 피크는 결과에서 제외로 표시됩니다." />
          <SwitchControl name="includeCheckpointSave" label="체크포인트 저장 포함" />
        </div>
      </Wide>
    </>
  );
}
