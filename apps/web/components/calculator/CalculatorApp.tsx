"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FormProvider, useForm, useWatch } from "react-hook-form";

import { TokenPrompt } from "@/components/layout/TokenPrompt";
import { DeleteAnalysis } from "@/components/results/DeleteAnalysis";
import { DetailTabs } from "@/components/results/DetailTabs";
import { ExportMenu, RetentionNote, type ExportTarget } from "@/components/results/ExportMenu";
import { SummaryCard } from "@/components/results/SummaryCard";
import { useApiEnvironment } from "@/lib/api/context";
import type { GpuPreset } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { buildRequest, fromAnalysisRequest } from "@/lib/form/convert";
import { canonicalJson } from "@/lib/form/fingerprint";
import { datasetInspectCall, modelInspectCall } from "@/lib/form/inspect";
import type { ExplicitAnalysisRequest } from "@/lib/form/request-schema";
import { useRevalidateShownErrors } from "@/lib/form/revalidate";
import { DEFAULT_FORM_VALUES, EXAMPLE_FORM_VALUES, formSchema, type FormValues } from "@/lib/form/values";
import { isRunActive, useAnalysisRun } from "@/lib/hooks/useAnalysisRun";
import { useFitsViewport } from "@/lib/hooks/useFitsViewport";
import { useHydrated } from "@/lib/hooks/useHydrated";
import { useBackendProfiles, useDatasetInspection, useModelInspection } from "@/lib/hooks/useInspection";
import { useRecompute } from "@/lib/hooks/useRecompute";

import { AdvancedSettings, groupsWithErrors } from "./AdvancedSettings";
import { DatasetSection } from "./DatasetSection";
import { HardwareSection } from "./HardwareSection";
import { syncMappingWithInspection } from "./MappingEditor";
import { MethodSection } from "./MethodSection";
import { ModelSection } from "./ModelSection";
import { ProgressPanel } from "./ProgressPanel";
import { RunBar } from "./RunBar";

const NO_PRESETS: GpuPreset[] = [];

/** Same request object while the form content is unchanged (keeps the recompute debounce stable). */
function useCurrentRequest(values: FormValues, presets: readonly GpuPreset[]): ExplicitAnalysisRequest | null {
  const valid = formSchema.safeParse(values).success;
  const built = valid ? buildRequest(values, presets) : null;
  const json = built?.ok ? canonicalJson(built.request) : null;
  return useMemo(() => (json ? (JSON.parse(json) as ExplicitAnalysisRequest) : null), [json]);
}

function Card({ title, children, id }: { title: string; children: React.ReactNode; id: string }) {
  return (
    <section aria-labelledby={id} className="flex flex-col gap-4 rounded-xl border border-line bg-surface p-4 sm:p-6">
      <h2 id={id} className="text-lg font-semibold tracking-tight text-ink">
        {title}
      </h2>
      {children}
    </section>
  );
}

export function CalculatorApp() {
  const env = useApiEnvironment();
  const profiles = useBackendProfiles();
  const presets = profiles.data?.hardware_presets ?? NO_PRESETS;
  const form = useForm<FormValues>({
    defaultValues: DEFAULT_FORM_VALUES,
    resolver: zodResolver(formSchema),
    mode: "onChange",
  });
  const values = useWatch({ control: form.control }) as FormValues;
  useRevalidateShownErrors(form);
  const run = useAnalysisRun();
  const modelInspection = useModelInspection();
  const datasetInspection = useDatasetInspection();
  const [submitError, setSubmitError] = useState<string | null>(null);

  const currentRequest = useCurrentRequest(values, presets);
  const runState = run.state;
  const terminalResult = runState.phase === "terminal" ? (runState.status?.result ?? null) : null;
  const baseResult = runState.jobStatus === "COMPLETED" ? terminalResult : null;
  // What the analysis was started with: this page's submission, else the stored request.
  const baseRequest = runState.submittedRequest ?? runState.status?.request ?? baseResult?.requested_config ?? null;
  const recompute = useRecompute({
    analysisId: runState.analysisId,
    baseResult,
    baseRequest,
    currentRequest,
    enabled: baseResult != null,
  });

  // After a reload (?analysis=<id>) the form starts empty: restore it once from the stored
  // request, which the status carries from the first answer on (queued or running, before any
  // result); older answers without it fall back to the result's requested_config. Edits made
  // before the answer arrived are kept.
  const hydratedFor = useRef<string | null>(null);
  const resumed = runState.submittedRequest ? null : runState.status;
  const resumedId = resumed?.analysis_id ?? null;
  const resumedRequest = resumed?.request ?? resumed?.result?.requested_config ?? null;
  useEffect(() => {
    if (!resumedId || !resumedRequest || hydratedFor.current === resumedId) return;
    hydratedFor.current = resumedId;
    // formState.isDirty is only computed when it is read during render, so compare with the
    // untouched defaults instead.
    const untouched = canonicalJson(form.getValues()) === canonicalJson(DEFAULT_FORM_VALUES);
    if (untouched) form.reset(fromAnalysisRequest(resumedRequest));
  }, [resumedId, resumedRequest, form]);

  const onValid = useCallback(
    (formValues: FormValues) => {
      const built = buildRequest(formValues, presets);
      if (!built.ok) {
        setSubmitError(built.message);
        return;
      }
      setSubmitError(null);
      void run.start(built.request);
    },
    [presets, run],
  );
  // Fields of a closed Advanced group are unmounted: open the groups that hold an error so the
  // message is visible and the field can take focus.
  const [advancedOpen, setAdvancedOpen] = useState<string[]>([]);
  const submit = form.handleSubmit(onValid, (errors) => {
    const groups = groupsWithErrors(errors);
    if (groups.length > 0) setAdvancedOpen((open) => Array.from(new Set([...open, ...groups])));
  });

  const loadExample = () => {
    form.reset(EXAMPLE_FORM_VALUES);
    modelInspection.forget();
    datasetInspection.forget();
    const model = modelInspectCall(EXAMPLE_FORM_VALUES);
    if (model) void modelInspection.inspect(model.key, model.body);
    const dataset = datasetInspectCall(EXAMPLE_FORM_VALUES);
    if (dataset) {
      void datasetInspection.inspect(dataset.key, dataset.body).then((data) => {
        if (data) syncMappingWithInspection(form.setValue, form.getValues, data);
      });
    }
  };

  // Exports follow what the summary shows: a recomputed scenario is exported with the request it
  // was computed for; otherwise the stored analysis. The slot is memoized so the memoized summary
  // card does not re-render on every keystroke.
  const expiresAt = runState.status?.expires_at ?? null;
  const exportTarget = useMemo<ExportTarget>(
    () =>
      recompute.scenario && recompute.display
        ? { kind: "scenario", result: recompute.display, request: recompute.scenario.request, changes: recompute.scenario.changes, stale: recompute.stale }
        : { kind: "stored", result: terminalResult, jobStatus: runState.jobStatus, formDiffers: baseResult != null && recompute.stale },
    [recompute.scenario, recompute.display, recompute.stale, terminalResult, runState.jobStatus, baseResult],
  );
  const resetRun = run.reset;
  const running = isRunActive(runState);
  const exportSlot = useMemo(
    () => (
      <>
        <ExportMenu api={env.api} analysisId={runState.analysisId} target={exportTarget} expiresAt={expiresAt} />
        <DeleteAnalysis api={env.api} analysisId={runState.analysisId} disabled={running} onDeleted={resetRun} />
        <RetentionNote expiresAt={expiresAt} className="w-full text-[12px] leading-snug text-muted" />
      </>
    ),
    [env.api, runState.analysisId, exportTarget, expiresAt, running, resetRun],
  );

  const hydrated = useHydrated();
  const summaryRef = useRef<HTMLDivElement>(null);
  const summaryFits = useFitsViewport(summaryRef);
  const shown = recompute.display ?? runState.status?.result ?? null;
  const invalidCount = Object.keys(form.formState.errors).length;
  const datasetData = datasetInspection.state.status === "done" ? datasetInspection.state.data : null;

  return (
    <FormProvider {...form}>
      {env.devMockLabel ? (
        <div role="note" className="mx-auto mb-4 w-full max-w-[1320px] px-4 sm:px-6">
          <p className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
            개발용 mock 데이터 모드 ({env.devMockLabel}): 표시되는 수치는 fixture이며 실제 분석 결과가 아닙니다.
          </p>
        </div>
      ) : null}
      {/* data-hydrated lets end-to-end tests wait until the client handlers are attached. */}
      <main data-hydrated={hydrated ? "true" : "false"} className="mx-auto w-full max-w-[1320px] px-4 pb-16 sm:px-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,58fr)_minmax(0,42fr)]">
          <form onSubmit={submit} noValidate aria-label="계산 입력" className="flex min-w-0 flex-col gap-6 lg:col-start-1 lg:row-start-1">
            <Card title="모델 · 데이터" id="sources-title">
              <ModelSection inspection={modelInspection} />
              <div className="border-t border-line" />
              <DatasetSection inspection={datasetInspection} />
            </Card>
            <Card title="학습 방식" id="method-title">
              <MethodSection />
            </Card>
            <Card title="하드웨어" id="hardware-title">
              <HardwareSection />
            </Card>
            <AdvancedSettings
              resolved={shown?.resolved_config ?? null}
              datasetInspection={datasetData}
              open={advancedOpen}
              onOpenChange={setAdvancedOpen}
            />
            <RunBar
              running={running}
              needsReanalysis={recompute.mode === "reanalysis"}
              reanalysisReasons={recompute.reanalysisReasons}
              invalidCount={invalidCount}
              onExample={loadExample}
            />
            {submitError ? (
              <p role="alert" className="text-[13px] text-err">
                {submitError}
              </p>
            ) : null}
          </form>

          <div className="min-w-0 lg:col-span-2 lg:row-start-2">
            <ProgressPanel
              run={runState}
              onCancel={() => void run.cancel()}
              onRerun={() => void submit()}
              onRetry={() => {
                if (runState.analysisId) void run.resume(runState.analysisId);
              }}
            />
          </div>

          {/* The aside stretches over the input row; the inner box sticks inside it, so the card can
              never slide over the progress panel below. */}
          <aside aria-label="결과 요약" className="min-w-0 lg:col-start-2 lg:row-start-1">
            <div ref={summaryRef} className={cn(summaryFits && "lg:sticky lg:top-6")}>
              <SummaryCard
                run={runState}
                view={recompute}
                hardwareRequested={values.hardwareMode !== "capacity_only"}
                gpuWorkerConnected={profiles.data?.gpu_worker_connected ?? null}
                exportSlot={exportSlot}
              />
            </div>
          </aside>

          <div className="min-w-0 lg:col-span-2 lg:row-start-3">
            <DetailTabs
              result={shown}
              base={baseResult}
              history={recompute.history}
              partial={shown != null && runState.jobStatus !== "COMPLETED"}
              stale={recompute.stale}
            />
          </div>
        </div>
      </main>
      <TokenPrompt />
    </FormProvider>
  );
}
