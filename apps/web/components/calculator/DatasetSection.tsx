"use client";

import { useRef, useState } from "react";
import { useFormContext, useWatch } from "react-hook-form";

import { CopyButton } from "@/components/ui/controls";
import { Badge, Button, Field, Mono, NativeSelect, TextInput, describedBy } from "@/components/ui/primitives";
import { IssueList } from "@/components/ui/values";
import { ApiError } from "@/lib/api/client";
import { useApi } from "@/lib/api/context";
import type { DatasetInspection, UploadResponse } from "@/lib/api/types";
import { formatCount, formatSize, shortDigest } from "@/lib/format/bytes";
import { DATASET_FORMAT_LABEL } from "@/lib/format/labels";
import { datasetInspectCall, referenceOfKey, shouldApplySuggestion } from "@/lib/form/inspect";
import { hasViewerRowParam } from "@/lib/form/references";
import type { FormValues } from "@/lib/form/values";
import { useLocalRoots, type useDatasetInspection } from "@/lib/hooks/useInspection";

import { MappingEditor, applyMapping } from "./MappingEditor";

type Inspection = ReturnType<typeof useDatasetInspection>;

const ACCEPT = ".json,.jsonl,.parquet,.arrow,.csv";

export function DatasetSection({ inspection }: { inspection: Inspection }) {
  const api = useApi();
  const { register, getValues, setValue, formState } = useFormContext<FormValues>();
  const reference = useWatch<FormValues, "datasetReference">({ name: "datasetReference" });
  const config = useWatch<FormValues, "datasetConfig">({ name: "datasetConfig" });
  const fileInput = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<{ status: "idle" | "uploading" | "done" | "error"; data?: UploadResponse; error?: ApiError }>({ status: "idle" });
  const roots = useLocalRoots();
  const error = formState.errors.datasetReference?.message;
  const id = "dataset-reference";
  const hint = "Hugging Face 데이터셋 ID·URL, 업로드 파일 또는 서버의 local: 경로. 조회는 메타데이터만 읽고 전체 스캔은 분석 시작 후 진행합니다.";

  const trigger = (force: boolean, overrides: Partial<Pick<FormValues, "datasetConfig">> = {}) => {
    const call = datasetInspectCall({ ...getValues(), ...overrides });
    if (!call) return;
    const run = force ? inspection.reinspect(call.key, call.body) : inspection.inspect(call.key, call.body);
    void run.then((data) => {
      // An unambiguous suggestion is applied (and shown) unless the user already edited it.
      // An auto-selected split stays "auto" (null) so the result can report split_auto_selected.
      if (data?.suggested_mapping && shouldApplySuggestion(data, getValues("mappingEnabled"))) {
        applyMapping(setValue, data.suggested_mapping);
      }
    });
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setUpload({ status: "uploading" });
    try {
      const data = await api.upload(file);
      setUpload({ status: "done", data });
      const opts = { shouldDirty: true, shouldValidate: true };
      setValue("datasetUploadRef", data.reference, opts);
      setValue("datasetReference", data.reference, opts);
      setValue("datasetConfig", "", opts);
      setValue("datasetSplit", "", opts);
      setValue("mappingEnabled", false, opts);
      inspection.forget();
      trigger(true);
    } catch (err) {
      setUpload({ status: "error", error: err instanceof ApiError ? err : undefined });
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const field = register("datasetReference", { onBlur: () => trigger(false) });
  const data: DatasetInspection | null = inspection.state.status === "done" ? inspection.state.data : null;
  const inspectedRef = referenceOfKey(inspection.state.key);
  const outdated = inspectedRef != null && inspectedRef !== reference.trim();
  const configs = data?.configs ?? [];
  const splits = data?.splits ?? [];
  const needConfig = configs.length > 1 || (configs.length > 0 && !data?.selected_config);
  const needSplit = splits.length > 1 || (splits.length > 0 && !data?.split_auto_selected);

  return (
    <div className="flex flex-col gap-3">
      <Field label="Dataset" htmlFor={id} hint={hint} error={error}>
        <div className="flex flex-wrap gap-2 sm:flex-nowrap">
          <TextInput
            id={id}
            {...field}
            placeholder="org/dataset · https://huggingface.co/datasets/org/dataset · local:datasets/…"
            className="min-w-0 flex-1 font-mono text-[14px]"
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
          <Button onClick={() => trigger(true)} aria-label="데이터셋 확인">
            확인
          </Button>
          <Button onClick={() => fileInput.current?.click()} disabled={upload.status === "uploading"}>
            {upload.status === "uploading" ? "업로드 중…" : "파일 업로드"}
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            tabIndex={-1}
            aria-label="데이터 파일 선택 (JSON, JSONL, Parquet, Arrow, CSV)"
            onChange={(event) => void onFile(event.target.files?.[0])}
          />
        </div>
      </Field>

      {hasViewerRowParam(reference) ? (
        <p className="text-[12px] text-muted">URL의 row 위치는 분석 범위를 제한하지 않습니다. 선택한 split 전체를 분석합니다.</p>
      ) : null}

      {upload.status === "done" && upload.data ? (
        <p className="flex flex-wrap items-center gap-2 text-[12px] text-ink-2">
          <Badge tone="ok">업로드됨</Badge>
          <Mono>{upload.data.filename}</Mono>
          <span className="num">{formatSize(upload.data.size_bytes)}</span>
          <span className="text-muted">sha256 {shortDigest(upload.data.sha256, 10)}</span>
          <span className="text-muted">보관 기한 {new Date(upload.data.expires_at).toLocaleString("ko-KR")}</span>
        </p>
      ) : null}
      {upload.status === "error" ? (
        <IssueList issues={[upload.error?.issue ?? { code: "INTERNAL_ERROR", severity: "error", user_message: "파일을 업로드하지 못했습니다." }]} />
      ) : null}

      {roots.data && roots.data.roots && roots.data.roots.length > 0 ? (
        <details className="text-[12px] text-muted">
          <summary className="cursor-pointer select-none">서버 로컬 경로 사용 (읽기 전용)</summary>
          <p className="mt-1">로컬 경로는 API 서버 기준입니다. 브라우저가 실행 중인 PC의 경로는 읽을 수 없습니다.</p>
          <ul className="mt-1 flex flex-col gap-1">
            {roots.data.roots.map((root) => (
              <li key={root.reference_prefix} className="flex min-w-0 flex-wrap items-center gap-2">
                <Mono className="text-ink">{root.reference_prefix}</Mono>
                <span>{root.description}</span>
                <CopyButton value={root.reference_prefix} />
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <div aria-live="polite" className="min-w-0">
        {inspection.state.status === "loading" ? <p className="text-[13px] text-muted">데이터셋 구조를 확인하는 중…</p> : null}
        {inspection.state.status === "error" && inspection.state.error ? <IssueList issues={[inspection.state.error.issue]} /> : null}
      </div>

      {data ? (
        <div className={`flex flex-col gap-3 ${outdated ? "opacity-60" : ""}`}>
          {outdated ? <p className="text-[12px] text-warn">입력한 데이터셋이 바뀌었습니다. 확인을 눌러 다시 조회하세요.</p> : null}
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            {data.detected_format ? <Badge tone="info">형식: {DATASET_FORMAT_LABEL[data.detected_format]}</Badge> : null}
            {splits.map((s) => (
              <span key={s.name} className="num text-ink-2">
                <Mono>{s.name}</Mono> {s.num_rows != null ? `${formatCount(s.num_rows)} rows` : "row 수 미확인"}
              </span>
            ))}
            {data.split_auto_selected && data.selected_split ? (
              <span className="text-muted">({data.selected_split} 자동 선택)</span>
            ) : null}
          </div>
          {needConfig || needSplit ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {needConfig ? (
                <Field label="Config" htmlFor="dataset-config-inline" hint="config마다 split과 컬럼이 다를 수 있습니다.">
                  <NativeSelect
                    id="dataset-config-inline"
                    value={config}
                    onChange={(event) => {
                      setValue("datasetConfig", event.target.value, { shouldDirty: true });
                      trigger(true, { datasetConfig: event.target.value });
                    }}
                  >
                    <option value="">— 선택 —</option>
                    {configs.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </NativeSelect>
                </Field>
              ) : null}
              {needSplit ? (
                <Field label="학습 split" htmlFor="dataset-split-inline" hint="선택하지 않은 split은 분석·학습에 섞지 않습니다.">
                  <NativeSelect id="dataset-split-inline" {...register("datasetSplit")}>
                    <option value="">— 선택 —</option>
                    {splits.map((s) => (
                      <option key={s.name} value={s.name}>
                        {s.name}
                      </option>
                    ))}
                  </NativeSelect>
                </Field>
              ) : null}
            </div>
          ) : null}
          <IssueList issues={data.issues ?? []} />
          <MappingEditor inspection={data} />
        </div>
      ) : (
        <MappingSummaryWithoutColumns />
      )}
    </div>
  );
}

/** Restored mapping (e.g. after reload) before the dataset is inspected again. */
function MappingSummaryWithoutColumns() {
  const values = useWatch<FormValues>();
  if (!values.mappingEnabled) return null;
  const pairs: Array<[string, string | undefined]> = [
    ["system", values.mapSystem],
    ["prompt", values.mapPrompt],
    ["chosen", values.mapChosen],
    ["rejected", values.mapRejected],
    ["completion", values.mapCompletion],
    ["messages", values.mapMessages],
    ["text", values.mapText],
  ];
  return (
    <p className="text-[12px] text-ink-2">
      적용 매핑:{" "}
      {pairs
        .filter(([, column]) => column)
        .map(([role, column]) => `${column} → ${role}`)
        .join(", ")}
      <span className="text-muted"> (데이터셋 확인 후 편집할 수 있습니다)</span>
    </p>
  );
}
