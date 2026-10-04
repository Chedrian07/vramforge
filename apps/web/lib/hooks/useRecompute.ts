"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError, isAbortError } from "@/lib/api/client";
import { useApiEnvironment } from "@/lib/api/context";
import type { AnalysisRequest, AnalysisResult } from "@/lib/api/types";
import { classifyChanges, describeChange, type ChangeSummary, type ResolvedSelection } from "@/lib/form/changes";
import { requestFingerprint } from "@/lib/form/fingerprint";

/** plan.md §4.2: light changes are recomputed ~300 ms after the last edit. */
export const RECOMPUTE_DEBOUNCE_MS = 300;

export type RecomputeMode =
  | "idle" // no completed analysis yet
  | "current" // displayed result matches the form
  | "pending" // waiting for the debounce
  | "loading" // scenario request in flight
  | "reanalysis" // a full scan is needed (local classification or server answer)
  | "invalid" // the form cannot be turned into a request
  | "error";

export interface ScenarioHistoryEntry {
  base: string;
  key: string;
  changes: string[];
  result: AnalysisResult;
}

export interface RecomputeView {
  mode: RecomputeMode;
  display: AnalysisResult | null;
  /** Displayed numbers belong to an earlier setting ("이전 설정"). */
  stale: boolean;
  reanalysisReasons: string[];
  changes: ChangeSummary | null;
  error: ApiError | null;
  history: ScenarioHistoryEntry[];
  retry: () => void;
}

export interface RecomputeInput {
  analysisId: string | null;
  baseResult: AnalysisResult | null;
  /** What was submitted for the base analysis (falls back to its requested_config). */
  baseRequest: AnalysisRequest | null;
  currentRequest: AnalysisRequest | null;
  /** Scenario requests are only possible on a completed analysis with stored artifacts. */
  enabled: boolean;
}

const NO_RESULT_ERROR = new ApiError(0, {
  code: "INTERNAL_ERROR",
  severity: "error",
  retryable: true,
  user_message: "재계산 응답에 결과가 없습니다.",
});

export function useRecompute({ analysisId, baseResult, baseRequest, currentRequest, enabled }: RecomputeInput): RecomputeView {
  const { api } = useApiEnvironment();
  const baseKey = baseResult?.analysis_fingerprint ?? null;
  const [cache, setCache] = useState<ReadonlyMap<string, AnalysisResult>>(() => new Map());
  const [history, setHistory] = useState<ScenarioHistoryEntry[]>([]);
  const [serverReanalysis, setServerReanalysis] = useState<{ key: string; reasons: string[] } | null>(null);
  const [failure, setFailure] = useState<{ key: string; error: ApiError } | null>(null);
  const [settled, setSettled] = useState<{ base: string; result: AnalysisResult } | null>(null);
  const [inflight, setInflight] = useState<string | null>(null);
  const latestKey = useRef<string | null>(null);

  const reference = baseRequest ?? baseResult?.requested_config ?? null;
  const resolved: ResolvedSelection | undefined = useMemo(() => {
    const scan = baseResult?.dataset_scan;
    return scan ? { split: scan.split, config: scan.config, mapping: scan.mapping_applied } : undefined;
  }, [baseResult]);

  const changes = useMemo(
    () => (reference && currentRequest ? classifyChanges(reference, currentRequest, resolved) : null),
    [reference, currentRequest, resolved],
  );
  const fingerprint = useMemo(() => (currentRequest ? requestFingerprint(currentRequest) : null), [currentRequest]);
  const key = baseKey && fingerprint ? `${baseKey}|${fingerprint}` : null;

  const fallback = settled && settled.base === baseKey ? settled.result : baseResult;
  let mode: RecomputeMode;
  let display: AnalysisResult | null = fallback;
  let stale = true;
  let reasons: string[] = [];
  let error: ApiError | null = null;

  if (!baseResult) {
    mode = "idle";
    display = null;
    stale = false;
  } else if (!currentRequest || !changes || !key) {
    mode = "invalid";
  } else if (changes.reanalysis.length > 0) {
    mode = "reanalysis";
    reasons = changes.reanalysis;
  } else if (changes.recompute.length === 0) {
    mode = "current";
    display = baseResult;
    stale = false;
  } else if (cache.has(key)) {
    mode = "current";
    display = cache.get(key) ?? baseResult;
    stale = false;
  } else if (serverReanalysis?.key === key) {
    mode = "reanalysis";
    reasons = serverReanalysis.reasons.length ? serverReanalysis.reasons : ["서버가 전체 데이터 재분석이 필요하다고 판단했습니다."];
  } else if (failure?.key === key) {
    mode = "error";
    error = failure.error;
  } else {
    mode = inflight === key ? "loading" : "pending";
  }

  const needsRequest = enabled && analysisId != null && (mode === "pending" || mode === "loading");
  const recomputeChanges = changes?.recompute;

  useEffect(() => {
    if (!needsRequest || !key || !currentRequest || !analysisId || !baseKey || !fingerprint) return;
    latestKey.current = key;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setInflight(key);
      api
        .scenarios(analysisId, { request: currentRequest, client_fingerprint: fingerprint }, controller.signal)
        .then(
          (response) => {
            // Drop answers for settings that are no longer on screen (plan.md §4.2, §19.4).
            if (latestKey.current !== key) return;
            if (response.client_fingerprint != null && response.client_fingerprint !== fingerprint) return;
            if (response.requires_reanalysis) {
              setServerReanalysis({ key, reasons: response.reanalysis_reasons ?? [] });
            } else if (response.result) {
              const result = response.result;
              setCache((prev) => new Map(prev).set(key, result));
              setSettled({ base: baseKey, result });
              setHistory((prev) => [
                ...prev.filter((entry) => entry.key !== key),
                { base: baseKey, key, changes: (recomputeChanges ?? []).map(describeChange), result },
              ]);
            } else {
              setFailure({ key, error: NO_RESULT_ERROR });
            }
          },
          (err: unknown) => {
            if (isAbortError(err) || latestKey.current !== key) return;
            setFailure({ key, error: err instanceof ApiError ? err : NO_RESULT_ERROR });
          },
        )
        .finally(() => setInflight((current) => (current === key ? null : current)));
    }, RECOMPUTE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [needsRequest, key, currentRequest, analysisId, baseKey, fingerprint, api, recomputeChanges]);

  return {
    mode,
    display,
    stale,
    reanalysisReasons: reasons,
    changes,
    error,
    history: history.filter((entry) => entry.base === baseKey),
    retry: () => setFailure(null),
  };
}
