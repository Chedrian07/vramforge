"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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

/** One answered scenario: the request it was computed for, what changed, and its result. */
export interface ScenarioHistoryEntry {
  base: string;
  key: string;
  changes: string[];
  request: AnalysisRequest;
  result: AnalysisResult;
}

/** The recomputed scenario on screen (null while the stored base analysis is shown). */
export interface DisplayedScenario {
  request: AnalysisRequest;
  changes: string[];
}

export interface RecomputeView {
  mode: RecomputeMode;
  display: AnalysisResult | null;
  /** Set when `display` is a recomputed scenario rather than the stored analysis. */
  scenario: DisplayedScenario | null;
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

const UNREADABLE_ERROR = new ApiError(0, {
  code: "INTERNAL_ERROR",
  severity: "error",
  retryable: true,
  user_message: "재계산 응답을 처리하지 못했습니다. 잠시 후 다시 시도하세요.",
});

const SERVER_REANALYSIS = ["서버가 전체 데이터 재분석이 필요하다고 판단했습니다."];

const asScenario = (entry: ScenarioHistoryEntry): DisplayedScenario => ({ request: entry.request, changes: entry.changes });

/**
 * Recomputes light changes on the server and decides what the result area shows. The returned
 * view keeps its identity while nothing changed, so memoized result components do not re-render
 * on every keystroke.
 */
export function useRecompute({ analysisId, baseResult, baseRequest, currentRequest, enabled }: RecomputeInput): RecomputeView {
  const { api } = useApiEnvironment();
  const baseKey = baseResult?.analysis_fingerprint ?? null;
  const [cache, setCache] = useState<ReadonlyMap<string, ScenarioHistoryEntry>>(() => new Map());
  const [history, setHistory] = useState<ScenarioHistoryEntry[]>([]);
  const [serverReanalysis, setServerReanalysis] = useState<{ key: string; reasons: string[] } | null>(null);
  const [failure, setFailure] = useState<{ key: string; error: ApiError } | null>(null);
  const [settled, setSettled] = useState<ScenarioHistoryEntry | null>(null);
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

  const decided = useMemo(() => {
    const fallback = settled && settled.base === baseKey ? settled : null;
    let mode: RecomputeMode;
    let display: AnalysisResult | null = fallback?.result ?? baseResult;
    let scenario: DisplayedScenario | null = fallback ? asScenario(fallback) : null;
    let stale = true;
    let reasons: string[] = [];
    let error: ApiError | null = null;

    if (!baseResult) {
      mode = "idle";
      display = null;
      scenario = null;
      stale = false;
    } else if (!currentRequest || !changes || !key) {
      mode = "invalid";
    } else if (changes.reanalysis.length > 0) {
      mode = "reanalysis";
      reasons = changes.reanalysis;
    } else if (changes.recompute.length === 0) {
      mode = "current";
      display = baseResult;
      scenario = null;
      stale = false;
    } else if (cache.has(key)) {
      const entry = cache.get(key)!;
      mode = "current";
      display = entry.result;
      scenario = asScenario(entry);
      stale = false;
    } else if (serverReanalysis?.key === key) {
      mode = "reanalysis";
      reasons = serverReanalysis.reasons.length ? serverReanalysis.reasons : SERVER_REANALYSIS;
    } else if (failure?.key === key) {
      mode = "error";
      error = failure.error;
    } else {
      mode = inflight === key ? "loading" : "pending";
    }
    return { mode, display, scenario, stale, reasons, error };
  }, [baseKey, baseResult, cache, changes, currentRequest, failure, inflight, key, serverReanalysis, settled]);

  const needsRequest = enabled && analysisId != null && (decided.mode === "pending" || decided.mode === "loading");
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
              const entry: ScenarioHistoryEntry = {
                base: baseKey,
                key,
                changes: (recomputeChanges ?? []).map(describeChange),
                request: currentRequest,
                result: response.result,
              };
              setCache((prev) => new Map(prev).set(key, entry));
              setSettled(entry);
              setHistory((prev) => [...prev.filter((e) => e.key !== key), entry]);
            } else {
              setFailure({ key, error: NO_RESULT_ERROR });
            }
          },
          (err: unknown) => {
            if (isAbortError(err) || latestKey.current !== key) return;
            setFailure({ key, error: err instanceof ApiError ? err : UNREADABLE_ERROR });
          },
        )
        .finally(() => setInflight((current) => (current === key ? null : current)));
    }, RECOMPUTE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [needsRequest, key, currentRequest, analysisId, baseKey, fingerprint, api, recomputeChanges]);

  const baseHistory = useMemo(() => history.filter((entry) => entry.base === baseKey), [history, baseKey]);
  const retry = useCallback(() => setFailure(null), []);

  return useMemo(
    () => ({
      mode: decided.mode,
      display: decided.display,
      scenario: decided.scenario,
      stale: decided.stale,
      reanalysisReasons: decided.reasons,
      changes,
      error: decided.error,
      history: baseHistory,
      retry,
    }),
    [decided, changes, baseHistory, retry],
  );
}
