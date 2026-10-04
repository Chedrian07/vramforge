"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { ApiError, isAbortError } from "@/lib/api/client";
import { useApi } from "@/lib/api/context";
import type { DatasetInspection, InspectRequest, ModelInspection } from "@/lib/api/types";

export interface InspectionState<T> {
  key: string | null;
  status: "idle" | "loading" | "done" | "error";
  data: T | null;
  error: ApiError | null;
}

const IDLE = { key: null, status: "idle", data: null, error: null } as const;

/**
 * Metadata inspection triggered only by blur or the 확인 button (plan.md §3.3), never per keystroke.
 * Results are cached per reference; a newer request cancels the older one and late answers for an
 * older reference are ignored.
 */
function useInspection<T>(kind: "model" | "dataset", pick: (r: Awaited<ReturnType<ReturnType<typeof useApi>["inspect"]>>) => T | null) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [state, setState] = useState<InspectionState<T>>(IDLE);
  const sequence = useRef(0);
  const currentKey = useRef<string | null>(null);

  const inspect = useCallback(
    async (key: string, body: InspectRequest): Promise<T | null> => {
      if (currentKey.current === key) return null; // same reference already inspected or loading
      currentKey.current = key;
      const n = ++sequence.current;
      setState({ key, status: "loading", data: null, error: null });
      await queryClient.cancelQueries({ queryKey: ["inspect", kind], predicate: (q) => q.queryKey[2] !== key });
      try {
        const response = await queryClient.fetchQuery({
          queryKey: ["inspect", kind, key],
          queryFn: ({ signal }) => api.inspect(body, signal),
          staleTime: 5 * 60_000,
        });
        if (n !== sequence.current) return null;
        const data = pick(response);
        setState({ key, status: "done", data, error: null });
        return data;
      } catch (error) {
        if (n !== sequence.current || isAbortError(error)) return null;
        currentKey.current = null; // allow an explicit retry of the same reference
        setState({
          key,
          status: "error",
          data: null,
          error: error instanceof ApiError ? error : new ApiError(0, { code: "INTERNAL_ERROR", severity: "error", retryable: true, user_message: "메타데이터를 확인하지 못했습니다." }),
        });
        return null;
      }
    },
    [api, kind, pick, queryClient],
  );

  const forget = useCallback(() => {
    sequence.current += 1;
    currentKey.current = null;
    setState(IDLE);
  }, []);

  /** Re-run even when the key did not change (explicit 확인 click). */
  const reinspect = useCallback(
    (key: string, body: InspectRequest) => {
      currentKey.current = null;
      void queryClient.invalidateQueries({ queryKey: ["inspect", kind, key] });
      return inspect(key, body);
    },
    [inspect, kind, queryClient],
  );

  return { state, inspect, reinspect, forget };
}

const pickModel = (r: { model?: ModelInspection | null }) => r.model ?? null;
const pickDataset = (r: { dataset?: DatasetInspection | null }) => r.dataset ?? null;

export function useModelInspection() {
  return useInspection<ModelInspection>("model", pickModel);
}

export function useDatasetInspection() {
  return useInspection<DatasetInspection>("dataset", pickDataset);
}

export function useBackendProfiles() {
  const api = useApi();
  return useQuery({
    queryKey: ["backend-profiles"],
    queryFn: ({ signal }) => api.backendProfiles(signal),
    staleTime: Infinity,
    retry: false,
  });
}

export function useLocalRoots() {
  const api = useApi();
  return useQuery({
    queryKey: ["local-roots"],
    queryFn: ({ signal }) => api.localRoots(signal),
    staleTime: Infinity,
    retry: false,
  });
}
