"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { ApiError, newIdempotencyKey } from "@/lib/api/client";
import { useApiEnvironment } from "@/lib/api/context";
import type { AnalysisEvent, EventIssue } from "@/lib/api/events";
import { openAnalysisStream, type ConnectionState } from "@/lib/api/stream";
import { isTerminalStatus, type AnalysisRequest, type AnalysisStatus, type JobProgress, type JobStatus } from "@/lib/api/types";
import { readAnalysisParam, writeAnalysisParam } from "@/lib/url";

export type RunPhase = "idle" | "creating" | "running" | "terminal" | "error";

export interface RunState {
  phase: RunPhase;
  analysisId: string | null;
  jobStatus: JobStatus | null;
  progress: JobProgress | null;
  partial: Record<string, number | string | null> | null;
  liveIssues: EventIssue[];
  status: AnalysisStatus | null;
  error: ApiError | null;
  connection: ConnectionState | "idle";
  submittedRequest: AnalysisRequest | null;
  cancelling: boolean;
}

const INITIAL: RunState = {
  phase: "idle",
  analysisId: null,
  jobStatus: null,
  progress: null,
  partial: null,
  liveIssues: [],
  status: null,
  error: null,
  connection: "idle",
  submittedRequest: null,
  cancelling: false,
};

type Action =
  | { type: "create"; request: AnalysisRequest }
  | { type: "created"; id: string; jobStatus: JobStatus }
  | { type: "resume"; id: string }
  | { type: "event"; event: AnalysisEvent }
  | { type: "status"; status: AnalysisStatus }
  | { type: "connection"; state: ConnectionState }
  | { type: "cancelling" }
  | { type: "error"; error: ApiError; fatal: boolean }
  | { type: "reset" };

function reducer(state: RunState, action: Action): RunState {
  switch (action.type) {
    case "create":
      return { ...INITIAL, phase: "creating", submittedRequest: action.request };
    case "created":
      return { ...state, phase: "running", analysisId: action.id, jobStatus: action.jobStatus };
    case "resume":
      return { ...INITIAL, phase: "running", analysisId: action.id };
    case "event": {
      const e = action.event;
      if (state.analysisId && e.analysis_id !== state.analysisId) return state;
      return {
        ...state,
        jobStatus: e.status,
        progress: (e.progress as JobProgress | null | undefined) ?? state.progress,
        partial: e.partial ?? state.partial,
        liveIssues: e.type === "warning" && e.issue ? [...state.liveIssues, e.issue] : state.liveIssues,
      };
    }
    case "status": {
      const s = action.status;
      const terminal = isTerminalStatus(s.status);
      return {
        ...state,
        analysisId: s.analysis_id,
        status: s,
        jobStatus: s.status,
        progress: s.progress ?? state.progress,
        phase: terminal ? "terminal" : "running",
        cancelling: terminal ? false : state.cancelling || s.status === "CANCEL_REQUESTED",
        error: null,
      };
    }
    case "connection":
      return { ...state, connection: action.state };
    case "cancelling":
      return { ...state, cancelling: true };
    case "error":
      // Fatal: the run cannot continue from here (create/resume/status failed). A failed cancel
      // request is not fatal: the job keeps running and the stream stays open.
      return action.fatal
        ? { ...state, phase: "error", error: action.error, cancelling: false, connection: "closed" }
        : { ...state, error: action.error, cancelling: false };
    case "reset":
      return INITIAL;
  }
}

const GONE = new Set([401, 403, 404]);

export function useAnalysisRun() {
  const { api, eventSourceFactory } = useApiEnvironment();
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const streamRef = useRef<{ close: () => void } | null>(null);

  const closeStream = useCallback(() => {
    streamRef.current?.close();
    streamRef.current = null;
  }, []);

  const fetchStatus = useCallback(
    async (id: string): Promise<AnalysisStatus | null> => {
      try {
        const status = await api.getAnalysis(id);
        dispatch({ type: "status", status });
        return status;
      } catch (error) {
        if (error instanceof ApiError) dispatch({ type: "error", error, fatal: true });
        return null;
      }
    },
    [api],
  );

  const follow = useCallback(
    (id: string, lastEventId: number | null) => {
      closeStream();
      streamRef.current = openAnalysisStream(
        {
          url: api.eventsUrl(id),
          factory: eventSourceFactory,
          lastEventId,
          resolveClosed: async () => {
            try {
              const status = await api.getAnalysis(id);
              dispatch({ type: "status", status });
              return isTerminalStatus(status.status) ? "terminal" : "retry";
            } catch (error) {
              if (error instanceof ApiError && GONE.has(error.status)) {
                dispatch({ type: "error", error, fatal: true });
                return "stop";
              }
              return "retry";
            }
          },
        },
        {
          onEvent: (event) => dispatch({ type: "event", event }),
          onTerminal: () => {
            void fetchStatus(id);
          },
          onConnection: (connection) => dispatch({ type: "connection", state: connection }),
        },
      );
    },
    [api, eventSourceFactory, closeStream, fetchStatus],
  );

  const start = useCallback(
    async (request: AnalysisRequest) => {
      closeStream();
      dispatch({ type: "create", request });
      try {
        // A fresh key per click: retries of the same click are deduplicated by the server.
        const created = await api.createAnalysis(request, newIdempotencyKey());
        writeAnalysisParam(created.analysis_id);
        dispatch({ type: "created", id: created.analysis_id, jobStatus: created.status });
        if (isTerminalStatus(created.status)) void fetchStatus(created.analysis_id);
        else follow(created.analysis_id, null);
      } catch (error) {
        if (error instanceof ApiError) dispatch({ type: "error", error, fatal: true });
        else throw error;
      }
    },
    [api, closeStream, fetchStatus, follow],
  );

  const resume = useCallback(
    async (id: string) => {
      closeStream();
      dispatch({ type: "resume", id });
      const status = await fetchStatus(id);
      if (status && !isTerminalStatus(status.status)) follow(id, status.last_event_id ?? null);
      if (!status) writeAnalysisParam(null);
    },
    [closeStream, fetchStatus, follow],
  );

  const cancel = useCallback(async () => {
    if (!state.analysisId) return;
    dispatch({ type: "cancelling" });
    try {
      const status = await api.cancelAnalysis(state.analysisId);
      dispatch({ type: "status", status });
    } catch (error) {
      if (error instanceof ApiError) dispatch({ type: "error", error, fatal: false });
    }
  }, [api, state.analysisId]);

  const reset = useCallback(() => {
    closeStream();
    writeAnalysisParam(null);
    dispatch({ type: "reset" });
  }, [closeStream]);

  // Reconnect to ?analysis=<id> after a reload. Safe to run twice (StrictMode): a newer
  // follow() always closes the previous stream first.
  useEffect(() => {
    const id = readAnalysisParam();
    if (id) void resume(id);
    return closeStream;
  }, [resume, closeStream]);

  return { state, start, resume, cancel, reset, refresh: fetchStatus };
}
