"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { ApiError, newIdempotencyKey } from "@/lib/api/client";
import { useApiEnvironment } from "@/lib/api/context";
import type { AnalysisEvent, EventIssue } from "@/lib/api/events";
import { openAnalysisStream, type ConnectionState } from "@/lib/api/stream";
import {
  isTerminalStatus,
  type AnalysisRequest,
  type AnalysisStatus,
  type Issue,
  type JobProgress,
  type JobStatus,
} from "@/lib/api/types";
import { readAnalysisParam, writeAnalysisParam } from "@/lib/url";

export type RunPhase = "idle" | "creating" | "running" | "terminal" | "error";

export interface RunState {
  phase: RunPhase;
  analysisId: string | null;
  /** Latest known job status: from events while running, from GET /analyses/{id} at the end. */
  jobStatus: JobStatus | null;
  progress: JobProgress | null;
  partial: Record<string, number | string | null> | null;
  liveIssues: EventIssue[];
  /** The issue of the event that ended the stream; shown until (or unless) the final GET answers. */
  endIssue: EventIssue | null;
  status: AnalysisStatus | null;
  error: ApiError | null;
  connection: ConnectionState | "idle";
  submittedRequest: AnalysisRequest | null;
  cancelling: boolean;
}

export const INITIAL_RUN_STATE: RunState = {
  phase: "idle",
  analysisId: null,
  jobStatus: null,
  progress: null,
  partial: null,
  liveIssues: [],
  endIssue: null,
  status: null,
  error: null,
  connection: "idle",
  submittedRequest: null,
  cancelling: false,
};

const INITIAL = INITIAL_RUN_STATE;

/**
 * Work is still going on: creating, or running without a terminal status yet. After a terminal
 * event the run waits only for its final GET, so cancel and re-run are no longer blocked.
 */
export function isRunActive(run: Pick<RunState, "phase" | "jobStatus">): boolean {
  return run.phase === "creating" || (run.phase === "running" && !isTerminalStatus(run.jobStatus));
}

type Action =
  | { type: "create"; request: AnalysisRequest }
  | { type: "created"; id: string; jobStatus: JobStatus }
  | { type: "resume"; id: string }
  | { type: "event"; event: AnalysisEvent }
  | { type: "status"; status: AnalysisStatus }
  | { type: "connection"; state: ConnectionState }
  | { type: "cancelling" }
  /** `id`: the analysis the failed request was about; errors for another analysis are dropped. */
  | { type: "error"; error: ApiError; fatal: boolean; id?: string }
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
      if (e.analysis_id !== state.analysisId || state.phase === "terminal") return state;
      // The status says what happened (a `failed` event may carry PARTIAL); the type only names
      // the frame. The final GET then replaces this with the stored state.
      const ended = isTerminalStatus(e.status);
      return {
        ...state,
        jobStatus: e.status,
        progress: e.progress ?? state.progress,
        partial: e.partial ?? state.partial,
        liveIssues: e.type === "warning" && e.issue ? [...state.liveIssues, e.issue] : state.liveIssues,
        endIssue: ended ? (e.issue ?? null) : state.endIssue,
        cancelling: ended ? false : state.cancelling,
      };
    }
    case "status": {
      const s = action.status;
      // A late answer for another analysis (an earlier run, a request still in flight when a new
      // run started) never replaces the current run.
      if (s.analysis_id !== state.analysisId) return state;
      const terminal = isTerminalStatus(s.status);
      // A running snapshot fetched before the final status cannot reopen a finished run.
      if (!terminal && state.phase === "terminal") return state;
      return {
        ...state,
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
      if (action.id !== undefined && action.id !== state.analysisId) return state;
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

const UNREADABLE_ANSWER: Issue = {
  code: "INTERNAL_ERROR",
  severity: "error",
  retryable: true,
  user_message: "서버 응답을 처리하지 못했습니다. 잠시 후 다시 시도하세요.",
};

/** Every failure ends up as an ApiError so the run never stays stuck in a pending phase. */
function asApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError(0, UNREADABLE_ANSWER);
}

/** `partial_result` asks clients to refetch the analysis; refetch at most this often. */
const PARTIAL_REFRESH_MS = 2_000;

export function useAnalysisRun() {
  const { api, eventSourceFactory } = useApiEnvironment();
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const streamRef = useRef<{ close: () => void } | null>(null);
  const partialRefresh = useRef<{ last: number; timer: ReturnType<typeof setTimeout> | null }>({ last: 0, timer: null });

  const closeStream = useCallback(() => {
    streamRef.current?.close();
    streamRef.current = null;
    if (partialRefresh.current.timer) clearTimeout(partialRefresh.current.timer);
    partialRefresh.current.timer = null;
  }, []);

  /** Throttled, non-fatal refetch of the running analysis for its partial result. */
  const refreshPartial = useCallback(
    (id: string) => {
      const slot = partialRefresh.current;
      if (slot.timer) return;
      const run = () => {
        slot.timer = null;
        slot.last = Date.now();
        api.getAnalysis(id).then(
          (status) => {
            if (!isTerminalStatus(status.status)) dispatch({ type: "status", status });
          },
          () => undefined,
        );
      };
      const wait = Math.max(0, slot.last + PARTIAL_REFRESH_MS - Date.now());
      slot.timer = setTimeout(run, wait);
    },
    [api],
  );

  const fetchStatusOrError = useCallback(
    async (id: string): Promise<AnalysisStatus | ApiError | null> => {
      try {
        const status = await api.getAnalysis(id);
        dispatch({ type: "status", status });
        return status;
      } catch (error) {
        const apiError = asApiError(error);
        dispatch({ type: "error", error: apiError, fatal: true, id });
        return apiError;
      }
    },
    [api],
  );

  const fetchStatus = useCallback(
    async (id: string): Promise<AnalysisStatus | null> => {
      const outcome = await fetchStatusOrError(id);
      return outcome instanceof ApiError ? null : outcome;
    },
    [fetchStatusOrError],
  );

  const followRef = useRef<(id: string, lastEventId: number | null) => void>(() => undefined);

  /**
   * The stream ended. The stored status decides what the page shows (an event's type alone never
   * does); should the job in fact still be running, follow it again after the last seen event.
   */
  const settle = useCallback(
    async (id: string, lastSeen: number | null) => {
      const outcome = await fetchStatusOrError(id);
      if (outcome == null || outcome instanceof ApiError || isTerminalStatus(outcome.status)) return;
      const after = Math.max(outcome.last_event_id ?? -1, lastSeen ?? -1);
      followRef.current(id, after >= 0 ? after : null);
    },
    [fetchStatusOrError],
  );

  const follow = useCallback(
    (id: string, lastEventId: number | null) => {
      closeStream();
      streamRef.current = openAnalysisStream(
        {
          url: (after) => api.eventsUrl(id, after),
          factory: eventSourceFactory,
          lastEventId,
          resolveClosed: async () => {
            try {
              const status = await api.getAnalysis(id);
              dispatch({ type: "status", status });
              return isTerminalStatus(status.status) ? "terminal" : "retry";
            } catch (error) {
              if (error instanceof ApiError && GONE.has(error.status)) {
                dispatch({ type: "error", error, fatal: true, id });
                return "stop";
              }
              return "retry";
            }
          },
        },
        {
          onEvent: (event) => {
            dispatch({ type: "event", event });
            if (event.type === "partial_result") refreshPartial(id);
          },
          onTerminal: (event) => {
            void settle(id, event?.event_id ?? lastEventId);
          },
          onConnection: (connection) => dispatch({ type: "connection", state: connection }),
        },
      );
    },
    [api, eventSourceFactory, closeStream, settle, refreshPartial],
  );
  useEffect(() => {
    followRef.current = follow;
  }, [follow]);

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
        dispatch({ type: "error", error: asApiError(error), fatal: true });
      }
    },
    [api, closeStream, fetchStatus, follow],
  );

  const resume = useCallback(
    async (id: string) => {
      closeStream();
      dispatch({ type: "resume", id });
      const outcome = await fetchStatusOrError(id);
      if (outcome instanceof ApiError) {
        // Gone or not ours: forget the id. Transient failures keep it so a refresh can reconnect.
        if (outcome.status === 403 || outcome.status === 404) writeAnalysisParam(null);
        return;
      }
      if (outcome && !isTerminalStatus(outcome.status)) follow(id, outcome.last_event_id ?? null);
    },
    [closeStream, fetchStatusOrError, follow],
  );

  const cancel = useCallback(async () => {
    const id = state.analysisId;
    if (!id) return;
    dispatch({ type: "cancelling" });
    try {
      const status = await api.cancelAnalysis(id);
      dispatch({ type: "status", status });
    } catch (error) {
      dispatch({ type: "error", error: asApiError(error), fatal: false, id });
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
