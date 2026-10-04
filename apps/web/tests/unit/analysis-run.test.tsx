import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, type ApiClient } from "@/lib/api/client";
import { ApiProvider } from "@/lib/api/context";
import { useAnalysisRun } from "@/lib/hooks/useAnalysisRun";

import { completedGrpoStatus, runningStatus } from "../fixtures/analysis-states";
import { grpoRequest } from "../fixtures/analysis-grpo";
import { FakeEventSource } from "../utils/fake-event-source";
import { makeEvent } from "../utils/events";
import { makeEnvironment } from "../utils/render";

const ID = completedGrpoStatus.analysis_id;

function setup(api: Partial<ApiClient>) {
  const environment = makeEnvironment(api);
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <ApiProvider value={environment}>{children}</ApiProvider>
    </QueryClientProvider>
  );
  return { ...renderHook(() => useAnalysisRun(), { wrapper }), environment };
}

beforeEach(() => {
  FakeEventSource.reset();
  window.history.replaceState(null, "", "/");
});
afterEach(() => window.history.replaceState(null, "", "/"));

describe("useAnalysisRun", () => {
  it("creates with an idempotency key, follows SSE and fetches the final status", async () => {
    const createAnalysis = vi.fn(async () => ({ analysis_id: ID, status: "QUEUED" as const, fingerprint: "f", created_at: "2026-10-04T12:00:00Z", reused: false }));
    const getAnalysis = vi.fn(async () => completedGrpoStatus);
    const { result } = setup({ createAnalysis, getAnalysis });

    await act(() => result.current.start(grpoRequest));
    expect(createAnalysis).toHaveBeenCalledWith(grpoRequest, expect.stringMatching(/^[0-9a-f-]{36}$/));
    expect(window.location.search).toBe(`?analysis=${ID}`);
    expect(result.current.state.phase).toBe("running");

    const es = FakeEventSource.latest();
    expect(es.url).toBe(`/api/v1/analyses/${ID}/events`);
    act(() => {
      es.open();
      es.emit("progress", makeEvent({ event_id: 1, analysis_id: ID, type: "progress", status: "TOKENIZING", progress: { stage: "TOKENIZING", processed_rows: 2_048, total_rows: null } }));
      es.emit("warning", makeEvent({ event_id: 2, analysis_id: ID, type: "warning", status: "TOKENIZING", issue: { code: "SCAN_FAILED_ROWS", severity: "warning", user_message: "실패 row 1건" } }));
    });
    expect(result.current.state.progress?.processed_rows).toBe(2_048);
    expect(result.current.state.progress?.total_rows).toBeNull();
    expect(result.current.state.liveIssues).toHaveLength(1);

    act(() => es.emit("completed", makeEvent({ event_id: 3, analysis_id: ID, type: "completed", status: "COMPLETED" })));
    await waitFor(() => expect(result.current.state.phase).toBe("terminal"));
    expect(getAnalysis).toHaveBeenCalledWith(ID);
    expect(result.current.state.status?.result?.analysis_id).toBe(ID);
    expect(es.closed).toBe(true);
  });

  it("uses a new idempotency key for every click", async () => {
    const createAnalysis = vi.fn(async () => ({ analysis_id: ID, status: "QUEUED" as const, fingerprint: "f", created_at: "", reused: false }));
    const { result } = setup({ createAnalysis, getAnalysis: vi.fn(async () => runningStatus) });
    await act(() => result.current.start(grpoRequest));
    await act(() => result.current.start(grpoRequest));
    const keys = createAnalysis.mock.calls.map((call) => (call as unknown as [unknown, string])[1]);
    expect(new Set(keys).size).toBe(2);
  });

  it("reconnects to ?analysis=<id> after a reload", async () => {
    window.history.replaceState(null, "", `/?analysis=${ID}`);
    const getAnalysis = vi.fn(async () => runningStatus);
    const { result } = setup({ getAnalysis });
    await waitFor(() => expect(result.current.state.jobStatus).toBe("TOKENIZING"));
    expect(result.current.state.progress?.processed_rows).toBe(1_536);
    // A fresh EventSource cannot send Last-Event-ID, so the resume point goes in ?after=.
    expect(FakeEventSource.latest().url).toBe(`/api/v1/analyses/${ID}/events?after=${runningStatus.last_event_id}`);
  });

  it("refetches the analysis on partial_result events (throttled)", async () => {
    vi.useFakeTimers();
    const createAnalysis = vi.fn(async () => ({ analysis_id: ID, status: "QUEUED" as const, fingerprint: "f", created_at: "", reused: false }));
    const getAnalysis = vi.fn(async () => runningStatus);
    const { result } = setup({ createAnalysis, getAnalysis });
    await act(() => result.current.start(grpoRequest));
    const es = FakeEventSource.latest();
    expect(es.url).toBe(`/api/v1/analyses/${ID}/events`);
    act(() => {
      es.emit("partial_result", makeEvent({ event_id: 1, analysis_id: ID, type: "partial_result", status: "TOKENIZING" }));
      es.emit("partial_result", makeEvent({ event_id: 2, analysis_id: ID, type: "partial_result", status: "TOKENIZING" }));
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(getAnalysis).toHaveBeenCalledTimes(1);
    act(() => es.emit("partial_result", makeEvent({ event_id: 3, analysis_id: ID, type: "partial_result", status: "TOKENIZING" })));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_100);
    });
    expect(getAnalysis).toHaveBeenCalledTimes(2);
    expect(result.current.state.phase).toBe("running");
    vi.useRealTimers();
  });

  it("drops the URL parameter when the analysis no longer exists", async () => {
    window.history.replaceState(null, "", "/?analysis=gone-1");
    const getAnalysis = vi.fn(async () => {
      throw new ApiError(404, { code: "NOT_FOUND", severity: "error", retryable: false, user_message: "없음" });
    });
    const { result } = setup({ getAnalysis });
    await waitFor(() => expect(result.current.state.error?.status).toBe(404));
    expect(window.location.search).toBe("");
    // The run is over: the page must not stay in a "running" state with disabled buttons.
    expect(result.current.state.phase).toBe("error");
  });

  it("keeps ?analysis= after a transient failure so a refresh can reconnect", async () => {
    window.history.replaceState(null, "", `/?analysis=${ID}`);
    const getAnalysis = vi.fn(async () => {
      throw new ApiError(0, { code: "INTERNAL_ERROR", severity: "error", retryable: true, user_message: "연결 실패" });
    });
    const { result } = setup({ getAnalysis });
    await waitFor(() => expect(result.current.state.phase).toBe("error"));
    expect(window.location.search).toBe(`?analysis=${ID}`);
  });

  it("keeps running when only the cancel request fails", async () => {
    const createAnalysis = vi.fn(async () => ({ analysis_id: ID, status: "QUEUED" as const, fingerprint: "f", created_at: "", reused: false }));
    const cancelAnalysis = vi.fn(async () => {
      throw new ApiError(409, { code: "CONFLICTING_OPTIONS", severity: "error", retryable: false, user_message: "이미 끝난 분석입니다." });
    });
    const { result } = setup({ createAnalysis, cancelAnalysis, getAnalysis: vi.fn(async () => runningStatus) });
    await act(() => result.current.start(grpoRequest));
    await act(() => result.current.cancel());
    expect(result.current.state.phase).toBe("running");
    expect(result.current.state.error?.status).toBe(409);
    expect(result.current.state.cancelling).toBe(false);
  });

  it("requests cancellation", async () => {
    const createAnalysis = vi.fn(async () => ({ analysis_id: ID, status: "QUEUED" as const, fingerprint: "f", created_at: "", reused: false }));
    const cancelAnalysis = vi.fn(async () => ({ ...runningStatus, status: "CANCEL_REQUESTED" as const }));
    const { result } = setup({ createAnalysis, cancelAnalysis, getAnalysis: vi.fn(async () => runningStatus) });
    await act(() => result.current.start(grpoRequest));
    await act(() => result.current.cancel());
    expect(cancelAnalysis).toHaveBeenCalledWith(ID);
    expect(result.current.state.jobStatus).toBe("CANCEL_REQUESTED");
    expect(result.current.state.cancelling).toBe(true);
  });
});
