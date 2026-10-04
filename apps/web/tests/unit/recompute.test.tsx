import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api/client";
import { ApiProvider } from "@/lib/api/context";
import type { AnalysisRequest, ScenarioResponse } from "@/lib/api/types";
import { toAnalysisRequest } from "@/lib/form/convert";
import { requestFingerprint } from "@/lib/form/fingerprint";
import { useRecompute, type RecomputeInput } from "@/lib/hooks/useRecompute";
import { EXAMPLE_FORM_VALUES, type FormValues } from "@/lib/form/values";

import { dpoResult, sftResult } from "../fixtures/analysis-dpo-sft";
import { makeEnvironment } from "../utils/render";

const base = sftResult;
const baseRequest = base.requested_config;

function requestWith(overrides: Partial<FormValues>): AnalysisRequest {
  return toAnalysisRequest({ ...EXAMPLE_FORM_VALUES, objective: "sft", microbatch: "1", accumulation: "8", ...overrides });
}

function setup(scenarios: ApiClient["scenarios"]) {
  const environment = makeEnvironment({ scenarios });
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <ApiProvider value={environment}>{children}</ApiProvider>
    </QueryClientProvider>
  );
  const initial: RecomputeInput = {
    analysisId: base.analysis_id,
    baseResult: base,
    baseRequest,
    currentRequest: baseRequest,
    enabled: true,
  };
  return renderHook((props: RecomputeInput) => useRecompute(props), { wrapper, initialProps: initial });
}

function answer(request: AnalysisRequest, result = dpoResult): ScenarioResponse {
  return {
    fingerprint: `server-${requestFingerprint(request)}`,
    client_fingerprint: requestFingerprint(request),
    requires_reanalysis: false,
    reanalysis_reasons: [],
    result,
  };
}

afterEach(() => vi.useRealTimers());

describe("useRecompute", () => {
  it("shows the base result as current when nothing changed", () => {
    const scenarios = vi.fn();
    const { result } = setup(scenarios);
    expect(result.current.mode).toBe("current");
    expect(result.current.display).toBe(base);
    expect(result.current.stale).toBe(false);
    expect(scenarios).not.toHaveBeenCalled();
  });

  it("debounces light changes, marks stale values and sends the client fingerprint", async () => {
    vi.useFakeTimers();
    const scenarios = vi.fn(async (_id: string, body: { request: AnalysisRequest }) => answer(body.request));
    const { result, rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    const r32 = requestWith({ loraR: "32" });
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: r32, enabled: true });
    expect(result.current.mode).toBe("pending");
    expect(result.current.stale).toBe(true);
    expect(result.current.display).toBe(base);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(299);
    });
    expect(scenarios).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(scenarios).toHaveBeenCalledTimes(1);
    expect(scenarios.mock.calls[0]![1]).toMatchObject({ client_fingerprint: requestFingerprint(r32) });
    await act(async () => {
      await vi.runOnlyPendingTimersAsync();
    });
    expect(result.current.mode).toBe("current");
    expect(result.current.display).toBe(dpoResult);
    expect(result.current.stale).toBe(false);
    expect(result.current.history[0]?.changes).toEqual(["LoRA r 16 → 32"]);
    // Exports of what is shown need the request the displayed scenario was computed for.
    expect(result.current.scenario).toEqual({ request: r32, changes: ["LoRA r 16 → 32"] });

    // Back to the base settings: the stored analysis is shown again, not a scenario.
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: baseRequest, enabled: true });
    expect(result.current.display).toBe(base);
    expect(result.current.scenario).toBeNull();
  });

  it("keeps the shown scenario's request while a newer setting is still pending", async () => {
    vi.useFakeTimers();
    const scenarios = vi.fn(async (_id: string, body: { request: AnalysisRequest }) => answer(body.request));
    const { result, rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    const r32 = requestWith({ loraR: "32" });
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: r32, enabled: true });
    await act(async () => {
      await vi.runOnlyPendingTimersAsync();
    });
    expect(result.current.scenario?.request).toEqual(r32);
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: requestWith({ objective: "dpo" }), enabled: true });
    // Re-analysis needed: the "이전 설정" numbers on screen are still the r=32 scenario.
    expect(result.current.mode).toBe("reanalysis");
    expect(result.current.stale).toBe(true);
    expect(result.current.display).toBe(dpoResult);
    expect(result.current.scenario?.request).toEqual(r32);
  });

  it("keeps the same view object while nothing changed (memoized result components)", () => {
    const { result, rerender } = setup(vi.fn());
    const first = result.current;
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: baseRequest, enabled: true });
    expect(result.current).toBe(first);
    expect(result.current.history).toBe(first.history);
    expect(result.current.retry).toBe(first.retry);
  });

  it("names an unreadable scenario answer instead of calling it empty", async () => {
    const scenarios = vi.fn(async () => {
      throw new SyntaxError("Unexpected token <");
    });
    const { result, rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: requestWith({ loraR: "8" }), enabled: true });
    await waitFor(() => expect(result.current.mode).toBe("error"));
    expect(result.current.error?.issue.user_message).toMatch(/처리하지 못했습니다/);
  });

  it("sends only the latest of rapid edits", async () => {
    vi.useFakeTimers();
    const scenarios = vi.fn(async (_id: string, body: { request: AnalysisRequest }) => answer(body.request));
    const { rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    for (const r of ["24", "32", "64"]) {
      rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: requestWith({ loraR: r }), enabled: true });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(100);
      });
    }
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(scenarios).toHaveBeenCalledTimes(1);
    expect((scenarios.mock.calls[0]![1] as { request: AnalysisRequest }).request.training.lora?.r).toBe(64);
  });

  it("drops a late response that belongs to an older setting", async () => {
    const pending: Array<(value: ScenarioResponse) => void> = [];
    const scenarios = vi.fn(
      (_id: string, body: { request: AnalysisRequest }) =>
        new Promise<ScenarioResponse>((resolve) => pending.push(() => resolve(answer(body.request, body.request.training.lora?.r === 32 ? dpoResult : sftResult)))),
    );
    const { result, rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    const r32 = requestWith({ loraR: "32" });
    const r64 = requestWith({ loraR: "64" });
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: r32, enabled: true });
    await waitFor(() => expect(scenarios).toHaveBeenCalledTimes(1));
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: r64, enabled: true });
    await waitFor(() => expect(scenarios).toHaveBeenCalledTimes(2));
    // The newer request answers first, then the stale one arrives late.
    await act(async () => pending[1]!(undefined as never));
    await waitFor(() => expect(result.current.mode).toBe("current"));
    const shown = result.current.display;
    await act(async () => pending[0]!(undefined as never));
    expect(result.current.display).toBe(shown);
    expect(result.current.history).toHaveLength(1);
  });

  it("asks for re-analysis on preprocessing changes without calling the server", () => {
    const scenarios = vi.fn();
    const { result, rerender } = setup(scenarios);
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: requestWith({ objective: "dpo" }), enabled: true });
    expect(result.current.mode).toBe("reanalysis");
    expect(result.current.reanalysisReasons).toEqual(["학습 방식 변경 (SFT → DPO)"]);
    expect(result.current.stale).toBe(true);
    expect(scenarios).not.toHaveBeenCalled();
  });

  it("follows the server when it requires re-analysis", async () => {
    const scenarios = vi.fn(async (_id: string, body: { request: AnalysisRequest }) => ({
      ...answer(body.request),
      requires_reanalysis: true,
      reanalysis_reasons: ["attention kernel 변경에 필요한 mask 정보가 없습니다."],
      result: null,
    }));
    const { result, rerender } = setup(scenarios as unknown as ApiClient["scenarios"]);
    rerender({ analysisId: base.analysis_id, baseResult: base, baseRequest, currentRequest: requestWith({ attentionBackend: "eager" }), enabled: true });
    await waitFor(() => expect(result.current.mode).toBe("reanalysis"));
    expect(result.current.reanalysisReasons).toEqual(["attention kernel 변경에 필요한 mask 정보가 없습니다."]);
  });
});
