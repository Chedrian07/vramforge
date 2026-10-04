// Fixture-backed API for local UI development: NEXT_PUBLIC_VF_DEV_MOCKS=1 pnpm dev.
// Reached only through the `@vf/dev-mocks` alias (next.config.ts), so production bundles get
// lib/dev-mocks/disabled.ts instead; scripts/check-no-mocks.mjs verifies it after every build.
// Demo switches (dev only): a dataset reference containing "ambiguous" / "needs-input" / "fail" /
// "partial", or a model reference containing "missing" / "unknown", selects the matching state.
import { ApiError, type ApiClient } from "@/lib/api/client";
import type { AnalysisEvent } from "@/lib/api/events";
import type { EventSourceFactory, EventSourceLike } from "@/lib/api/stream";
import type { AnalysisRequest, AnalysisResult, AnalysisStatus, JobProgress, JobStatus } from "@/lib/api/types";
import { requestFingerprint } from "@/lib/form/fingerprint";

import { dpoResult, sftResult } from "../../tests/fixtures/analysis-dpo-sft";
import { grpoExplicitBudgetResult, grpoResult } from "../../tests/fixtures/analysis-grpo";
import { cancelledStatus, failedStatus, needsInputStatus, partialStatus, unknownPeakResult } from "../../tests/fixtures/analysis-states";
import { fixtureId } from "../../tests/fixtures/builders";
import {
  ambiguousDatasetInspection,
  backendProfiles,
  datasetInspection,
  localRoots,
  missingModelInspection,
  modelInspection,
  uploadResponse,
} from "../../tests/fixtures/sources";
import type { DevMocks } from "./types";

const STEP_MS = 350;
const TOTAL_ROWS = 4_656;

interface MockRun {
  id: string;
  request: AnalysisRequest;
  final: AnalysisStatus;
  events: AnalysisEvent[];
  startedAt: number;
  cancelledAt: number | null;
}

const runs = new Map<string, MockRun>();
let counter = 0;

function now(): string {
  return new Date().toISOString();
}

function pickFinal(id: string, request: AnalysisRequest): AnalysisStatus {
  const dataset = request.dataset.reference.toLowerCase();
  const model = request.model.reference.toLowerCase();
  const stamp = { analysis_id: id, created_at: now(), updated_at: now(), finished_at: now() };
  const withRequest = (result: AnalysisResult): AnalysisResult => ({ ...result, analysis_id: id, requested_config: request });
  if (dataset.includes("needs-input")) return { ...needsInputStatus, ...stamp, result: withRequest(needsInputStatus.result!) };
  if (dataset.includes("fail")) return { ...failedStatus, ...stamp, result: withRequest(failedStatus.result!) };
  if (dataset.includes("partial")) return { ...partialStatus, ...stamp, result: withRequest(partialStatus.result!) };
  let result: AnalysisResult;
  if (model.includes("unknown")) result = unknownPeakResult;
  else if (request.training.objective === "dpo") result = dpoResult;
  else if (request.training.objective === "sft") result = sftResult;
  else result = request.grpo?.completion_budget != null ? grpoExplicitBudgetResult : grpoResult;
  return {
    status: "COMPLETED",
    fingerprint: fixtureId(`fp-${id}`),
    progress: { stage: "COMPLETED", processed_rows: TOTAL_ROWS, total_rows: TOTAL_ROWS, shard_progress: { completed: 1, total: 1 } },
    ...stamp,
    last_event_id: null,
    result: withRequest(result),
    error: null,
  };
}

function script(run: Omit<MockRun, "events">): AnalysisEvent[] {
  const events: AnalysisEvent[] = [];
  const push = (type: AnalysisEvent["type"], status: JobStatus, progress: JobProgress | null, partial: AnalysisEvent["partial"] = null) =>
    events.push({ event_id: events.length + 1, analysis_id: run.id, type, status, progress: progress as AnalysisEvent["progress"], fingerprint: run.final.fingerprint, issue: null, partial, timestamp: now() });
  push("progress", "RESOLVING", { stage: "RESOLVING", message: "소스 revision 고정 중" });
  push("progress", "INSPECTING", { stage: "INSPECTING", message: "config · tokenizer · safetensors header 확인" });
  if (run.final.status === "FAILED") {
    events.push({ ...events.at(-1)!, event_id: events.length + 1, type: "failed", status: "FAILED", issue: run.final.error ?? null });
    return events;
  }
  if (run.final.status === "NEEDS_INPUT") {
    events.push({ ...events.at(-1)!, event_id: events.length + 1, type: "needs_input", status: "NEEDS_INPUT" });
    return events;
  }
  for (let rows = 0; rows <= TOTAL_ROWS; rows += 1_024) {
    const known = rows >= 2_048;
    push("partial_result", "TOKENIZING", { stage: "TOKENIZING", processed_rows: rows, total_rows: known ? TOTAL_ROWS : null, shard_progress: { completed: 0, total: known ? 1 : null } }, { max_tokens: Math.min(268 + rows / 2, 2_272), rows_failed: 0 });
    if (run.final.status === "PARTIAL" && rows >= 2_048) {
      events.push({ ...events.at(-1)!, event_id: events.length + 1, type: "completed", status: "PARTIAL" });
      return events;
    }
  }
  push("progress", "VALIDATING_DATA", { stage: "VALIDATING_DATA", processed_rows: TOTAL_ROWS, total_rows: TOTAL_ROWS });
  push("progress", "PLANNING_BATCHES", { stage: "PLANNING_BATCHES" });
  push("progress", "ESTIMATING", { stage: "ESTIMATING" });
  push("completed", "COMPLETED", run.final.progress ?? null);
  return events;
}

function elapsedEvents(run: MockRun): AnalysisEvent[] {
  const until = run.cancelledAt ?? Date.now();
  const count = Math.floor((until - run.startedAt) / STEP_MS) + 1;
  return run.events.slice(0, Math.max(0, count));
}

function statusOf(run: MockRun): AnalysisStatus {
  if (run.cancelledAt != null) {
    return { ...cancelledStatus, analysis_id: run.id, result: cancelledStatus.result ? { ...cancelledStatus.result, analysis_id: run.id, requested_config: run.request } : null };
  }
  const seen = elapsedEvents(run);
  const last = seen.at(-1);
  if (last && ["completed", "failed", "needs_input", "cancelled"].includes(last.type)) return { ...run.final, last_event_id: last.event_id };
  return {
    analysis_id: run.id,
    status: last?.status ?? "QUEUED",
    fingerprint: run.final.fingerprint,
    progress: [...seen].reverse().find((e) => e.progress)?.progress ?? null,
    created_at: run.final.created_at,
    updated_at: now(),
    finished_at: null,
    last_event_id: last?.event_id ?? null,
    result: null,
    error: null,
  } as AnalysisStatus;
}

class MockEventSource implements EventSourceLike {
  readyState = 0;
  onopen: ((event: Event) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  private listeners = new Map<string, Array<(event: MessageEvent) => void>>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private sent = 0;

  constructor(private readonly run: MockRun | undefined) {
    setTimeout(() => this.start(), 0);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  close(): void {
    this.readyState = 2;
    if (this.timer) clearInterval(this.timer);
  }

  private start() {
    if (!this.run) {
      this.readyState = 2;
      this.onerror?.(new Event("error"));
      return;
    }
    this.readyState = 1;
    this.onopen?.(new Event("open"));
    this.timer = setInterval(() => this.tick(), STEP_MS / 2);
    this.tick();
  }

  private tick() {
    const run = this.run!;
    if (run.cancelledAt != null) {
      this.dispatch({ ...run.events[0]!, event_id: 10_000, type: "cancelled", status: "CANCELLED", progress: null, issue: null, partial: null, timestamp: now() });
      this.close();
      return;
    }
    const due = elapsedEvents(run);
    while (this.sent < due.length) this.dispatch(due[this.sent++]!);
    if (this.sent >= run.events.length) this.close();
  }

  private dispatch(event: AnalysisEvent) {
    const message = new MessageEvent(event.type, { data: JSON.stringify(event), lastEventId: String(event.event_id) });
    for (const listener of this.listeners.get(event.type) ?? []) listener(message);
  }
}

function notFound(): ApiError {
  return new ApiError(404, { code: "NOT_FOUND", severity: "error", retryable: false, user_message: "요청한 분석을 찾을 수 없습니다 (개발용 mock)." });
}

const delay = <T,>(value: T, ms = 250) => new Promise<T>((resolve) => setTimeout(() => resolve(value), ms));

const api: ApiClient = {
  inspect: async (body) => {
    const model = body.model ? (body.model.reference.toLowerCase().includes("missing") ? missingModelInspection : modelInspection) : null;
    const dataset = body.dataset ? (body.dataset.reference.toLowerCase().includes("ambiguous") ? ambiguousDatasetInspection : datasetInspection) : null;
    return delay({ model, dataset });
  },
  upload: async (file) => delay({ ...uploadResponse, filename: file.name, size_bytes: file.size }, 400),
  backendProfiles: async () => delay(backendProfiles, 50),
  localRoots: async () => delay(localRoots, 50),
  createAnalysis: async (request) => {
    counter += 1;
    const id = fixtureId(`run-${counter}`);
    const base = { id, request, final: pickFinal(id, request), startedAt: Date.now(), cancelledAt: null };
    runs.set(id, { ...base, events: script(base) });
    return delay({ analysis_id: id, status: "QUEUED" as const, fingerprint: base.final.fingerprint, created_at: now(), reused: false }, 150);
  },
  getAnalysis: async (id) => {
    const run = runs.get(id);
    if (!run) throw notFound();
    return delay(statusOf(run), 80);
  },
  cancelAnalysis: async (id) => {
    const run = runs.get(id);
    if (!run) throw notFound();
    run.cancelledAt = Date.now();
    return delay({ ...statusOf(run), status: "CANCEL_REQUESTED" as const }, 80);
  },
  scenarios: async (id, body) => {
    const run = runs.get(id);
    if (!run) throw notFound();
    const base = run.final.result;
    if (!base) throw notFound();
    const result: AnalysisResult =
      body.request.training.objective === "grpo" && body.request.grpo?.completion_budget != null
        ? { ...grpoExplicitBudgetResult, analysis_id: id, requested_config: body.request }
        : { ...base, requested_config: body.request };
    return delay(
      { fingerprint: fixtureId(`scenario-${requestFingerprint(body.request)}`), client_fingerprint: body.client_fingerprint ?? null, requires_reanalysis: false, reanalysis_reasons: [], result },
      400,
    );
  },
  createSession: async () => delay(undefined, 100),
  eventsUrl: (id) => `vf-dev-mock://events/${encodeURIComponent(id)}`,
  exportUrl: (id, format) => `/api/v1/analyses/${encodeURIComponent(id)}/export?format=${encodeURIComponent(format)}`,
};

const eventSourceFactory: EventSourceFactory = (url) => {
  const id = decodeURIComponent(url.split("/").at(-1) ?? "");
  return new MockEventSource(runs.get(id));
};

export const devMocks: DevMocks | null =
  process.env.NODE_ENV === "production" ? null : { api, eventSourceFactory, label: "NEXT_PUBLIC_VF_DEV_MOCKS=1" };
