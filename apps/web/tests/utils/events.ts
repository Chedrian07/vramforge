import type { AnalysisEvent } from "@/lib/api/events";

let counter = 0;

/** Builds a contract-shaped SSE payload (schemas/events.py AnalysisEvent). */
export function makeEvent(partial: Partial<AnalysisEvent> & Pick<AnalysisEvent, "type" | "status">): AnalysisEvent {
  counter += 1;
  return {
    event_id: partial.event_id ?? counter,
    analysis_id: partial.analysis_id ?? "vf-fixture-analysis",
    fingerprint: partial.fingerprint ?? "vf-fixture-fp",
    timestamp: partial.timestamp ?? "2026-10-04T12:00:00Z",
    progress: partial.progress ?? null,
    issue: partial.issue ?? null,
    partial: partial.partial ?? null,
    ...partial,
  };
}
