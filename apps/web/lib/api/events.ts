// SSE payload contract. FastAPI's OpenAPI output leaves text/event-stream untyped, so this Zod
// schema mirrors packages/estimator/src/vramforge_estimator/schemas/events.py (`AnalysisEvent`)
// and validates every received event (docs/research/stack-compat.md W8 option c).
import { z } from "zod";

import type { JobStatus, Severity } from "./types";

export const EVENT_TYPES = [
  "progress",
  "partial_result",
  "warning",
  "needs_input",
  "completed",
  "failed",
  "cancelled",
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

/** schemas/events.py TERMINAL_EVENT_TYPES */
export const TERMINAL_EVENT_TYPES: ReadonlySet<EventType> = new Set<EventType>([
  "needs_input",
  "completed",
  "failed",
  "cancelled",
]);

const JOB_STATUSES = [
  "QUEUED",
  "RESOLVING",
  "INSPECTING",
  "NEEDS_INPUT",
  "TOKENIZING",
  "VALIDATING_DATA",
  "PLANNING_BATCHES",
  "ESTIMATING",
  "COMPLETED",
  "CANCEL_REQUESTED",
  "CANCELLED",
  "FAILED",
  "PARTIAL",
] as const satisfies readonly JobStatus[];

// Compile-time guard: every contract JobStatus is listed above.
const jobStatusesComplete: [Exclude<JobStatus, (typeof JOB_STATUSES)[number]>] extends [never]
  ? true
  : never = true;
void jobStatusesComplete;

const SEVERITIES = ["info", "warning", "error"] as const satisfies readonly Severity[];

export const eventIssueSchema = z.object({
  code: z.string(),
  severity: z.enum(SEVERITIES),
  stage: z.string().nullish(),
  retryable: z.boolean().optional(),
  user_message: z.string(),
  technical_detail_ref: z.string().nullish(),
  affected_component: z.string().nullish(),
  details: z.record(z.string(), z.unknown()).optional(),
});

export const jobProgressSchema = z.object({
  stage: z.enum(JOB_STATUSES),
  processed_rows: z.number().int().nullish(),
  total_rows: z.number().int().nullish(),
  shard_progress: z
    .object({ completed: z.number().int(), total: z.number().int().nullish() })
    .nullish(),
  message_code: z.string().nullish(),
  message: z.string().nullish(),
});

export const analysisEventSchema = z.object({
  event_id: z.number().int(),
  analysis_id: z.string(),
  type: z.enum(EVENT_TYPES),
  status: z.enum(JOB_STATUSES),
  progress: jobProgressSchema.nullish(),
  fingerprint: z.string(),
  issue: eventIssueSchema.nullish(),
  partial: z.record(z.string(), z.union([z.number(), z.string(), z.null()])).nullish(),
  timestamp: z.string(),
});

export type AnalysisEvent = z.infer<typeof analysisEventSchema>;
export type EventIssue = z.infer<typeof eventIssueSchema>;
export type EventProgress = z.infer<typeof jobProgressSchema>;

/** Parses one SSE `data:` payload; malformed or unknown-shaped events are dropped. */
export function parseEventData(data: unknown): AnalysisEvent | null {
  if (typeof data !== "string") return null;
  let json: unknown;
  try {
    json = JSON.parse(data);
  } catch {
    return null;
  }
  const parsed = analysisEventSchema.safeParse(json);
  return parsed.success ? parsed.data : null;
}
