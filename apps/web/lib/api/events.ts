// SSE payload contract. The types come from the generated OpenAPI schema (AnalysisEvent,
// EventType, JobProgress; `pnpm gen:api`); every received `data:` payload is still validated at
// runtime with the Zod mirror below (docs/research/stack-compat.md W8 option c), whose keys and
// output type are checked against the generated ones at compile time.
import { z } from "zod";

import type { EventType, JobStatus, Schemas, Severity } from "./types";

export type { EventType } from "./types";

export const EVENT_TYPES = [
  "progress",
  "partial_result",
  "warning",
  "needs_input",
  "completed",
  "failed",
  "cancelled",
] as const satisfies readonly EventType[];

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

/** The issue of an event: the generated Issue, but `code` stays a plain string so that an error
 * code newer than this build still shows its Korean message instead of dropping the event. */
export type EventIssue = z.infer<typeof eventIssueSchema>;
export type EventProgress = Schemas["JobProgress"];
/** The generated AnalysisEvent with the forward-compatible issue. */
export type AnalysisEvent = Omit<Schemas["AnalysisEvent"], "issue"> & { issue?: EventIssue | null };

// ---------------------------------------------------------------- contract parity (compile time)

type KeyParity<A, B> = [Exclude<keyof A, keyof B>, Exclude<keyof B, keyof A>] extends [never, never]
  ? true
  : { onlyInContract: Exclude<keyof A, keyof B>; onlyInMirror: Exclude<keyof B, keyof A> };
type Mirror<T extends z.ZodType> = z.infer<T>;

export const eventContractParity: {
  event: KeyParity<Schemas["AnalysisEvent"], Mirror<typeof analysisEventSchema>>;
  progress: KeyParity<Schemas["JobProgress"], Mirror<typeof jobProgressSchema>>;
  shard: KeyParity<Schemas["ShardProgress"], NonNullable<Mirror<typeof jobProgressSchema>["shard_progress"]>>;
  issue: KeyParity<Schemas["Issue"], EventIssue>;
} = { event: true, progress: true, shard: true, issue: true };

// What the mirror validates is a contract event (and its progress a contract JobProgress).
const eventAssignable: AnalysisEvent = null as unknown as Mirror<typeof analysisEventSchema>;
const progressAssignable: EventProgress = null as unknown as Mirror<typeof jobProgressSchema>;
void eventAssignable;
void progressAssignable;

// Every event type of the contract is listed in EVENT_TYPES.
const eventTypesComplete: [Exclude<EventType, (typeof EVENT_TYPES)[number]>] extends [never] ? true : never = true;
void eventTypesComplete;

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
