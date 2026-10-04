// SSE payload contract. FastAPI's OpenAPI output leaves text/event-stream untyped, so this Zod
// schema mirrors packages/estimator/src/vramforge_estimator/schemas/events.py (`AnalysisEvent`)
// and validates every received event (docs/research/stack-compat.md W8 option c). The parts that
// are in the generated contract (JobProgress, ShardProgress, Issue) are checked against it at
// compile time below; AnalysisEvent and EventType are checked as soon as the OpenAPI document
// lists them in components (`pnpm gen:api`).
import { z } from "zod";

import type { JobStatus, Schemas, Severity } from "./types";

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
/** The generated contract type: the validated payload is assignable to it (checked below). */
export type EventProgress = Schemas["JobProgress"];

// ---------------------------------------------------------------- contract parity (compile time)

type KeyParity<A, B> = [Exclude<keyof A, keyof B>, Exclude<keyof B, keyof A>] extends [never, never]
  ? true
  : { onlyInContract: Exclude<keyof A, keyof B>; onlyInMirror: Exclude<keyof B, keyof A> };
/** A component of the generated schema, or never while the OpenAPI document lacks it. */
type Generated<K extends string> = Schemas extends Record<K, infer T> ? T : never;
type ParityOnceGenerated<T, Mirror> = [T] extends [never] ? true : KeyParity<T, Mirror>;

export const eventContractParity: {
  progress: KeyParity<Schemas["JobProgress"], z.infer<typeof jobProgressSchema>>;
  shard: KeyParity<Schemas["ShardProgress"], NonNullable<z.infer<typeof jobProgressSchema>["shard_progress"]>>;
  // Same keys as Issue; `code` stays a plain string so an unknown new code never drops an event.
  issue: KeyParity<Schemas["Issue"], EventIssue>;
  event: ParityOnceGenerated<Generated<"AnalysisEvent">, AnalysisEvent>;
} = { progress: true, shard: true, issue: true, event: true };

// Validated progress is usable wherever the generated JobProgress type is expected.
const progressAssignable: EventProgress = null as unknown as z.infer<typeof jobProgressSchema>;
void progressAssignable;

// Every event type of the contract is listed in EVENT_TYPES (once EventType is generated).
const eventTypesComplete: [Exclude<Generated<"EventType">, EventType>] extends [never] ? true : never = true;
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
