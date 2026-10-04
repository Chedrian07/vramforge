// Readable aliases over the generated OpenAPI types (lib/api/schema.d.ts, `pnpm gen:api`).
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type ApiPath = keyof paths;

export type AnalysisRequest = Schemas["AnalysisRequest"];
export type AnalysisResult = Schemas["AnalysisResult"];
export type AnalysisStatus = Schemas["AnalysisStatus"];
export type AnalysisCreated = Schemas["AnalysisCreated"];
export type ScenarioRequest = Schemas["ScenarioRequest"];
export type ScenarioResponse = Schemas["ScenarioResponse"];
export type InspectRequest = Schemas["InspectRequest"];
export type InspectResponse = Schemas["InspectResponse"];
export type ModelInspection = Schemas["ModelInspection"];
export type DatasetInspection = Schemas["DatasetInspection"];
export type UploadResponse = Schemas["UploadResponse"];
export type SessionStatus = Schemas["SessionStatus"];
export type BackendProfilesResponse = Schemas["BackendProfilesResponse"];
export type GpuPreset = Schemas["GpuPreset"];
export type LocalRootsResponse = Schemas["LocalRootsResponse"];
export type ErrorResponse = Schemas["ErrorResponse"];
export type Issue = Schemas["Issue"];
export type ErrorCode = Schemas["ErrorCode"];
export type Severity = Schemas["Severity"];
export type JobStatus = Schemas["JobStatus"];
export type JobProgress = Schemas["JobProgress"];
export type StatusAxes = Schemas["StatusAxes"];
export type ScenarioEstimate = Schemas["ScenarioEstimate"];
export type DeviceEstimate = Schemas["DeviceEstimate"];
export type PhasePeak = Schemas["PhasePeak"];
export type PeakBreakdown = Schemas["PeakBreakdown"];
export type BreakdownItem = Schemas["BreakdownItem"];
export type HardwareFitResult = Schemas["HardwareFitResult"];
export type CapacityRecommendation = Schemas["CapacityRecommendation"];
export type MemoryEstimate = Schemas["MemoryEstimate"];
export type DatasetScanResult = Schemas["DatasetScanResult"];
export type BranchStats = Schemas["BranchStats"];
export type ColumnMapping = Schemas["ColumnMapping"];
export type NeedsInput = Schemas["NeedsInput"];
export type ResolvedConfig = Schemas["ResolvedConfig"];
export type UnknownComponent = Schemas["UnknownComponent"];
export type ExcludedComponent = Schemas["ExcludedComponent"];
export type Objective = Schemas["Objective"];
export type Strategy = Schemas["Strategy"];
export type Phase = Schemas["Phase"];
export type AllocationCategory = Schemas["AllocationCategory"];
export type Evidence = Schemas["Evidence"];
export type EvidenceLevel = Schemas["EvidenceLevel"];
export type HardwareFit = Schemas["HardwareFit"];
export type ScanCoverage = Schemas["ScanCoverage"];
export type DataPreservation = Schemas["DataPreservation"];
export type TrainingReadiness = Schemas["TrainingReadiness"];
export type Branch = Schemas["Branch"];
export type DatasetFormat = Schemas["DatasetFormat"];
export type PreservationCheckName = Schemas["PreservationCheckName"];

/** Body of POST /analyses/{id}/scenarios/export: the scenario's full request and the format. */
export type ScenarioExportRequest = Schemas["ScenarioExportRequest"];
export type ExportFormat = ScenarioExportRequest["format"];
export type EventType = Schemas["EventType"];

/** Job states after which no further SSE events arrive (schemas/common.py TERMINAL_JOB_STATUSES). */
export const TERMINAL_JOB_STATUSES: ReadonlySet<JobStatus> = new Set<JobStatus>([
  "NEEDS_INPUT",
  "COMPLETED",
  "CANCELLED",
  "FAILED",
  "PARTIAL",
]);

export function isTerminalStatus(status: JobStatus | null | undefined): boolean {
  return status != null && TERMINAL_JOB_STATUSES.has(status);
}
