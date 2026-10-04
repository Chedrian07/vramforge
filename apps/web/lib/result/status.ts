// Visual tone per status value. Colour is always paired with the text label (plan.md §3.2).
import type {
  DataPreservation,
  EvidenceLevel,
  HardwareFit,
  JobStatus,
  ScanCoverage,
  Severity,
  TrainingReadiness,
} from "@/lib/api/types";

export type Tone = "ok" | "warn" | "err" | "info" | "neutral";

export const SCAN_TONE: Record<ScanCoverage, Tone> = {
  not_started: "neutral",
  partial: "warn",
  complete: "ok",
  failed: "err",
};

export const PRESERVATION_TONE: Record<DataPreservation, Tone> = {
  pending: "neutral",
  verified: "ok",
  violated: "err",
  unknown: "warn",
};

export const READINESS_TONE: Record<TrainingReadiness, Tone> = {
  ready: "ok",
  conditional: "warn",
  unsupported: "err",
};

export const EVIDENCE_TONE: Record<EvidenceLevel, Tone> = {
  metadata_only: "warn",
  analytic: "info",
  calibrated: "ok",
  measured: "ok",
};

export const FIT_TONE: Record<HardwareFit, Tone> = {
  not_evaluated: "neutral",
  expected_fit: "ok",
  low_margin: "warn",
  exceeds: "err",
  unknown: "warn",
};

export const SEVERITY_TONE: Record<Severity, Tone> = { info: "info", warning: "warn", error: "err" };

export function jobTone(status: JobStatus | null | undefined): Tone {
  switch (status) {
    case "COMPLETED":
      return "ok";
    case "FAILED":
      return "err";
    case "PARTIAL":
    case "NEEDS_INPUT":
    case "CANCELLED":
    case "CANCEL_REQUESTED":
      return "warn";
    case null:
    case undefined:
      return "neutral";
    default:
      return "info";
  }
}
