// Local, instant syntax checks for model/dataset references (plan.md §3.3: "URL 검증은 로컬에서
// 즉시"). The server's SourceResolver stays authoritative (normalization, revisions, SSRF rules).
import type { Schemas } from "@/lib/api/types";

type SourceType = Schemas["SourceType"];

// Accepted forms follow packages/estimator/src/vramforge_estimator/sources/references.py:
// "org/name" or a canonical single-part id, "hf:" / "hf-dataset:" display forms, http(s) URLs of
// huggingface.co / hf.co, local:<root>/<path> and upload:<id>. The server re-validates everything.
const PART = "[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?";
const REPO_ID = `${PART}(?:/${PART})?`;
const HF_ID = new RegExp(`^${REPO_ID}$`);
const HF_MODEL_PREFIXED = new RegExp(`^hf:${REPO_ID}$`);
const HF_DATASET_PREFIXED = new RegExp(`^hf-dataset:${REPO_ID}$`);
const HF_URL = /^https?:\/\/(?:www\.)?(?:huggingface\.co|hf\.co)\/\S+$/i;
const LOCAL_REF = /^local:[A-Za-z0-9._-]+\/\S.*$/;
const UPLOAD_REF = /^upload:\S+$/;

export function isLocalReference(value: string): boolean {
  return LOCAL_REF.test(value.trim());
}

export function isValidModelReference(value: string): boolean {
  const v = value.trim();
  if (HF_ID.test(v) || HF_MODEL_PREFIXED.test(v) || LOCAL_REF.test(v)) return true;
  return HF_URL.test(v) && !/(?:huggingface|hf)\.co\/datasets\//i.test(v);
}

export function isValidDatasetReference(value: string): boolean {
  const v = value.trim();
  return HF_ID.test(v) || HF_DATASET_PREFIXED.test(v) || LOCAL_REF.test(v) || UPLOAD_REF.test(v) || HF_URL.test(v);
}

/** Model references are HF or server-local (uploads are datasets only). */
export function modelSourceType(reference: string): SourceType {
  return isLocalReference(reference) ? "local" : "huggingface";
}

/** Upload references come from POST /uploads; everything else is detected from the text. */
export function datasetSourceType(reference: string, uploadedReference: string): SourceType {
  const v = reference.trim();
  if ((uploadedReference !== "" && v === uploadedReference) || UPLOAD_REF.test(v)) return "upload";
  return isLocalReference(v) ? "local" : "huggingface";
}

/** True for HF dataset viewer links that carry a row selection (it never limits the scan). */
export function hasViewerRowParam(reference: string): boolean {
  return /huggingface\.co\/datasets\/.+[?&](?:row|p)=/.test(reference.trim());
}
