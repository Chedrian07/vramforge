// `?analysis=<id>` keeps a running job reconnectable after a reload (plan.md §19.4 작업 재연결).
const ANALYSIS_PARAM = "analysis";
const ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;

export function readAnalysisParam(): string | null {
  if (typeof window === "undefined") return null;
  const value = new URLSearchParams(window.location.search).get(ANALYSIS_PARAM);
  return value && ID_PATTERN.test(value) ? value : null;
}

export function writeAnalysisParam(id: string | null): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (id) url.searchParams.set(ANALYSIS_PARAM, id);
  else url.searchParams.delete(ANALYSIS_PARAM);
  window.history.replaceState(window.history.state, "", url);
}
