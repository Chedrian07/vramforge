// Same-origin REST client for the VRAMForge API (docs/architecture.md §6).
// Ownership is the httpOnly `vf_owner` cookie (same-origin fetch sends it); state-changing
// requests carry the CSRF header. Error bodies are `ErrorResponse { error: Issue }`.
//
// The API mints a new `vf_owner` cookie for every request that arrives without one. A fresh
// browser fires several requests at once (profiles, local roots, a resumed analysis), each would
// get a different owner and the last Set-Cookie would win, hiding analyses created under the
// others. So one GET /session runs first and every other request waits for it.
import type { paths } from "./schema";
import type {
  AnalysisCreated,
  AnalysisRequest,
  AnalysisStatus,
  BackendProfilesResponse,
  ErrorCode,
  ExportFormat,
  InspectRequest,
  InspectResponse,
  Issue,
  LocalRootsResponse,
  ScenarioExportRequest,
  ScenarioRequest,
  ScenarioResponse,
  SessionStatus,
  UploadResponse,
} from "./types";

export const API_BASE = "/api/v1";
export const CSRF_HEADER = "X-VramForge-Request";

/** Fills `{param}` placeholders of a path declared in the OpenAPI contract. */
export function apiPath<K extends keyof paths>(template: K, params: Record<string, string> = {}): string {
  return template.replace(/\{(\w+)\}/g, (_match, name: string) => {
    const value = params[name];
    if (value == null) throw new Error(`missing path parameter: ${name}`);
    return encodeURIComponent(value);
  });
}

export class ApiError extends Error {
  readonly status: number;
  readonly issue: Issue;
  readonly issues: Issue[];

  constructor(status: number, issue: Issue, issues: Issue[] = []) {
    super(issue.user_message);
    this.name = "ApiError";
    this.status = status;
    this.issue = issue;
    this.issues = issues;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export interface ApiClient {
  inspect(body: InspectRequest, signal?: AbortSignal): Promise<InspectResponse>;
  upload(file: File, signal?: AbortSignal): Promise<UploadResponse>;
  backendProfiles(signal?: AbortSignal): Promise<BackendProfilesResponse>;
  localRoots(signal?: AbortSignal): Promise<LocalRootsResponse>;
  createAnalysis(body: AnalysisRequest, idempotencyKey: string): Promise<AnalysisCreated>;
  getAnalysis(analysisId: string, signal?: AbortSignal): Promise<AnalysisStatus>;
  cancelAnalysis(analysisId: string): Promise<AnalysisStatus>;
  /** Deletes the analysis with its artifacts (plan.md §16.4). */
  deleteAnalysis(analysisId: string): Promise<void>;
  scenarios(analysisId: string, body: ScenarioRequest, signal?: AbortSignal): Promise<ScenarioResponse>;
  /**
   * Export file of a recomputed scenario (POST /analyses/{id}/scenarios/export): what the screen
   * shows after a light change, which the stored analysis behind `exportUrl` does not contain.
   */
  exportScenario(analysisId: string, body: ScenarioExportRequest, signal?: AbortSignal): Promise<ExportFile>;
  createSession(token: string): Promise<void>;
  /** SSE URL; `after` resumes after that event id on a fresh connection (no header needed). */
  eventsUrl(analysisId: string, after?: number | null): string;
  exportUrl(analysisId: string, format: ExportFormat): string;
}

export interface ExportFile {
  blob: Blob;
  /** From Content-Disposition; null when the server did not name the file. */
  filename: string | null;
}

/** `attachment; filename="report.md"` -> "report.md" (plain or RFC 5987 form). */
export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const extended = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(header);
  if (extended?.[1]) {
    try {
      return decodeURIComponent(extended[1].trim());
    } catch {
      return null;
    }
  }
  const plain = /filename\s*=\s*"([^"]*)"|filename\s*=\s*([^;]+)/.exec(header);
  const name = (plain?.[1] ?? plain?.[2] ?? "").trim();
  // Only a bare file name is ever used for the download.
  return name && !/[\\/]/.test(name) ? name : null;
}

export interface HttpClientOptions {
  fetchImpl?: typeof fetch;
  /** Called on every 401 so the UI can ask for the access token. */
  onUnauthorized?: () => void;
}

const FALLBACK_MESSAGE: Record<number, [ErrorCode, string]> = {
  0: ["INTERNAL_ERROR", "서버에 연결하지 못했습니다. 네트워크나 서비스 상태를 확인하세요."],
  400: ["INVALID_REQUEST", "요청을 처리할 수 없습니다. 입력값을 확인하세요."],
  401: ["UNAUTHORIZED", "접근 토큰이 필요합니다."],
  403: ["FORBIDDEN", "이 작업에 대한 권한이 없습니다."],
  404: ["NOT_FOUND", "요청한 항목을 찾을 수 없습니다."],
  409: ["CONFLICTING_OPTIONS", "현재 상태에서는 요청을 처리할 수 없습니다."],
  422: ["INVALID_REQUEST", "요청 형식이 올바르지 않습니다."],
  429: ["CONCURRENCY_LIMIT", "동시에 실행할 수 있는 작업 수를 초과했습니다. 잠시 후 다시 시도하세요."],
  501: ["INTERNAL_ERROR", "서버가 아직 이 기능을 제공하지 않습니다."],
  503: ["INTERNAL_ERROR", "서비스를 일시적으로 사용할 수 없습니다."],
};

function fallbackIssue(status: number): Issue {
  const [code, message] =
    FALLBACK_MESSAGE[status] ??
    (status >= 500
      ? (["INTERNAL_ERROR", "서버 오류가 발생했습니다. 잠시 후 다시 시도하세요."] as const)
      : (["INVALID_REQUEST", "요청을 처리하지 못했습니다."] as const));
  return {
    code,
    severity: "error",
    retryable: status === 0 || status === 429 || status >= 500,
    user_message: message,
  };
}

/** A 2xx whose body is not the contract JSON (e.g. an HTML page from a misrouted proxy). */
function unreadableBody(status: number): ApiError {
  return new ApiError(status, {
    code: "INTERNAL_ERROR",
    severity: "error",
    retryable: true,
    user_message: "서버 응답 형식이 올바르지 않습니다. 프록시 설정이나 서비스 상태를 확인하세요.",
  });
}

function isIssue(value: unknown): value is Issue {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.code === "string" && typeof v.user_message === "string";
}

async function errorFrom(response: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (typeof body === "object" && body !== null) {
    const b = body as Record<string, unknown>;
    if (isIssue(b.error)) {
      const issues = Array.isArray(b.issues) ? b.issues.filter(isIssue) : [];
      return new ApiError(response.status, b.error, issues);
    }
  }
  // Non-contract bodies (e.g. framework defaults) are never shown raw.
  return new ApiError(response.status, fallbackIssue(response.status));
}

type Method = "GET" | "POST" | "DELETE";

interface SendOptions {
  json?: unknown;
  body?: BodyInit;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export function createHttpClient(options: HttpClientOptions = {}): ApiClient {
  const doFetch: typeof fetch = options.fetchImpl ?? ((input, init) => fetch(input, init));

  /** One request; non-2xx answers become ApiError (401 also opens the token prompt). */
  async function fetchOk(method: Method, url: string, init: SendOptions = {}): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json", ...init.headers };
    if (method !== "GET") headers[CSRF_HEADER] = "1";
    let body = init.body;
    if (init.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(init.json);
    }
    let response: Response;
    try {
      response = await doFetch(url, {
        method,
        headers,
        body,
        credentials: "same-origin",
        cache: "no-store",
        signal: init.signal,
      });
    } catch (error) {
      if (isAbortError(error)) throw error;
      throw new ApiError(0, fallbackIssue(0));
    }
    if (!response.ok) {
      const error = await errorFrom(response);
      if (response.status === 401) options.onUnauthorized?.();
      throw error;
    }
    return response;
  }

  async function send<T>(method: Method, url: string, init: SendOptions = {}): Promise<T> {
    const response = await fetchOk(method, url, init);
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    if (!text) return undefined as T;
    try {
      return JSON.parse(text) as T;
    } catch {
      throw unreadableBody(response.status);
    }
  }

  // One owner bootstrap per client (GET /session is exempt from the access token, so it also says
  // whether the token prompt is needed). Any API answer, even an error status, carries the cookie;
  // when the API was not reached (network, proxy 5xx) the next request tries again. A failed
  // attempt never blocks the waiting request, which then reports its own error.
  let ownerReady: Promise<void> | null = null;
  function ensureOwner(): Promise<void> {
    ownerReady ??= send<SessionStatus | undefined>("GET", apiPath("/api/v1/session")).then(
      (status) => {
        if (status?.auth_required && !status.authenticated) options.onUnauthorized?.();
      },
      (error: unknown) => {
        if (!(error instanceof ApiError) || error.status === 0 || error.status >= 500) ownerReady = null;
      },
    );
    return ownerReady;
  }

  async function request<T>(method: Method, url: string, init: SendOptions = {}): Promise<T> {
    await ensureOwner();
    return send<T>(method, url, init);
  }

  const analysisPath = (analysisId: string) =>
    apiPath("/api/v1/analyses/{analysis_id}", { analysis_id: analysisId });

  return {
    inspect: (body, signal) =>
      request<InspectResponse>("POST", apiPath("/api/v1/sources/inspect"), { json: body, signal }),
    upload: (file, signal) => {
      const form = new FormData();
      form.append("file", file);
      return request<UploadResponse>("POST", apiPath("/api/v1/uploads"), { body: form, signal });
    },
    backendProfiles: (signal) =>
      request<BackendProfilesResponse>("GET", apiPath("/api/v1/backend-profiles"), { signal }),
    localRoots: (signal) =>
      request<LocalRootsResponse>("GET", apiPath("/api/v1/local-roots"), { signal }),
    createAnalysis: (body, idempotencyKey) =>
      request<AnalysisCreated>("POST", apiPath("/api/v1/analyses"), {
        json: body,
        headers: { "Idempotency-Key": idempotencyKey },
      }),
    getAnalysis: (analysisId, signal) =>
      request<AnalysisStatus>("GET", analysisPath(analysisId), { signal }),
    cancelAnalysis: (analysisId) =>
      request<AnalysisStatus>(
        "POST",
        apiPath("/api/v1/analyses/{analysis_id}/cancel", { analysis_id: analysisId }),
      ),
    deleteAnalysis: async (analysisId) => {
      await request<undefined>("DELETE", analysisPath(analysisId));
    },
    scenarios: (analysisId, body, signal) =>
      request<ScenarioResponse>(
        "POST",
        apiPath("/api/v1/analyses/{analysis_id}/scenarios", { analysis_id: analysisId }),
        { json: body, signal },
      ),
    exportScenario: async (analysisId, body, signal) => {
      await ensureOwner();
      // Not in the generated paths yet: built next to the scenarios route it belongs to.
      const url = `${apiPath("/api/v1/analyses/{analysis_id}/scenarios", { analysis_id: analysisId })}/export`;
      const response = await fetchOk("POST", url, { json: body, signal, headers: { Accept: "*/*" } });
      let blob: Blob;
      try {
        blob = await response.blob();
      } catch (error) {
        if (isAbortError(error)) throw error;
        throw unreadableBody(response.status);
      }
      return { blob, filename: filenameFromDisposition(response.headers.get("Content-Disposition")) };
    },
    createSession: async (token) => {
      await request<unknown>("POST", apiPath("/api/v1/session"), { json: { token } });
    },
    eventsUrl: (analysisId, after) => {
      const base = apiPath("/api/v1/analyses/{analysis_id}/events", { analysis_id: analysisId });
      return after != null && after >= 0 ? `${base}?after=${Math.floor(after)}` : base;
    },
    exportUrl: (analysisId, format) =>
      `${apiPath("/api/v1/analyses/{analysis_id}/export", { analysis_id: analysisId })}?format=${encodeURIComponent(format)}`,
  };
}

/** UUID v4 per click; works outside secure contexts (plain-HTTP LAN deployments). */
export function newIdempotencyKey(): string {
  const c = globalThis.crypto;
  if (typeof c?.randomUUID === "function") return c.randomUUID();
  const bytes = new Uint8Array(16);
  c.getRandomValues(bytes);
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40;
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
