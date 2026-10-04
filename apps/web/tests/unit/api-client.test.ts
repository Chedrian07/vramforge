import { describe, expect, it, vi } from "vitest";

import { ApiError, CSRF_HEADER, apiPath, createHttpClient, filenameFromDisposition, newIdempotencyKey } from "@/lib/api/client";
import type { AnalysisRequest } from "@/lib/api/types";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SESSION_URL = "/api/v1/session";
const OPEN_SESSION = { auth_required: false, authenticated: true };

const isSessionCheck = (url: unknown, init?: RequestInit) => String(url) === SESSION_URL && (init?.method ?? "GET") === "GET";

/** `response` answers every call except the owner bootstrap (GET /session); `call(i)` skips it. */
function setup(response: Response | (() => Promise<Response>), session: () => Promise<Response> = async () => jsonResponse(200, OPEN_SESSION)) {
  const fetchImpl = vi.fn<typeof fetch>(async (url, init) => {
    if (isSessionCheck(url, init)) return session();
    return typeof response === "function" ? response() : response.clone();
  });
  const onUnauthorized = vi.fn();
  const api = createHttpClient({ fetchImpl, onUnauthorized });
  const calls = () => fetchImpl.mock.calls.filter(([url, init]) => !isSessionCheck(url, init));
  const call = (i = 0) => {
    const [url, init] = calls()[i] ?? [];
    return { url: String(url), init: init ?? {}, headers: (init?.headers ?? {}) as Record<string, string> };
  };
  const sessionChecks = () => fetchImpl.mock.calls.filter(([url, init]) => isSessionCheck(url, init)).length;
  return { api, fetchImpl, onUnauthorized, call, sessionChecks };
}

const request = {
  schema_version: "1.0",
  model: { source_type: "huggingface", reference: "org/model", revision: null, loading_scope: "auto_verified" },
  dataset: { source_type: "huggingface", reference: "org/data" },
  training: { objective: "sft", strategy: "lora" },
} as unknown as AnalysisRequest;

describe("http client", () => {
  it("sends GETs same-origin without the CSRF header", async () => {
    const { api, call } = setup(jsonResponse(200, { roots: [] }));
    await api.localRoots();
    expect(call().url).toBe("/api/v1/local-roots");
    expect(call().init.method).toBe("GET");
    expect(call().init.credentials).toBe("same-origin");
    expect(call().headers[CSRF_HEADER]).toBeUndefined();
  });

  it("adds the CSRF header and Idempotency-Key when creating an analysis", async () => {
    const { api, call } = setup(
      jsonResponse(202, { analysis_id: "a1", status: "QUEUED", fingerprint: "f", created_at: "2026-10-04T00:00:00Z" }),
    );
    const created = await api.createAnalysis(request, "key-1");
    expect(created.analysis_id).toBe("a1");
    expect(call().url).toBe("/api/v1/analyses");
    expect(call().headers[CSRF_HEADER]).toBe("1");
    expect(call().headers["Idempotency-Key"]).toBe("key-1");
    expect(call().headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(String(call().init.body))).toEqual(request);
  });

  it("posts scenarios and cancel with the CSRF header", async () => {
    const { api, call } = setup(jsonResponse(200, { fingerprint: "x", requires_reanalysis: false }));
    await api.scenarios("a/b", { request, client_fingerprint: "c" });
    expect(call().url).toBe("/api/v1/analyses/a%2Fb/scenarios");
    expect(call().headers[CSRF_HEADER]).toBe("1");
    await api.cancelAnalysis("a1");
    expect(call(1).url).toBe("/api/v1/analyses/a1/cancel");
    expect(call(1).headers[CSRF_HEADER]).toBe("1");
  });

  it("deletes with the CSRF header and accepts 204", async () => {
    const { api, call } = setup(new Response(null, { status: 204 }));
    await expect(api.deleteAnalysis("a1")).resolves.toBeUndefined();
    expect(call().url).toBe("/api/v1/analyses/a1");
    expect(call().init.method).toBe("DELETE");
    expect(call().headers[CSRF_HEADER]).toBe("1");
  });

  it("uploads multipart without forcing a content type", async () => {
    const { api, call } = setup(jsonResponse(201, { upload_id: "u1" }));
    await api.upload(new File(["{}\n"], "rows.jsonl"));
    expect(call().init.body).toBeInstanceOf(FormData);
    expect(call().headers["Content-Type"]).toBeUndefined();
    expect(call().headers[CSRF_HEADER]).toBe("1");
  });

  it("parses ErrorResponse issues", async () => {
    const issue = {
      code: "SOURCE_NOT_FOUND",
      severity: "error",
      retryable: false,
      user_message: "모델을 찾을 수 없습니다.",
    };
    const { api } = setup(jsonResponse(404, { error: issue, issues: [issue] }));
    const error = await api.getAnalysis("missing").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
    expect((error as ApiError).issue.code).toBe("SOURCE_NOT_FOUND");
    expect((error as ApiError).issues).toHaveLength(1);
  });

  it("never surfaces non-contract error bodies verbatim", async () => {
    const { api } = setup(jsonResponse(501, { detail: "not implemented <script>" }));
    const error = (await api.backendProfiles().catch((e: unknown) => e)) as ApiError;
    expect(error.issue.user_message).toBe("서버가 아직 이 기능을 제공하지 않습니다.");
    expect(error.message).not.toContain("script");
  });

  it("reports 401 to the token prompt hook", async () => {
    const { api, onUnauthorized } = setup(
      jsonResponse(401, { error: { code: "UNAUTHORIZED", severity: "error", retryable: false, user_message: "토큰 필요" } }),
    );
    await expect(api.localRoots()).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it("maps network failures to a retryable Korean issue", async () => {
    const { api } = setup(() => Promise.reject(new TypeError("Failed to fetch")));
    const error = (await api.localRoots().catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(0);
    expect(error.issue.retryable).toBe(true);
  });

  it("turns an unreadable success body into a Korean ApiError", async () => {
    // e.g. a reverse proxy that serves the app's HTML page for /api by mistake
    const { api } = setup(new Response("<!doctype html><html></html>", { status: 200, headers: { "Content-Type": "text/html" } }));
    const error = (await api.getAnalysis("a1").catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(200);
    expect(error.issue.code).toBe("INTERNAL_ERROR");
    expect(error.issue.user_message).not.toContain("<");
  });

  it("rethrows aborts untouched", async () => {
    const { api } = setup(() => Promise.reject(new DOMException("aborted", "AbortError")));
    await expect(api.localRoots()).rejects.toMatchObject({ name: "AbortError" });
  });

  it("posts the access token to the session endpoint", async () => {
    const { api, call } = setup(new Response(null, { status: 204 }));
    await api.createSession("secret-token");
    expect(call().url).toBe("/api/v1/session");
    expect(call().headers[CSRF_HEADER]).toBe("1");
    expect(JSON.parse(String(call().init.body))).toEqual({ token: "secret-token" });
  });

  it("exports a recomputed scenario with a POST and returns the file", async () => {
    const { api, call } = setup(
      new Response("# 보고서\n", { status: 200, headers: { "Content-Type": "text/markdown; charset=utf-8", "Content-Disposition": 'attachment; filename="report.md"' } }),
    );
    const file = await api.exportScenario("a/1", { request, format: "md" });
    expect(call().url).toBe("/api/v1/analyses/a%2F1/scenarios/export");
    expect(call().init.method).toBe("POST");
    expect(call().headers[CSRF_HEADER]).toBe("1");
    expect(JSON.parse(String(call().init.body))).toEqual({ request, format: "md" });
    expect(file.filename).toBe("report.md");
    expect(await file.blob.text()).toBe("# 보고서\n");
  });

  it("reports a refused scenario export as a Korean ApiError", async () => {
    const refusal = { code: "REANALYSIS_REQUIRED", severity: "error", retryable: false, user_message: "데이터 재분석이 필요합니다." };
    const { api } = setup(jsonResponse(409, { error: refusal }));
    const error = (await api.exportScenario("a1", { request, format: "json" }).catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.issue.user_message).toBe("데이터 재분석이 필요합니다.");
  });

  it("builds events and export URLs", () => {
    const { api } = setup(jsonResponse(200, {}));
    expect(api.eventsUrl("a1")).toBe("/api/v1/analyses/a1/events");
    expect(api.eventsUrl("a1", 42)).toBe("/api/v1/analyses/a1/events?after=42");
    expect(api.eventsUrl("a1", null)).toBe("/api/v1/analyses/a1/events");
    expect(api.exportUrl("a1", "trainer-config")).toBe("/api/v1/analyses/a1/export?format=trainer-config");
  });
});

describe("owner cookie bootstrap", () => {
  it("sends one GET /session before any other request and holds parallel calls until it answers", async () => {
    let answer: (response: Response) => void = () => {};
    const { api, fetchImpl, call, sessionChecks } = setup(
      jsonResponse(200, { roots: [] }),
      () => new Promise<Response>((resolve) => (answer = resolve)),
    );
    // A fresh page: profiles, local roots and a resumed analysis all start at once.
    const pending = Promise.all([api.localRoots(), api.backendProfiles(), api.getAnalysis("a1")]);
    await vi.waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(1));
    expect(String(fetchImpl.mock.calls[0]![0])).toBe(SESSION_URL);
    expect(fetchImpl.mock.calls[0]![1]?.method).toBe("GET");
    expect(fetchImpl.mock.calls[0]![1]?.credentials).toBe("same-origin");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(fetchImpl).toHaveBeenCalledTimes(1); // nothing else went out without the cookie

    answer(jsonResponse(200, OPEN_SESSION));
    await pending;
    expect(fetchImpl).toHaveBeenCalledTimes(4);
    expect([call(0).url, call(1).url, call(2).url].sort()).toEqual(["/api/v1/analyses/a1", "/api/v1/backend-profiles", "/api/v1/local-roots"]);
    await api.localRoots();
    expect(sessionChecks()).toBe(1);
  });

  it("does not block requests when the bootstrap fails and tries again on the next request", async () => {
    let failures = 1;
    const { api, sessionChecks } = setup(jsonResponse(200, { roots: [] }), async () => {
      if (failures-- > 0) throw new TypeError("Failed to fetch");
      return jsonResponse(200, OPEN_SESSION);
    });
    await expect(api.localRoots()).resolves.toEqual({ roots: [] });
    await api.localRoots();
    expect(sessionChecks()).toBe(2);
    await api.localRoots();
    expect(sessionChecks()).toBe(2);
  });

  it("keeps the owner after an API error answer (the cookie came with it)", async () => {
    const { api, sessionChecks } = setup(jsonResponse(200, { roots: [] }), async () => jsonResponse(404, { detail: "Not Found" }));
    await api.localRoots();
    await api.localRoots();
    expect(sessionChecks()).toBe(1);
  });

  it("asks for the access token when the session says one is required", async () => {
    const { api, onUnauthorized } = setup(jsonResponse(200, { roots: [] }), async () =>
      jsonResponse(200, { auth_required: true, authenticated: false }),
    );
    await api.localRoots();
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });
});

describe("helpers", () => {
  it("reads only a bare file name from Content-Disposition", () => {
    expect(filenameFromDisposition('attachment; filename="analysis.json"')).toBe("analysis.json");
    expect(filenameFromDisposition("attachment; filename=report.md")).toBe("report.md");
    expect(filenameFromDisposition("attachment; filename*=UTF-8''%EB%B3%B4%EA%B3%A0%EC%84%9C.md")).toBe("보고서.md");
    expect(filenameFromDisposition('attachment; filename="../../etc/passwd"')).toBeNull();
    expect(filenameFromDisposition("attachment")).toBeNull();
    expect(filenameFromDisposition(null)).toBeNull();
  });

  it("fills contract path parameters", () => {
    expect(apiPath("/api/v1/analyses/{analysis_id}", { analysis_id: "x y" })).toBe("/api/v1/analyses/x%20y");
    expect(() => apiPath("/api/v1/analyses/{analysis_id}")).toThrow();
  });

  it("creates unique v4 idempotency keys", () => {
    const a = newIdempotencyKey();
    const b = newIdempotencyKey();
    expect(a).not.toBe(b);
    expect(a).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });
});
