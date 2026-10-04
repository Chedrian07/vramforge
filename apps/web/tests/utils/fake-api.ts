import { vi } from "vitest";

import { ApiError, type ApiClient } from "@/lib/api/client";

import { backendProfiles, localRoots } from "../fixtures/sources";

const notConfigured = (name: string) =>
  vi.fn(async () => {
    throw new ApiError(501, { code: "INTERNAL_ERROR", severity: "error", retryable: false, user_message: `fake ${name} not configured` });
  });

/** ApiClient with vi.fn() members; profiles and local roots answer with fixtures by default. */
export function createFakeApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    inspect: notConfigured("inspect"),
    upload: notConfigured("upload"),
    backendProfiles: vi.fn(async () => backendProfiles),
    localRoots: vi.fn(async () => localRoots),
    createAnalysis: notConfigured("createAnalysis"),
    getAnalysis: notConfigured("getAnalysis"),
    cancelAnalysis: notConfigured("cancelAnalysis"),
    scenarios: notConfigured("scenarios"),
    createSession: vi.fn(async () => undefined),
    eventsUrl: (id: string) => `/api/v1/analyses/${id}/events`,
    exportUrl: (id: string, format: string) => `/api/v1/analyses/${id}/export?format=${format}`,
    ...overrides,
  } as ApiClient;
}
