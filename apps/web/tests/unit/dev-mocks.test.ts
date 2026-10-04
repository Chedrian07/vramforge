import { afterEach, describe, expect, it, vi } from "vitest";

import { parseEventData } from "@/lib/api/events";
import { toAnalysisRequest } from "@/lib/form/convert";
import { EXAMPLE_FORM_VALUES } from "@/lib/form/values";
import { devMocks } from "@/lib/dev-mocks/index";
import { devMocks as disabled } from "@/lib/dev-mocks/disabled";

afterEach(() => vi.useRealTimers());

describe("dev mock mode", () => {
  it("is empty in the default (production) alias target", () => {
    expect(disabled).toBeNull();
  });

  it("replays a scripted run and serves the fixture result", async () => {
    vi.useFakeTimers();
    expect(devMocks).not.toBeNull();
    const { api, eventSourceFactory, label } = devMocks!;
    expect(label).toContain("NEXT_PUBLIC_VF_DEV_MOCKS");
    const request = toAnalysisRequest(EXAMPLE_FORM_VALUES);
    const createdPromise = api.createAnalysis(request, "key");
    await vi.advanceTimersByTimeAsync(200);
    const created = await createdPromise;
    expect(created.analysis_id).toMatch(/^vf-fixture-run-/);

    const es = eventSourceFactory(api.eventsUrl(created.analysis_id));
    const types: string[] = [];
    for (const type of ["progress", "partial_result", "completed"]) {
      es.addEventListener(type, (event) => {
        const parsed = parseEventData(event.data);
        if (parsed) types.push(parsed.type);
      });
    }
    await vi.advanceTimersByTimeAsync(10_000);
    expect(types.at(-1)).toBe("completed");
    expect(types).toContain("partial_result");

    const statusPromise = api.getAnalysis(created.analysis_id);
    await vi.advanceTimersByTimeAsync(100);
    const status = await statusPromise;
    expect(status.status).toBe("COMPLETED");
    expect(status.result?.requested_config).toEqual(request);
    expect(status.result?.memory?.primary_scenario_id).toBeNull();
  });
});
