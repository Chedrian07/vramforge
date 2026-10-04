import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { parseEventData } from "@/lib/api/events";
import { openAnalysisStream, type ClosedResolution } from "@/lib/api/stream";

import { FakeEventSource, fakeEventSourceFactory } from "../utils/fake-event-source";
import { makeEvent } from "../utils/events";

function start(resolveClosed: () => Promise<ClosedResolution> = async () => "retry") {
  const onEvent = vi.fn();
  const onTerminal = vi.fn();
  const onConnection = vi.fn();
  const onStop = vi.fn();
  const stream = openAnalysisStream(
    { url: "/api/v1/analyses/a1/events", factory: fakeEventSourceFactory, resolveClosed, initialBackoffMs: 100, maxBackoffMs: 400 },
    { onEvent, onTerminal, onConnection, onStop },
  );
  return { stream, onEvent, onTerminal, onConnection, onStop };
}

beforeEach(() => FakeEventSource.reset());
afterEach(() => vi.useRealTimers());

describe("SSE payload parsing", () => {
  it("accepts contract events and rejects malformed ones", () => {
    const event = makeEvent({ type: "progress", status: "TOKENIZING", progress: { stage: "TOKENIZING", processed_rows: 10, total_rows: null } });
    expect(parseEventData(JSON.stringify(event))?.progress?.processed_rows).toBe(10);
    expect(parseEventData("not json")).toBeNull();
    expect(parseEventData(JSON.stringify({ type: "progress" }))).toBeNull();
    expect(parseEventData(JSON.stringify({ ...event, type: "unknown" }))).toBeNull();
  });
});

describe("analysis stream", () => {
  it("delivers named events in order and closes on a terminal event", () => {
    const { onEvent, onTerminal } = start();
    const es = FakeEventSource.latest();
    es.open();
    es.emit("progress", makeEvent({ event_id: 1, type: "progress", status: "RESOLVING" }));
    es.emit("progress", makeEvent({ event_id: 2, type: "progress", status: "TOKENIZING" }));
    es.emit("completed", makeEvent({ event_id: 3, type: "completed", status: "COMPLETED" }));
    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(onTerminal).toHaveBeenCalledWith(expect.objectContaining({ event_id: 3, type: "completed" }));
    expect(es.closed).toBe(true);
  });

  it("drops replayed and malformed events", () => {
    const { onEvent } = start();
    const es = FakeEventSource.latest();
    es.emit("progress", makeEvent({ event_id: 5, type: "progress", status: "TOKENIZING" }));
    es.emit("progress", makeEvent({ event_id: 5, type: "progress", status: "TOKENIZING" }));
    es.emit("progress", makeEvent({ event_id: 4, type: "progress", status: "TOKENIZING" }));
    es.emit("progress", "{broken");
    expect(onEvent).toHaveBeenCalledTimes(1);
  });

  it("lets the browser retry on a blip without opening a second stream", () => {
    const { onConnection } = start();
    FakeEventSource.latest().blip();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(onConnection).toHaveBeenLastCalledWith("reconnecting");
  });

  it("finishes through the status endpoint when the stream is closed after the job ended", async () => {
    const { onTerminal } = start(async () => "terminal");
    FakeEventSource.latest().fail();
    await vi.waitFor(() => expect(onTerminal).toHaveBeenCalledWith(null));
  });

  it("reconnects with backoff while the job is still running and dedupes the replay", async () => {
    vi.useFakeTimers();
    const resolveClosed = vi.fn(async (): Promise<ClosedResolution> => "retry");
    const { onEvent } = start(resolveClosed);
    const first = FakeEventSource.latest();
    first.emit("progress", makeEvent({ event_id: 1, type: "progress", status: "TOKENIZING" }));
    first.fail();
    await vi.advanceTimersByTimeAsync(0);
    expect(resolveClosed).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(FakeEventSource.instances).toHaveLength(2);
    const second = FakeEventSource.latest();
    second.emit("progress", makeEvent({ event_id: 1, type: "progress", status: "TOKENIZING" }));
    second.emit("progress", makeEvent({ event_id: 2, type: "progress", status: "ESTIMATING" }));
    expect(onEvent).toHaveBeenCalledTimes(2);
  });

  it("stops for good when told to (e.g. analysis deleted)", async () => {
    const { onStop, onTerminal } = start(async () => "stop");
    FakeEventSource.latest().fail();
    await vi.waitFor(() => expect(onStop).toHaveBeenCalled());
    expect(onTerminal).not.toHaveBeenCalled();
  });

  it("close() ends the stream and ignores later events", () => {
    const { stream, onEvent } = start();
    const es = FakeEventSource.latest();
    stream.close();
    es.emit("progress", makeEvent({ event_id: 9, type: "progress", status: "TOKENIZING" }));
    expect(es.closed).toBe(true);
    expect(onEvent).not.toHaveBeenCalled();
  });
});
