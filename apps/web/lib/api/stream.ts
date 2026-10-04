// Resilient EventSource wrapper for /analyses/{id}/events.
// The browser reconnects by itself (sending Last-Event-ID) while readyState is CONNECTING.
// When it gives up (readyState CLOSED: HTTP error or 204), we ask the caller whether the job
// already ended; otherwise we open a new stream with exponential backoff. Events are deduplicated
// by event_id because a fresh EventSource replays from the start.
import { EVENT_TYPES, TERMINAL_EVENT_TYPES, parseEventData, type AnalysisEvent } from "./events";

export interface EventSourceLike {
  readonly readyState: number;
  onopen: ((event: Event) => unknown) | null;
  onerror: ((event: Event) => unknown) | null;
  addEventListener(type: string, listener: (event: MessageEvent) => void): void;
  close(): void;
}

export type EventSourceFactory = (url: string) => EventSourceLike;

export const browserEventSourceFactory: EventSourceFactory = (url) =>
  new EventSource(url) as unknown as EventSourceLike;

export type ConnectionState = "connecting" | "open" | "reconnecting" | "closed";

/** What to do after the browser closed the stream: the job ended, retry, or stop for good. */
export type ClosedResolution = "terminal" | "retry" | "stop";

export interface AnalysisStreamOptions {
  url: string;
  factory: EventSourceFactory;
  resolveClosed: () => Promise<ClosedResolution>;
  /** Events with an id at or below this value are ignored. */
  lastEventId?: number | null;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
}

export interface AnalysisStreamHandlers {
  onEvent: (event: AnalysisEvent) => void;
  /** Terminal event, or null when the end was detected through the status endpoint. */
  onTerminal: (event: AnalysisEvent | null) => void;
  onConnection?: (state: ConnectionState) => void;
  onStop?: () => void;
}

const CLOSED = 2;

export function openAnalysisStream(
  options: AnalysisStreamOptions,
  handlers: AnalysisStreamHandlers,
): { close: () => void } {
  const initial = options.initialBackoffMs ?? 1_000;
  const max = options.maxBackoffMs ?? 15_000;
  let backoff = initial;
  let lastId = options.lastEventId ?? -1;
  let source: EventSourceLike | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let done = false;
  let attempts = 0;

  const shutdown = () => {
    done = true;
    if (timer) clearTimeout(timer);
    timer = null;
    source?.close();
    source = null;
    handlers.onConnection?.("closed");
  };

  const finish = (event: AnalysisEvent | null) => {
    if (done) return;
    shutdown();
    handlers.onTerminal(event);
  };

  const onMessage = (message: MessageEvent) => {
    if (done) return;
    const event = parseEventData(message.data);
    if (!event || event.event_id <= lastId) return;
    lastId = event.event_id;
    backoff = initial;
    handlers.onEvent(event);
    if (TERMINAL_EVENT_TYPES.has(event.type)) finish(event);
  };

  const scheduleReconnect = () => {
    if (done) return;
    handlers.onConnection?.("reconnecting");
    timer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, max);
  };

  function connect() {
    if (done) return;
    timer = null;
    handlers.onConnection?.(attempts === 0 ? "connecting" : "reconnecting");
    attempts += 1;
    const es = options.factory(options.url);
    source = es;
    for (const type of EVENT_TYPES) es.addEventListener(type, onMessage);
    es.addEventListener("message", onMessage);
    es.onopen = () => {
      if (done || source !== es) return;
      backoff = initial;
      handlers.onConnection?.("open");
    };
    es.onerror = () => {
      if (done || source !== es) return;
      if (es.readyState !== CLOSED) {
        handlers.onConnection?.("reconnecting"); // native retry with Last-Event-ID
        return;
      }
      es.close();
      source = null;
      options.resolveClosed().then(
        (resolution) => {
          if (done) return;
          if (resolution === "terminal") finish(null);
          else if (resolution === "stop") {
            shutdown();
            handlers.onStop?.();
          } else scheduleReconnect();
        },
        () => scheduleReconnect(),
      );
    };
  }

  connect();
  return {
    close: () => {
      if (!done) shutdown();
    },
  };
}
