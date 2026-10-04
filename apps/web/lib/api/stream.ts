// Resilient EventSource wrapper for /analyses/{id}/events.
// The browser reconnects by itself (sending Last-Event-ID) while readyState is CONNECTING.
// When it gives up (readyState CLOSED: HTTP error or 204), we ask the caller whether the job
// already ended; otherwise we open a new stream with exponential backoff. Events are deduplicated
// by event_id because a fresh EventSource replays from the start.
import { EVENT_TYPES, TERMINAL_EVENT_TYPES, parseEventData, type AnalysisEvent } from "./events";
import { isTerminalStatus } from "./types";

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
  /** Fixed URL, or built from the last seen event id on every (re)connect. */
  url: string | ((lastEventId: number | null) => string);
  factory: EventSourceFactory;
  resolveClosed: () => Promise<ClosedResolution>;
  /** Events with an id at or below this value are ignored. */
  lastEventId?: number | null;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
}

/**
 * The stream is over after this event. The status decides, not the type alone: `failed` also
 * carries PARTIAL, and a terminal status on any event type means no more events will follow.
 */
export function endsStream(event: AnalysisEvent): boolean {
  return isTerminalStatus(event.status) || TERMINAL_EVENT_TYPES.has(event.type);
}

export interface AnalysisStreamHandlers {
  onEvent: (event: AnalysisEvent) => void;
  /**
   * The stream ended: the last event (read its `status`, never just its type), or null when the
   * end was detected through the status endpoint. Callers confirm with a final GET.
   */
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
    if (endsStream(event)) finish(event);
  };

  const scheduleReconnect = () => {
    if (done) return;
    handlers.onConnection?.("reconnecting");
    timer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, max);
  };

  /** The browser gave up (or never got a stream): did the job end, retry, or stop for good? */
  const resolveClosed = () => {
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

  function connect() {
    if (done) return;
    timer = null;
    handlers.onConnection?.(attempts === 0 ? "connecting" : "reconnecting");
    attempts += 1;
    const url = typeof options.url === "function" ? options.url(lastId >= 0 ? lastId : null) : options.url;
    let es: EventSourceLike;
    try {
      es = options.factory(url);
    } catch {
      // e.g. the constructor refused the URL: handled like a stream the browser closed.
      source = null;
      resolveClosed();
      return;
    }
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
      resolveClosed();
    };
  }

  connect();
  return {
    close: () => {
      if (!done) shutdown();
    },
  };
}
