import type { EventSourceLike } from "@/lib/api/stream";

/** In-memory EventSource (jsdom has none; docs/research/stack-compat.md §10.4). */
export class FakeEventSource implements EventSourceLike {
  static instances: FakeEventSource[] = [];
  static reset() {
    FakeEventSource.instances = [];
  }
  static latest(): FakeEventSource {
    const last = FakeEventSource.instances.at(-1);
    if (!last) throw new Error("no EventSource was opened");
    return last;
  }

  readyState = 0;
  onopen: ((event: Event) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  closed = false;
  private listeners = new Map<string, Array<(event: MessageEvent) => void>>();

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  open() {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  emit(type: string, data: unknown, id?: string) {
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    const event = new MessageEvent(type, { data: payload, lastEventId: id ?? "" });
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }

  /** Network blip: the browser keeps reconnecting by itself. */
  blip() {
    this.readyState = 0;
    this.onerror?.(new Event("error"));
  }

  /** The browser gave up (HTTP error / 204). */
  fail() {
    this.readyState = 2;
    this.onerror?.(new Event("error"));
  }

  close() {
    this.readyState = 2;
    this.closed = true;
  }
}

export const fakeEventSourceFactory = (url: string) => new FakeEventSource(url);
