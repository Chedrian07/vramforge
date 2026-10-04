"""Server-sent events for analysis progress (plan.md §15.3, docs/research/stack-compat.md §10.9).

PostgreSQL `analysis_events` is the replay log: a (re)connecting client gets every event with an
id greater than `Last-Event-ID`, then the stream polls for new rows. The stream ends right after a
terminal event, or as soon as a finished analysis has nothing left to send. Unknown/foreign ids
and already-finished replays are answered by the route with a normal (non-streaming) 404/204,
because a generator cannot change the status code once started.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable

import anyio

from vramforge_estimator.schemas import TERMINAL_EVENT_TYPES, AnalysisEvent

from . import store
from .models import Analysis
from .state import AppState

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
RETRY_MS = 1000


def parse_last_event_id(header: str | None, query: int | None) -> int:
    for value in (header, query):
        if value is None:
            continue
        try:
            parsed = int(str(value).strip())
        except ValueError:
            continue
        if parsed >= 0:
            return parsed
    return 0


def format_event(event: AnalysisEvent) -> bytes:
    data = event.model_dump_json()  # single line: no newline can break the SSE framing
    return f"id: {event.event_id}\nevent: {event.type.value}\ndata: {data}\n\n".encode()


def _poll(
    state: AppState, analysis_id: str, after_id: int
) -> tuple[list[AnalysisEvent], bool, bool]:
    """New events, whether the analysis still exists and whether it was already finished when
    the events were read. The status is read first: a terminal status is committed together with
    its terminal event, so that event is then either in `events` or was delivered before."""
    with state.sessions() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            return [], False, False
        finished = store.is_terminal(analysis)
        events = store.events_after(db, analysis_id, after_id)
    return events, True, finished


async def event_stream(
    state: AppState,
    analysis_id: str,
    after_id: int,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[bytes]:
    settings = state.settings
    yield f"retry: {RETRY_MS}\n\n".encode()
    last_id = after_id
    started = time.monotonic()
    last_write = started
    while True:
        if await is_disconnected():
            return
        events, exists, finished = await anyio.to_thread.run_sync(
            _poll, state, analysis_id, last_id
        )
        for event in events:
            yield format_event(event)
            last_id = event.event_id
            last_write = time.monotonic()
            if event.type in TERMINAL_EVENT_TYPES:
                return
        if not exists:
            return
        if finished and not events:
            # Nothing more can arrive (e.g. the cursor is past the terminal event): never keep a
            # finished analysis' stream polling until sse_max_duration_s.
            return
        now = time.monotonic()
        if now - started > settings.sse_max_duration_s:
            return
        if now - last_write >= settings.sse_keepalive_s:
            yield b": keepalive\n\n"
            last_write = now
        if not events:
            await anyio.sleep(settings.sse_poll_interval_s)


__all__ = ["SSE_HEADERS", "event_stream", "format_event", "parse_last_event_id"]
