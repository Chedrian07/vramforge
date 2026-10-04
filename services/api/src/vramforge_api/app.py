"""FastAPI application factory.

Route signatures (paths, parameters, response models) are the HTTP contract consumed by the web
app through generated TypeScript types (docs/api/openapi.json).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic.json_schema import models_json_schema
from redis import Redis

from vramforge_api import __version__
from vramforge_api.errors import install_error_handlers
from vramforge_api.routes import analyses, health, session, sources
from vramforge_api.security import SecurityMiddleware
from vramforge_api.settings import Settings, get_settings
from vramforge_api.state import AppState
from vramforge_estimator.schemas import AnalysisEvent, Branch

API_PREFIX = "/api/v1"
EVENTS_PATH = f"{API_PREFIX}/analyses/{{analysis_id}}/events"
SCHEMA_REF = "#/components/schemas/{model}"

# `AnalysisEvent.partial` (filled from the scanner's progress reports, sanitized by the worker).
PARTIAL_DESCRIPTION = (
    "Display-only scan statistics on `progress` events while TOKENIZING; never raw rows, token "
    "ids or credentials, and null on other events. Keys: `status` (`partial` while scanning; the "
    "final report carries the scan coverage: `complete`, `partial` or `failed`), `rows_ok` and "
    "`rows_failed` (rows tokenized / failed so far) and, in tokens, the longest row seen so far "
    "of every length branch that has data: "
    + ", ".join(f"`max_{branch.value}`" for branch in Branch)
    + ". Sanitized by the worker: at most 32 keys, strings cut at 200 characters, booleans sent "
    "as 0/1, nested values dropped. Partial values are not dataset maxima until the scan is "
    "complete."
)

# The upload route streams the multipart body itself (size cap while streaming), so the request
# body schema FastAPI would derive from `UploadFile` is declared explicitly to keep the contract.
_UPLOAD_BODY_SCHEMA: dict[str, Any] = {
    "properties": {
        "file": {
            "contentMediaType": "application/octet-stream",
            "title": "File",
            "type": "string",
        }
    },
    "required": ["file"],
    "title": sources.UPLOAD_BODY_SCHEMA,
    "type": "object",
}


def _without_none(value: Any) -> Any:
    """Drop None values like FastAPI's own `exclude_none` dump of the document."""
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


def _add_event_contract(schema: dict[str, Any]) -> None:
    """The SSE payload is not a route's response model, so FastAPI would leave `AnalysisEvent`
    (and `EventType`) out of the document the web types are generated from. Add them and describe
    the `/events` stream with the OpenAPI 3.2 SSE item schema FastAPI itself uses."""
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    _, definitions = models_json_schema([(AnalysisEvent, "serialization")], ref_template=SCHEMA_REF)
    for name, definition in definitions.get("$defs", {}).items():
        schemas.setdefault(name, _without_none(definition))  # keep FastAPI's existing entries
    event = schemas["AnalysisEvent"]
    event["description"] = (
        "One server-sent event of `GET /api/v1/analyses/{analysis_id}/events` (the JSON in "
        "`data:`). `event:` repeats `type`, `id:` is `event_id` (resume with Last-Event-ID)."
    )
    event["properties"]["partial"]["description"] = PARTIAL_DESCRIPTION
    route = schema.get("paths", {}).get(EVENTS_PATH, {}).get("get")
    if route is None:
        return
    content = route["responses"]["200"].setdefault("content", {})
    content["text/event-stream"] = {
        "itemSchema": {
            "type": "object",
            "required": ["data"],
            "properties": {
                "id": {"type": "string", "description": "AnalysisEvent.event_id"},
                "event": {"$ref": SCHEMA_REF.format(model="EventType")},
                "data": {
                    "type": "string",
                    "contentMediaType": "application/json",
                    "contentSchema": {"$ref": SCHEMA_REF.format(model="AnalysisEvent")},
                },
                "retry": {"type": "integer", "minimum": 0},
            },
        }
    }


def _install_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        schemas.setdefault(sources.UPLOAD_BODY_SCHEMA, _UPLOAD_BODY_SCHEMA)
        _add_event_contract(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def create_app(
    settings: Settings | None = None,
    *,
    redis: Redis | None = None,
    queue_is_async: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="VRAMForge API",
        version=__version__,
        description="Fine-tuning VRAM calculator: full-dataset analysis and peak memory estimates.",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
    )
    app.state.vf = AppState(settings=settings, redis_client=redis, queue_is_async=queue_is_async)
    install_error_handlers(app)
    app.add_middleware(SecurityMiddleware, settings=settings)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(session.router, prefix=API_PREFIX)
    app.include_router(sources.router, prefix=API_PREFIX)
    app.include_router(analyses.router, prefix=API_PREFIX)
    _install_openapi(app)
    return app


app = create_app()
