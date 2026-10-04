"""FastAPI application factory.

Route signatures (paths, parameters, response models) are the HTTP contract consumed by the web
app through generated TypeScript types (docs/api/openapi.json).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from redis import Redis

from vramforge_api import __version__
from vramforge_api.errors import install_error_handlers
from vramforge_api.routes import analyses, health, session, sources
from vramforge_api.security import SecurityMiddleware
from vramforge_api.settings import Settings, get_settings
from vramforge_api.state import AppState

API_PREFIX = "/api/v1"

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
