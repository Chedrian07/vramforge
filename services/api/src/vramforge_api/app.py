"""FastAPI application factory.

Route signatures (paths, parameters, response models) are the HTTP contract consumed by the web
app through generated TypeScript types. Handlers are implemented by the api agent.
"""

from __future__ import annotations

from fastapi import FastAPI

from vramforge_api import __version__
from vramforge_api.routes import analyses, health, sources

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="VRAMForge API",
        version=__version__,
        description="Fine-tuning VRAM calculator: full-dataset analysis and peak memory estimates.",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
    )
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(sources.router, prefix=API_PREFIX)
    app.include_router(analyses.router, prefix=API_PREFIX)
    return app


app = create_app()
