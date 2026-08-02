from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Settings, get_settings
from backend.core.errors import ServiceError
from backend.routers import api_router
from backend.services.query_analyzer import OllamaQueryParser
from backend.clients.visual_client import InternalPipelineClient as VisualPipelineClient
from backend.clients.semantic_client import InternalPipelineClient as SemanticPipelineClient
from backend.services.search_orchestrator import SearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(timeout=active_settings.request_timeout_seconds)
        parser = OllamaQueryParser(
            client,
            active_settings.ollama_base_url,
            active_settings.ollama_model,
        )
        dev1 = VisualPipelineClient(
            client,
            source="dev1",
            base_url=active_settings.dev1_base_url,
            text_path=active_settings.dev1_text_path,
            image_path=active_settings.dev1_image_path,
        )
        dev2 = SemanticPipelineClient(
            client,
            source="dev2",
            base_url=active_settings.dev2_base_url,
            text_path=active_settings.dev2_text_path,
        )
        app.state.search_service = SearchService(
            settings=active_settings,
            parser=parser,
            dev1=dev1,
            dev2=dev2,
        )
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    
    # Mount static files if needed for relative thumbnails
    # In a real app this directory must exist, but we mount it safely
    import os
    media_dir = "media/thumbnails"
    os.makedirs(media_dir, exist_ok=True)
    app.mount("/media/thumbnails", StaticFiles(directory=media_dir), name="thumbnails")

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    return app


app = create_app()
