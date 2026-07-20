from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.core.config import Settings, get_settings
from backend.core.errors import ServiceError
from backend.routers import api_router
from backend.services.llm_parser import OllamaQueryParser
from backend.services.pipeline_clients import InternalPipelineClient
from backend.services.search_service import SearchService


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
        dev1 = InternalPipelineClient(
            client,
            source="dev1",
            base_url=active_settings.dev1_base_url,
            text_path=active_settings.dev1_text_path,
            image_path=active_settings.dev1_image_path,
        )
        dev2 = InternalPipelineClient(
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
    app.include_router(api_router)

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    return app


app = create_app()
