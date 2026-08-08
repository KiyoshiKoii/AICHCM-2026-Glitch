import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Fix for httpx/ssl crashing when SSL_CERT_FILE points to a non-existent file
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
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
from backend.services.vqa_answerer import GeminiVQAAnswerer
from backend.services.vqa_service import VQAService


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(timeout=active_settings.request_timeout_seconds)
        if active_settings.gemini_api_key:
            from backend.services.query_analyzer import GeminiQueryParser
            parser = GeminiQueryParser(
                api_key=active_settings.gemini_api_key,
                model_name="gemini-3.1-flash-lite",
            )
        else:
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
        app.state.vqa_service = VQAService(
            search_service=app.state.search_service,
            answerer=GeminiVQAAnswerer(
                api_key=active_settings.gemini_api_key,
                model_name=active_settings.gemini_vqa_model,
            ),
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
    
    import os
    import re
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    @app.get("/media/thumbnails/{frame_id_ext}")
    async def get_thumbnail(frame_id_ext: str):
        if frame_id_ext.lower().endswith(".jpg.jpg"):
            frame_id_ext = frame_id_ext[:-4]
            
        match = re.match(r"(L\d+)_V(\d+)_f(\d+)\.jpg", frame_id_ext)
        if not match:
            raise HTTPException(status_code=404, detail="Invalid frame ID format")
        
        l_part, v_part, f_part = match.groups()
        f_int = int(f_part)
        filename = f"{f_int:03d}.jpg"
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(
            base_dir, 
            "data", 
            "keyframes", 
            f"{l_part}_V{v_part}", 
            filename
        )
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path)

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    return app


app = create_app()
