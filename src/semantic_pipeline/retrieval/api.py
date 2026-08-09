"""Internal FastAPI service consumed by the backend at port 8002."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from semantic_pipeline.retrieval.elasticsearch_backend import ElasticsearchTextSearch


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=200, ge=1, le=1000)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()) for value in values if isinstance(value, str) and value.strip()]
        if not normalized:
            raise ValueError("keywords must contain at least one non-blank string")
        return normalized


def create_app(search_backend: ElasticsearchTextSearch | None = None) -> FastAPI:
    app = FastAPI(title="AIC semantic retrieval", version="1.0")
    app.state.search_backend = search_backend

    def get_backend() -> ElasticsearchTextSearch:
        backend = app.state.search_backend
        if backend is None:
            backend = ElasticsearchTextSearch()
            app.state.search_backend = backend
        return backend

    @app.get("/health")
    def health() -> dict:
        try:
            backend = get_backend()
            if not backend.ping():
                raise RuntimeError("Elasticsearch ping returned false")
            return {"status": "ok", "documents": backend.document_count()}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Elasticsearch unavailable: {exc}") from exc

    @app.post("/internal/search/text")
    def search_text(body: TextSearchRequest) -> dict:
        active_backend = get_backend()
        try:
            return {"status": "success", "data": active_backend.search(body.keywords, top_k=body.top_k)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Elasticsearch search failed: {exc}") from exc

    return app


app = create_app()
