"""Internal FastAPI service consumed by the backend at port 8002."""

from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObjectQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    english_phrase: str = Field(min_length=1, max_length=300)
    vietnamese_phrase: str = Field(min_length=1, max_length=300)

    @field_validator("english_phrase", "vietnamese_phrase")
    @classmethod
    def normalize_phrase(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("object query phrase must not be blank")
        return normalized

from semantic_pipeline.retrieval.elasticsearch_backend import ElasticsearchTextSearch


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1, max_length=20)
    object_queries: list[ObjectQuery] = Field(default_factory=list, max_length=5)
    ocr_queries: list[str] = Field(default_factory=list, max_length=8)
    program_queries: list[str] = Field(default_factory=list, max_length=5)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=200, ge=1, le=1000)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()) for value in values if isinstance(value, str) and value.strip()]
        if not normalized:
            raise ValueError("keywords must contain at least one non-blank string")
        return normalized

    @field_validator("ocr_queries", "program_queries")
    @classmethod
    def normalize_query_phrases(cls, values: list[str]) -> list[str]:
        return [" ".join(value.split()) for value in values if isinstance(value, str) and value.strip()]

    @field_validator("batch_ids", "video_ids", mode="before")
    @classmethod
    def normalize_scope_ids(cls, values):
        if isinstance(values, str):
            values = re.split(r"[,;\s]+", values)
        if not isinstance(values, list):
            return values
        result = []
        seen = set()
        for value in values:
            if isinstance(value, str) and (normalized := value.strip().upper()):
                if re.fullmatch(r"\d{1,3}", normalized):
                    normalized = f"V{int(normalized):03d}"
                if normalized in seen:
                    continue
                seen.add(normalized)
                result.append(normalized)
        return result


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
            return {
                "status": "success",
                "data": active_backend.search(
                    body.keywords,
                    object_queries=[item.model_dump(mode="json") for item in body.object_queries],
                    ocr_queries=body.ocr_queries,
                    program_queries=body.program_queries,
                    batch_ids=body.batch_ids,
                    video_ids=body.video_ids,
                    top_k=body.top_k,
                ),
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Elasticsearch search failed: {exc}") from exc

    return app


app = create_app()
