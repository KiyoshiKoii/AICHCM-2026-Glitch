"""Internal FastAPI service consumed by the backend at port 8002."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class SpatialQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_english_phrase: str = Field(min_length=1, max_length=300)
    subject_vietnamese_phrase: str = Field(default="", max_length=300)
    predicate: Literal["left_of", "right_of", "above", "below", "overlapping"]
    object_english_phrase: str = Field(min_length=1, max_length=300)
    object_vietnamese_phrase: str = Field(default="", max_length=300)

    @field_validator(
        "subject_english_phrase",
        "subject_vietnamese_phrase",
        "object_english_phrase",
        "object_vietnamese_phrase",
    )
    @classmethod
    def normalize_phrase(cls, value: str) -> str:
        return " ".join(value.split())


class InteractionQuery(BaseModel):
    """One subject-action-object interaction bound to a shared relation row."""

    model_config = ConfigDict(extra="forbid")

    subject_english_phrase: str = Field(min_length=1, max_length=300)
    subject_vietnamese_phrase: str = Field(default="", max_length=300)
    action_english_phrase: str = Field(min_length=1, max_length=200)
    action_vietnamese_phrase: str = Field(default="", max_length=200)
    object_english_phrase: str = Field(min_length=1, max_length=300)
    object_vietnamese_phrase: str = Field(default="", max_length=300)

    @field_validator(
        "subject_english_phrase",
        "subject_vietnamese_phrase",
        "action_english_phrase",
        "action_vietnamese_phrase",
        "object_english_phrase",
        "object_vietnamese_phrase",
    )
    @classmethod
    def normalize_phrase(cls, value: str) -> str:
        return " ".join(value.split())


from semantic_pipeline.retrieval.elasticsearch_backend import ElasticsearchTextSearch
from semantic_pipeline.retrieval.temporal_event_search import (
    TemporalEventSearch,
    discover_temporal_corpus,
)
from semantic_pipeline.retrieval.temporal_query_expander import (
    GeminiTemporalQueryParser,
    temporal_query_to_retrieval_spec,
)
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query
from semantic_pipeline.retrieval.temporal_video_selector import ElasticsearchVideoSelector
from semantic_pipeline.retrieval.qwen_video_verifier import QwenTemporalVerifier
from semantic_pipeline.retrieval.dense_motion_verifier import DenseMotionTemporalVerifier


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1, max_length=20)
    object_queries: list[ObjectQuery] = Field(default_factory=list, max_length=5)
    spatial_queries: list[SpatialQuery] = Field(default_factory=list, max_length=5)
    interaction_queries: list[InteractionQuery] = Field(default_factory=list, max_length=5)
    ocr_queries: list[str] = Field(default_factory=list, max_length=8)
    program_queries: list[str] = Field(default_factory=list, max_length=5)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    exclude_quality_flags: list[str] = Field(default_factory=list, max_length=12)
    collapse_visual_duplicates: bool = False
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


class TemporalEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=5000)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k_videos: int = Field(default=20, ge=1, le=100)
    verify_with_qwen: bool = False
    verify_with_dense: bool = False
    qwen_candidate_events: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = "\n".join(" ".join(line.split()) for line in value.splitlines()).strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("batch_ids", "video_ids", mode="before")
    @classmethod
    def normalize_ids(cls, values):
        if isinstance(values, str):
            values = re.split(r"[,;\s]+", values)
        return [str(value).strip().upper() for value in values or [] if str(value).strip()]


class TemporalVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=5000)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k_videos: int = Field(default=20, ge=1, le=100)
    summary_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    event_weight: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = "\n".join(" ".join(line.split()) for line in value.splitlines()).strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("batch_ids", "video_ids", mode="before")
    @classmethod
    def normalize_ids(cls, values):
        if isinstance(values, str):
            values = re.split(r"[,;\s]+", values)
        return [str(value).strip().upper() for value in values or [] if str(value).strip()]

    @model_validator(mode="after")
    def validate_weights(self) -> "TemporalVideoRequest":
        if self.summary_weight + self.event_weight <= 0:
            raise ValueError("summary_weight and event_weight cannot both be zero")
        return self


def create_app(search_backend: ElasticsearchTextSearch | None = None) -> FastAPI:
    app = FastAPI(title="AIC semantic retrieval", version="1.0")
    app.state.search_backend = search_backend
    app.state.temporal_query_parser = GeminiTemporalQueryParser()

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
                    spatial_queries=[item.model_dump(mode="json") for item in body.spatial_queries],
                    interaction_queries=[item.model_dump(mode="json") for item in body.interaction_queries],
                    ocr_queries=body.ocr_queries,
                    program_queries=body.program_queries,
                    batch_ids=body.batch_ids,
                    video_ids=body.video_ids,
                    exclude_quality_flags=body.exclude_quality_flags,
                    collapse_visual_duplicates=body.collapse_visual_duplicates,
                    top_k=body.top_k,
                ),
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Elasticsearch search failed: {exc}") from exc

    @app.post("/internal/search/temporal-events")
    def search_temporal_events(body: TemporalEventRequest) -> dict:
        try:
            if body.verify_with_qwen and body.verify_with_dense:
                raise ValueError("choose only one temporal verifier: Qwen or dense")
            temporal_verifier = (
                QwenTemporalVerifier()
                if body.verify_with_qwen
                else DenseMotionTemporalVerifier()
                if body.verify_with_dense
                else None
            )
            parsed = app.state.temporal_query_parser.parse(parse_temporal_query(body.query))
            selected_video_ids = list(body.video_ids)
            video_selection: dict
            if selected_video_ids:
                video_selection = {
                    "mode": "explicit_video_ids",
                    "selected_video_id": selected_video_ids[0],
                    "candidates": [{"video_id": item} for item in selected_video_ids],
                }
            else:
                video_selection = ElasticsearchVideoSelector(get_backend().client).select(
                    parsed,
                    batch_ids=body.batch_ids,
                    top_k=body.top_k_videos,
                )
                selected_video_id = video_selection.get("selected_video_id")
                if not selected_video_id:
                    return {
                        "status": "success",
                        "data": {
                            "query": body.query,
                            "mode": "temporal_event_search",
                            "video_selection": video_selection,
                            "selected_video": None,
                            "videos": [],
                            "events": [],
                        },
                    }
                selected_video_ids = [str(selected_video_id)]
            corpora = discover_temporal_corpus(
                output_root=Path("data/processed/video_understanding"),
                caption_dir=Path("data/metadata/caption"),
                asr_dir=Path("data/metadata/metadata_asr"),
                map_dir=Path("data/map-keyframes"),
                keyframe_dir=Path("data/keyframes"),
                batch_ids=body.batch_ids,
                video_ids=selected_video_ids,
            )
            result = TemporalEventSearch(
                corpora,
                # The query was parsed once before video selection. Avoid a
                # second Gemini request in the selected-video event phase.
                query_parser=None,
                temporal_verifier=temporal_verifier,
                video_dir=Path("data/videos") if temporal_verifier else None,
                verifier_candidates=body.qwen_candidate_events,
            ).search(
                parsed,
                top_k_videos=body.top_k_videos,
            )
            result["query"] = body.query
            result["query_parsing"] = {
                "mode": app.state.temporal_query_parser.mode,
                "model": app.state.temporal_query_parser.model,
            }
            result["video_selection"] = video_selection
            return {
                "status": "success",
                "data": result,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Temporal search failed: {exc}") from exc

    @app.post("/internal/search/temporal-videos")
    def search_temporal_videos(body: TemporalVideoRequest) -> dict:
        try:
            parsed = app.state.temporal_query_parser.parse(parse_temporal_query(body.query))
            selection = ElasticsearchVideoSelector(get_backend().client).select(
                parsed,
                batch_ids=body.batch_ids,
                video_ids=body.video_ids,
                top_k=body.top_k_videos,
                summary_weight=body.summary_weight,
                event_weight=body.event_weight,
            )
            return {
                "status": "success",
                "data": {
                    "query": body.query,
                    "mode": "temporal_video_selection",
                    "query_parsing": {
                        "mode": app.state.temporal_query_parser.mode,
                        "model": app.state.temporal_query_parser.model,
                    },
                    "query_plan": temporal_query_to_retrieval_spec(parsed),
                    "selected_video_id": selection.get("selected_video_id"),
                    "candidates": selection.get("candidates", []),
                    "video_selection": selection,
                },
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Temporal video search failed: {exc}") from exc

    return app


app = create_app()
