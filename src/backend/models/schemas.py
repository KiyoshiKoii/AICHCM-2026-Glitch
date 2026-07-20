from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParsedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_prompt: str = Field(
        min_length=1,
        max_length=1000,
        description="English description of observable visual content.",
    )
    semantic_keywords: list[str] = Field(
        min_length=1,
        max_length=20,
        description="English semantic keywords and close synonyms.",
    )

    @field_validator("visual_prompt")
    @classmethod
    def normalize_visual_prompt(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("visual_prompt must not be blank")
        return normalized

    @field_validator("semantic_keywords", mode="before")
    @classmethod
    def accept_comma_separated_keywords(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.split(",")
        return value

    @field_validator("semantic_keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split()).strip(" ,;")
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        if not result:
            raise ValueError("semantic_keywords must contain a non-blank keyword")
        return result


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())


class UpstreamResult(BaseModel):
    frame_id: str = Field(min_length=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    frame_id: str
    rrf_score: float
    thumbnail_url: str
    source_ranks: dict[str, int]
    source_scores: dict[str, float | None]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextSearchResponse(BaseModel):
    query: str
    parsed_query: ParsedQuery
    count: int
    results: list[SearchHit]
    warnings: list[str] = Field(default_factory=list)


class FrameContextItem(BaseModel):
    frame_id: str
    offset: int
    thumbnail_url: str


class FrameContextResponse(BaseModel):
    current_frame_id: str
    frames: list[FrameContextItem]
