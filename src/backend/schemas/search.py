import re
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


_BATCH_ID_PATTERN = re.compile(r"^L(?:2[1-9]|30)$", re.IGNORECASE)
_VIDEO_ID_PATTERN = re.compile(r"^L(?:2[1-9]|30)_V\d{3}$", re.IGNORECASE)
_VIDEO_SUFFIX_PATTERN = re.compile(r"^V\d{3}$", re.IGNORECASE)
_VIDEO_NUMBER_PATTERN = re.compile(r"^\d{1,3}$")


def _normalize_filter_values(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;\s]+", value)
    if not isinstance(value, list):
        return value
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_batch_ids(value: Any) -> list[str]:
    values = _normalize_filter_values(value)
    invalid = [item for item in values if not _BATCH_ID_PATTERN.fullmatch(item)]
    if invalid:
        raise ValueError(f"batch_ids must contain only L21-L30; invalid={invalid}")
    return values


def _normalize_video_ids(value: Any) -> list[str]:
    values = _normalize_filter_values(value)
    result: list[str] = []
    for item in values:
        if _VIDEO_NUMBER_PATTERN.fullmatch(item):
            item = f"V{int(item):03d}"
        if not (_VIDEO_ID_PATTERN.fullmatch(item) or _VIDEO_SUFFIX_PATTERN.fullmatch(item)):
            raise ValueError(f"invalid video id: {item}")
        result.append(item)
    return result


class ObjectQuery(BaseModel):
    """One object constraint whose terms must refer to the same object."""

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
    """One subject-predicate-object relation constraint."""

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
    object_queries: list[ObjectQuery] = Field(
        default_factory=list,
        max_length=5,
        description="Object-specific phrases, preserving attributes bound to one object.",
    )
    spatial_queries: list[SpatialQuery] = Field(
        default_factory=list,
        max_length=5,
        description="Explicit subject-predicate-object spatial relations.",
    )
    interaction_queries: list[InteractionQuery] = Field(
        default_factory=list,
        max_length=5,
        description="Subject-action-object interactions bound to one relation row.",
    )
    ocr_queries: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Exact text expected to be visible in the frame.",
    )
    program_queries: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Program, series, broadcaster, or broadcast-slot constraints.",
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

    @field_validator("ocr_queries", "program_queries", mode="before")
    @classmethod
    def accept_query_phrase_strings(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("ocr_queries", "program_queries")
    @classmethod
    def normalize_query_phrases(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = " ".join(value.split()).strip(" ,;")
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=100, ge=1, le=100)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    text_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight of semantic text/Elasticsearch retrieval.",
    )
    visual_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight of visual/Qwen retrieval.",
    )
    asr_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relative weight of timestamped ASR retrieval.",
    )
    use_rerank: bool = Field(
        default=False,
        description="Call Gemini to re-rank the retrieved results.",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("batch_ids", mode="before")
    @classmethod
    def normalize_batch_filters(cls, value: Any) -> list[str]:
        return _normalize_batch_ids(value)

    @field_validator("video_ids", mode="before")
    @classmethod
    def normalize_video_filters(cls, value: Any) -> list[str]:
        return _normalize_video_ids(value)

    @model_validator(mode="after")
    def validate_fusion_weights(self) -> "TextSearchRequest":
        if self.text_weight + self.visual_weight + self.asr_weight <= 0:
            raise ValueError(
                "text_weight and visual_weight cannot both be zero unless asr_weight is positive"
            )
        return self


class TemporalEventOptions(BaseModel):
    """Retrieval/verifier controls belonging to one explicitly numbered event."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(pattern=r"(?i)^E[1-8]$")
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    use_rerank: bool = False
    requires_after_previous: bool = False
    verify_camera_motion: bool = False
    motion_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("event_id")
    @classmethod
    def normalize_event_id(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_kis_weights(self) -> "TemporalEventOptions":
        if self.text_weight + self.visual_weight <= 0:
            raise ValueError("event text_weight and visual_weight cannot both be zero")
        return self


class TemporalEventSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k_videos: int = Field(default=20, ge=1, le=100)
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    event_options: list[TemporalEventOptions] = Field(default_factory=list, max_length=8)

    @field_validator("query")
    @classmethod
    def normalize_temporal_query(cls, value: str) -> str:
        normalized = "\n".join(" ".join(line.split()) for line in value.splitlines()).strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("batch_ids", "video_ids", mode="before")
    @classmethod
    def normalize_temporal_ids(cls, value: Any) -> Any:
        return _normalize_filter_values(value)

    @model_validator(mode="after")
    def validate_kis_fusion_weights(self) -> "TemporalEventSearchRequest":
        if self.text_weight + self.visual_weight <= 0:
            raise ValueError("text_weight and visual_weight cannot both be zero")
        event_ids = [item.event_id for item in self.event_options]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_options must not contain duplicate event_id values")
        return self


class TemporalVideoSearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=2, max_length=5000)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k_videos: int = Field(default=100, ge=1, le=100)
    summary_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    kis_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("kis_weight", "event_weight"),
    )
    use_rerank: bool = False
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    event_options: list[TemporalEventOptions] = Field(default_factory=list, max_length=8)

    @field_validator("query")
    @classmethod
    def normalize_temporal_query(cls, value: str) -> str:
        normalized = "\n".join(" ".join(line.split()) for line in value.splitlines()).strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("batch_ids", "video_ids", mode="before")
    @classmethod
    def normalize_temporal_ids(cls, value: Any) -> Any:
        return _normalize_filter_values(value)

    @model_validator(mode="after")
    def validate_fusion_weights(self) -> "TemporalVideoSearchRequest":
        if self.summary_weight + self.kis_weight <= 0:
            raise ValueError("summary_weight and kis_weight cannot both be zero")
        if self.text_weight + self.visual_weight <= 0:
            raise ValueError("text_weight and visual_weight cannot both be zero")
        event_ids = [item.event_id for item in self.event_options]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_options must not contain duplicate event_id values")
        return self


class ASRSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=50, ge=1, le=200)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("batch_ids", mode="before")
    @classmethod
    def normalize_batch_filters(cls, value: Any) -> list[str]:
        return _normalize_batch_ids(value)

    @field_validator("video_ids", mode="before")
    @classmethod
    def normalize_video_filters(cls, value: Any) -> list[str]:
        return _normalize_video_ids(value)


class UpstreamResult(BaseModel):
    frame_id: str = Field(min_length=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    frame_id: str
    video_name: str | None = None
    frame_index: int | None = None
    score: float
    thumbnail_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchData(BaseModel):
    total_results: int
    results: list[SearchHit]
    llm_reranked_results: list[SearchHit] | None = Field(default=None, description="Kết quả sau khi LLM chấm điểm lại")


class TextSearchResponse(BaseModel):
    status: str = "success"
    message: str = "Retrieved successfully"
    data: SearchData
