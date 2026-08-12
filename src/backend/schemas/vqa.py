from pydantic import BaseModel, Field, field_validator, model_validator

from backend.schemas.search import SearchHit


class VQARequest(BaseModel):
    """A visual event description plus the question to answer from candidate frames."""

    query: str = Field(min_length=2, max_length=2000)
    question: str = Field(min_length=2, max_length=1000)
    use_rerank: bool = Field(
        default=False,
        description="Run the optional Gemini listwise re-ranker before answering.",
    )
    retrieval_top_k: int = Field(default=50, ge=1, le=100)
    answer_top_k: int = Field(default=10, ge=1, le=20)

    @field_validator("query", "question")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_candidate_limits(self) -> "VQARequest":
        if self.answer_top_k > self.retrieval_top_k:
            raise ValueError("answer_top_k must not exceed retrieval_top_k")
        return self


class VQAAnswer(BaseModel):
    """Structured output expected from a vision-language model for one image."""

    answer: str = Field(min_length=1, max_length=300)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("answer must not be blank")
        return normalized


class VQAFrameAnswer(VQAAnswer):
    frame_id: str = Field(min_length=1)


class VQABatchAnswer(BaseModel):
    answers: list[VQAFrameAnswer]


class VQACandidate(SearchHit):
    answer: str
    confidence: float | None = None


class VQAData(BaseModel):
    total_candidates: int
    candidates: list[VQACandidate]
    # The full RRF list is kept so the client can show the retrieval baseline
    # below the answered/reranked candidates when reranking is enabled.
    results: list[SearchHit] = Field(default_factory=list)
    # ``None`` means reranking was disabled or unavailable.  When present this
    # preserves the ordering returned by the Gemini re-ranker; answers remain
    # in ``candidates`` because only the configured answer_top_k frames are
    # sent to the VQA model.
    llm_reranked_results: list[SearchHit] | None = None
    use_rerank: bool = False


class VQAResponse(BaseModel):
    status: str = "success"
    message: str = "Answered from selected VQA candidates"
    data: VQAData
