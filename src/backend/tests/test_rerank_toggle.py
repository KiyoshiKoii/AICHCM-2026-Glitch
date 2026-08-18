import pytest

from backend.config import Settings
from backend.schemas.search import (
    ParsedQuery,
    SearchHit,
    TemporalVideoSearchRequest,
    TextSearchRequest,
)
from backend.services.llm_reranker import GeminiReRanker
from backend.services.search_orchestrator import SearchService


class FakeParser:
    async def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(visual_prompt=query, semantic_keywords=[query])


class FakePipeline:
    async def search_text(self, payload: dict) -> list[dict]:
        return [{"frame_id": "L21_V001_f0001", "score": 0.9}]


class FakeReranker:
    client = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, hits: list) -> list:
        self.calls.append((query, [hit.frame_id for hit in hits]))
        return hits


class RecordingGeminiReranker(GeminiReRanker):
    """Avoid Gemini/network I/O while exposing hierarchical batch calls."""

    def __init__(self) -> None:
        self.client = object()
        self.calls: list[list[str]] = []

    async def _rerank_once(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        self.calls.append([hit.frame_id for hit in hits])
        return hits


def build_service() -> tuple[SearchService, FakeReranker]:
    service = SearchService(
        settings=Settings(),
        parser=FakeParser(),
        dev1=FakePipeline(),
        dev2=FakePipeline(),
    )
    reranker = FakeReranker()
    service.reranker = reranker
    return service, reranker


def test_text_search_request_disables_rerank_by_default():
    request = TextSearchRequest(query="a person")

    assert request.use_rerank is False
    assert request.top_k == 100
    assert request.text_weight == 0.5
    assert request.visual_weight == 0.5


def test_text_search_request_rejects_zero_fusion_weights():
    with pytest.raises(ValueError, match="cannot both be zero"):
        TextSearchRequest(query="a person", text_weight=0.0, visual_weight=0.0)


def test_text_search_request_normalizes_batch_and_video_filters():
    request = TextSearchRequest(
        query="a person",
        batch_ids="L21, l22",
        video_ids="v006, L22_V030",
    )

    assert request.batch_ids == ["L21", "L22"]
    assert request.video_ids == ["V006", "L22_V030"]


def test_text_search_request_rejects_unknown_batch_filter():
    with pytest.raises(ValueError, match="only L21-L30"):
        TextSearchRequest(query="a person", batch_ids=["L20"])


def test_temporal_video_request_exposes_summary_event_weights():
    request = TemporalVideoSearchRequest(
        query="A cycling race",
        summary_weight=0.6,
        event_weight=0.4,
    )

    assert request.summary_weight == 0.6
    assert request.event_weight == 0.4


def test_temporal_video_request_rejects_zero_fusion_weights():
    with pytest.raises(ValueError, match="cannot both be zero"):
        TemporalVideoSearchRequest(
            query="A cycling race",
            summary_weight=0.0,
            event_weight=0.0,
        )


@pytest.mark.asyncio
async def test_search_skips_gemini_reranking_when_disabled():
    service, reranker = build_service()

    response = await service.search_text("a person", 10, use_rerank=False)

    assert reranker.calls == []
    assert response.data.llm_reranked_results is None


@pytest.mark.asyncio
async def test_search_runs_gemini_reranking_when_enabled():
    service, reranker = build_service()

    response = await service.search_text("a person", 10, use_rerank=True)

    assert reranker.calls == [("a person", ["L21_V001_f0001"])]
    assert response.data.llm_reranked_results is not None


@pytest.mark.asyncio
async def test_gemini_reranker_uses_two_50_image_passes_then_one_shared_final():
    reranker = RecordingGeminiReranker()
    hits = [
        SearchHit(
            frame_id=f"L21_V001_f{index:04d}",
            score=1.0,
            thumbnail_url=f"/media/thumbnails/L21_V001/{index:04d}.jpg",
        )
        for index in range(100)
    ]

    result = await reranker.rerank("a person", hits)

    assert sorted(len(call) for call in reranker.calls) == [50, 50, 50]
    assert set(reranker.calls[-1]) == {
        *(f"L21_V001_f{index:04d}" for index in range(25)),
        *(f"L21_V001_f{index:04d}" for index in range(50, 75)),
    }
    assert [hit.frame_id for hit in result] == [
        *(f"L21_V001_f{index:04d}" for index in range(25)),
        *(f"L21_V001_f{index:04d}" for index in range(50, 75)),
        *(f"L21_V001_f{index:04d}" for index in range(25, 50)),
        *(f"L21_V001_f{index:04d}" for index in range(75, 100)),
    ]
