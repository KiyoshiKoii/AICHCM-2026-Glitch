import pytest

from backend.config import Settings
from backend.schemas.search import ParsedQuery, TextSearchRequest
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
    assert TextSearchRequest(query="a person").use_rerank is False


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
