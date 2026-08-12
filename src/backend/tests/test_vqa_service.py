import pytest

from backend.core.errors import VQAUnavailableError
from backend.schemas.search import SearchData, SearchHit, TextSearchResponse
from backend.schemas.vqa import VQAAnswer, VQARequest
from backend.services.vqa_service import VQAService


def hit(frame_id: str, score: float) -> SearchHit:
    return SearchHit(
        frame_id=frame_id,
        video_name="L21_V001",
        frame_index=90,
        score=score,
        thumbnail_url=f"/media/thumbnails/{frame_id}.jpg",
    )


class FakeSearchService:
    def __init__(self, response: TextSearchResponse) -> None:
        self.response = response
        self.calls = []

    async def search_text(
        self,
        query: str,
        top_k: int,
        *,
        use_rerank: bool = True,
    ) -> TextSearchResponse:
        self.calls.append((query, top_k, use_rerank))
        return self.response


class FakeAnswerer:
    available = True

    def __init__(self) -> None:
        self.calls = []

    async def answer_batch(self, question: str, search_hits: list[SearchHit]) -> dict[str, VQAAnswer]:
        self.calls.append((question, [hit.frame_id for hit in search_hits]))
        return {
            hit.frame_id: VQAAnswer(answer=f"answer for {hit.frame_id}", confidence=0.9)
            for hit in search_hits
        }


@pytest.mark.asyncio
async def test_uses_reranked_hits_and_preserves_their_order():
    rrf_hit = hit("L21_V001_f0001", 0.5)
    reranked_hits = [hit("L21_V001_f0003", 0.9), hit("L21_V001_f0002", 0.8)]
    search = FakeSearchService(
        TextSearchResponse(
            data=SearchData(
                total_results=3,
                results=[rrf_hit, *reranked_hits],
                llm_reranked_results=reranked_hits,
            )
        )
    )
    answerer = FakeAnswerer()
    service = VQAService(search_service=search, answerer=answerer)

    response = await service.answer(
        VQARequest(
            query="award ceremony on stage",
            question="How many people are there?",
            retrieval_top_k=50,
            answer_top_k=2,
            use_rerank=True,
        )
    )

    assert search.calls == [("award ceremony on stage", 50, True)]
    assert [candidate.frame_id for candidate in response.data.candidates] == [
        "L21_V001_f0003",
        "L21_V001_f0002",
    ]
    assert answerer.calls == [(
        "How many people are there?",
        ["L21_V001_f0003", "L21_V001_f0002"],
    )]
    assert [hit.frame_id for hit in response.data.results] == [
        "L21_V001_f0001",
        "L21_V001_f0003",
        "L21_V001_f0002",
    ]
    assert response.data.use_rerank is True


@pytest.mark.asyncio
async def test_falls_back_to_rrf_hits_when_reranking_is_unavailable():
    rrf_hits = [hit("L21_V001_f0001", 0.7), hit("L21_V001_f0002", 0.6)]
    search = FakeSearchService(
        TextSearchResponse(data=SearchData(total_results=2, results=rrf_hits))
    )
    service = VQAService(
        search_service=search,
        answerer=FakeAnswerer(),
    )

    response = await service.answer(
        VQARequest(query="a person", question="What is visible?", answer_top_k=1)
    )

    assert [candidate.frame_id for candidate in response.data.candidates] == ["L21_V001_f0001"]
    assert search.calls == [("a person", 50, False)]
    assert response.data.llm_reranked_results is None


@pytest.mark.asyncio
async def test_does_not_use_reranked_order_when_disabled():
    rrf_hits = [hit("L21_V001_f0001", 0.7), hit("L21_V001_f0002", 0.6)]
    reranked_hits = [hit("L21_V001_f0002", 0.9), hit("L21_V001_f0001", 0.8)]
    search = FakeSearchService(
        TextSearchResponse(
            data=SearchData(
                total_results=2,
                results=rrf_hits,
                llm_reranked_results=reranked_hits,
            )
        )
    )
    answerer = FakeAnswerer()
    service = VQAService(search_service=search, answerer=answerer)

    response = await service.answer(
        VQARequest(
            query="a person",
            question="What is visible?",
            answer_top_k=1,
            use_rerank=False,
        )
    )

    assert [candidate.frame_id for candidate in response.data.candidates] == [
        "L21_V001_f0001"
    ]
    assert answerer.calls == [("What is visible?", ["L21_V001_f0001"])]
    assert response.data.llm_reranked_results is None


@pytest.mark.asyncio
async def test_rejects_requests_when_the_answerer_is_not_configured():
    class UnavailableAnswerer:
        available = False

        async def answer_batch(self, question: str, search_hits: list[SearchHit]) -> dict[str, VQAAnswer]:
            raise AssertionError("answer_batch must not be called")

    search = FakeSearchService(TextSearchResponse(data=SearchData(total_results=0, results=[])))
    service = VQAService(search_service=search, answerer=UnavailableAnswerer())

    with pytest.raises(VQAUnavailableError):
        await service.answer(VQARequest(query="a person", question="Who is this?"))
