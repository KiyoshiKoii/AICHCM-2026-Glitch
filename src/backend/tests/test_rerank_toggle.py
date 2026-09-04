import pytest

from backend.config import Settings
from backend.schemas.search import (
    ParsedQuery,
    SearchHit,
    TemporalEventSearchRequest,
    TemporalVideoSearchRequest,
    TextSearchRequest,
)
from backend.services.llm_reranker import GeminiReRanker
from backend.services.search_orchestrator import SearchService


class FakeParser:
    async def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(visual_prompt=query, semantic_keywords=[query])


class RewritingParser:
    async def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(visual_prompt="a translated, shortened visual prompt", semantic_keywords=[query])


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search_text(self, payload: dict) -> list[dict]:
        self.calls.append(payload)
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
    assert request.asr_weight == 0.0


def test_text_search_request_rejects_zero_fusion_weights():
    with pytest.raises(ValueError, match="cannot both be zero"):
        TextSearchRequest(query="a person", text_weight=0.0, visual_weight=0.0)


def test_text_search_request_accepts_asr_only_fusion():
    request = TextSearchRequest(
        query="a person speaks",
        text_weight=0.0,
        visual_weight=0.0,
        asr_weight=1.0,
    )

    assert request.asr_weight == 1.0


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


def test_temporal_video_request_exposes_summary_kis_weights():
    request = TemporalVideoSearchRequest(
        query="A cycling race",
        summary_weight=0.6,
        kis_weight=0.4,
    )

    assert request.summary_weight == 0.6
    assert request.kis_weight == 0.4
    assert request.use_rerank is False
    assert request.text_weight == 0.5
    assert request.visual_weight == 0.5


def test_temporal_video_request_defaults_to_one_hundred_candidates():
    request = TemporalVideoSearchRequest(query="A cycling race")

    assert request.top_k_videos == 100


def test_temporal_video_request_accepts_gemini_rerank_toggle():
    request = TemporalVideoSearchRequest(query="A cycling race", use_rerank=True)

    assert request.use_rerank is True


def test_temporal_video_request_accepts_legacy_event_weight():
    request = TemporalVideoSearchRequest(query="A cycling race", event_weight=0.4)

    assert request.kis_weight == 0.4


def test_temporal_video_request_rejects_zero_fusion_weights():
    with pytest.raises(ValueError, match="cannot both be zero"):
        TemporalVideoSearchRequest(
            query="A cycling race",
            summary_weight=0.0,
            kis_weight=0.0,
        )


def test_temporal_video_request_rejects_zero_kis_text_visual_weights():
    with pytest.raises(ValueError, match="text_weight and visual_weight"):
        TemporalVideoSearchRequest(
            query="A cycling race",
            text_weight=0.0,
            visual_weight=0.0,
        )


def test_temporal_event_options_validate_and_normalize_per_event_controls():
    request = TemporalEventSearchRequest(
        query="E1: the cook opens the stove",
        event_options=[
            {
                "event_id": "e1",
                "text_weight": 0.3,
                "visual_weight": 0.7,
                "use_rerank": True,
                "verify_camera_motion": True,
                "motion_weight": 0.65,
            }
        ],
    )

    option = request.event_options[0]
    assert option.event_id == "E1"
    assert option.text_weight == 0.3
    assert option.visual_weight == 0.7
    assert option.use_rerank is True
    assert option.verify_camera_motion is True
    assert option.motion_weight == 0.65


def test_temporal_event_options_reject_duplicate_event_ids():
    with pytest.raises(ValueError, match="duplicate event_id"):
        TemporalVideoSearchRequest(
            query="E1: first\nE2: second",
            event_options=[
                {"event_id": "E1"},
                {"event_id": "e1"},
            ],
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
@pytest.mark.asyncio
async def test_qwen_visual_pipeline_always_uses_raw_user_query():
    dev1 = FakePipeline()
    service = SearchService(
        settings=Settings(),
        parser=RewritingParser(),
        dev1=dev1,
        dev2=FakePipeline(),
    )
    await service.search_text(
        "nguyên câu truy vấn dài của người dùng",
        10,
        use_rerank=False,
    )

    assert dev1.calls[0]["visual_prompt"] == "nguyên câu truy vấn dài của người dùng"


@pytest.mark.asyncio
async def test_search_fuses_asr_passage_with_frames_inside_its_time_window(monkeypatch) -> None:
    class ASRWindowPipeline:
        async def search_text(self, payload: dict) -> dict:
            return {"data": [{"frame_id": "L26_V001_f0008", "score": 0.9}]}

        async def search_asr(self, payload: dict) -> dict:
            assert payload["query"] == "squid with wine"
            return {
                "data": [
                    {
                        "asr_id": "L26_V001_asr_000009",
                        "video_id": "L26_V001",
                        "start_ms": 12_000,
                        "end_ms": 18_000,
                        "text": "add white wine and pepper to the squid",
                        "score": 1.0,
                    }
                ]
            }

    monkeypatch.setattr(
        "backend.utils.keyframe_mapper.get_nearest_keyframe_position",
        lambda _video_id, _timestamp_ms: (7, {"frame_index": 175, "timestamp_ms": 12_000, "fps": 25.0}),
    )
    monkeypatch.setattr(
        "backend.utils.keyframe_mapper.get_keyframe_position",
        lambda _video_id, ordinal: (
            {"frame_index": 200, "timestamp_ms": 15_000, "fps": 25.0}
            if ordinal == 8
            else {"frame_index": 175, "timestamp_ms": 12_000, "fps": 25.0}
        ),
    )
    pipeline = ASRWindowPipeline()
    service = SearchService(
        settings=Settings(gemini_api_key=None),
        parser=FakeParser(),
        dev1=pipeline,
        dev2=pipeline,
    )

    response = await service.search_text(
        "squid with wine",
        10,
        use_rerank=False,
        text_weight=0.3,
        visual_weight=0.3,
        asr_weight=0.4,
        batch_ids=["L26"],
    )

    hit = response.data.results[0]
    assert hit.frame_id == "L26_V001_f0008"
    assert hit.metadata["asr_temporal_match"] is True
    assert hit.metadata["transcript"] == "add white wine and pepper to the squid"
    assert hit.metadata["source_ranks"] == {"dev1": 1, "dev2": 1, "asr": 1}


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
