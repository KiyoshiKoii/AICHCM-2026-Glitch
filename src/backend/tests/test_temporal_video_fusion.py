from __future__ import annotations

import pytest

from backend.config import Settings
from backend.schemas.search import ParsedQuery, SearchHit
from backend.services.camera_motion_verifier import CameraMotionResult
from backend.services.search_orchestrator import SearchService
from backend.services.temporal_video_fusion import (
    KISQuery,
    aggregate_kis_rankings,
    build_kis_queries,
    fuse_summary_and_kis,
)


def _hit(frame_id: str, score: float) -> SearchHit:
    return SearchHit(frame_id=frame_id, score=score, thumbnail_url="")


def test_build_kis_queries_reuses_one_shot_temporal_visual_plan() -> None:
    queries = build_kis_queries(
        "Cooking video\nE1: The cook wraps filling in a leaf.",
        {
            "video_context": "Cooking video with raw fish",
            "events": [
                {
                    "event_id": "E1",
                    "description": "The cook wraps filling in a leaf.",
                    "retrieval_prompts": [
                        "hands placing fish filling on a green leaf",
                        "fish filling wrapped in a leaf",
                    ],
                    "target_predicates": ["filling rests on the leaf"],
                }
            ],
        },
    )

    assert len(queries) == 1
    assert queries[0].visual_prompt == "hands placing fish filling on a green leaf"
    assert queries[0].prompt_variants == ("fish filling wrapped in a leaf",)
    assert "Cooking video with raw fish" in queries[0].semantic_keywords
    assert "filling rests on the leaf" in queries[0].semantic_keywords


def test_build_kis_queries_routes_explicit_shot_sequence_to_observable_stages() -> None:
    queries = build_kis_queries(
        "Cảnh phim lần lượt giới thiệu nguyên liệu qua 3 chuyển cảnh: "
        "máy quay chéo lên tới nguyên liệu hải sản đầu tiên; "
        "quay cận nguyên liệu hải sản thứ hai rồi chuyển sang rau củ nhiều màu; "
        "cuối cùng là toàn cảnh nguyên liệu.",
        None,
        infer_sequence=True,
    )

    assert [item.event_id for item in queries] == ["S1", "S2", "S3", "S4"]
    assert [item.sequence_position for item in queries] == [1, 2, 3, 4]
    assert "cá thác lác" in queries[1].visual_prompt
    assert queries[2].description == "chuyển sang rau củ nhiều màu"


def test_build_kis_queries_keeps_plain_kis_query_on_single_frame_path() -> None:
    queries = build_kis_queries("hai người mang cây dù", None)

    assert len(queries) == 1
    assert queries[0].description == "hai người mang cây dù"
    assert queries[0].sequence_position is None


def test_aggregate_kis_rankings_rewards_ordered_compact_chain() -> None:
    stages = [
        KISQuery(
            f"S{index}",
            f"stage {index}",
            f"stage {index}",
            (),
            (),
            index,
            requires_after_previous=index > 1,
        )
        for index in range(1, 4)
    ]

    def positioned_hit(frame_id: str, score: float, position: int) -> SearchHit:
        return SearchHit(
            frame_id=frame_id,
            frame_index=position,
            score=score,
            thumbnail_url="",
        )

    scores, evidence = aggregate_kis_rankings(
        [
            (
                stages[0],
                [
                    positioned_hit("L26_V002_f0003", 1.0, 300),
                    positioned_hit("L26_V001_f0001", 0.9, 100),
                ],
            ),
            (
                stages[1],
                [
                    positioned_hit("L26_V002_f0002", 1.0, 200),
                    positioned_hit("L26_V001_f0002", 0.9, 110),
                ],
            ),
            (
                stages[2],
                [
                    positioned_hit("L26_V002_f0001", 1.0, 100),
                    positioned_hit("L26_V001_f0003", 0.9, 120),
                ],
            ),
        ]
    )

    assert scores["L26_V001"] > scores["L26_V002"]
    assert all(item["sequence_selected"] for item in evidence["L26_V001"])
    assert not any(item["sequence_selected"] for item in evidence["L26_V002"])


def test_aggregate_kis_rankings_rewards_cross_event_video_coverage() -> None:
    first = KISQuery("E1", "first", "first", (), ("first",))
    second = KISQuery("E2", "second", "second", (), ("second",))

    scores, evidence = aggregate_kis_rankings(
        [
            (first, [_hit("L26_V002_f0001", 1.0), _hit("L26_V001_f0001", 0.9)]),
            (second, [_hit("L26_V002_f0002", 1.0), _hit("L26_V003_f0001", 0.95)]),
        ]
    )

    assert scores["L26_V002"] > scores["L26_V001"]
    assert scores["L26_V002"] > scores["L26_V003"]
    assert {item["event_id"] for item in evidence["L26_V002"]} == {"E1", "E2"}
    assert all("thumbnail_url" in item for item in evidence["L26_V002"])


def test_selected_video_order_constraint_promotes_later_event_frame() -> None:
    service = SearchService.__new__(SearchService)
    first = KISQuery("E1", "first", "first", (), ())
    second = KISQuery(
        "E2",
        "second",
        "second",
        (),
        (),
        sequence_position=2,
        requires_after_previous=True,
    )
    rankings = [
        [SearchHit(frame_id="L26_V074_f0010", frame_index=100, score=1.0, thumbnail_url="")],
        [
            SearchHit(frame_id="L26_V074_f0009", frame_index=90, score=1.0, thumbnail_url=""),
            SearchHit(frame_id="L26_V074_f0011", frame_index=110, score=0.9, thumbnail_url=""),
        ],
    ]

    result = service._apply_selected_event_order([first, second], rankings)

    assert result[1][0].frame_id == "L26_V074_f0011"
    assert result[1][0].metadata["event_order"]["satisfies_order"] is True
    assert result[1][1].metadata["event_order"]["satisfies_order"] is False


@pytest.mark.asyncio
async def test_camera_verification_keeps_kis_order_when_video_is_unavailable() -> None:
    class _UnavailableVerifier:
        @staticmethod
        def has_explicit_camera_constraint(_: str) -> bool:
            return True

        @staticmethod
        def verify(**_: object) -> CameraMotionResult:
            return CameraMotionResult(
                score=0.5,
                reason="Video file is unavailable.",
                details={"available": False},
            )

    service = SearchService.__new__(SearchService)
    service.camera_motion_verifier = _UnavailableVerifier()
    spec = KISQuery(
        "E1",
        "camera tilts up",
        "camera tilts up",
        (),
        (),
        verify_camera_motion=True,
        motion_weight=1.0,
    )
    hits = [SearchHit(frame_id="L26_V074_f0010", frame_index=100, score=0.9, thumbnail_url="")]

    result = await service._apply_camera_motion_verification(spec, hits)

    assert result[0].score == 0.9
    assert "camera_motion" not in result[0].metadata


def test_fuse_summary_and_kis_can_promote_visual_match() -> None:
    candidates = fuse_summary_and_kis(
        [
            {"video_id": "L26_V001", "summary_score": 1.0, "summary_vi": "generic"},
            {"video_id": "L26_V002", "summary_score": 0.5, "summary_vi": "target"},
        ],
        {"L26_V002": 1.0},
        {"L26_V002": [{"frame_id": "L26_V002_f0001"}]},
        summary_weight=0.4,
        kis_weight=0.6,
        top_k=2,
    )

    assert [item["video_id"] for item in candidates] == ["L26_V002", "L26_V001"]
    assert candidates[0]["kis_score"] == 1.0
    assert candidates[0]["kis_evidence"][0]["frame_id"] == "L26_V002_f0001"



class _Parser:
    async def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(visual_prompt=query, semantic_keywords=[query])


class _VisualPipeline:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def search_text(self, payload: dict) -> list[dict]:
        self.payloads.append(payload)
        return [{"frame_id": "L26_V002_f0001", "score": 0.95}]


class _SemanticPipeline:
    def __init__(self) -> None:
        self.text_payloads: list[dict] = []
        self.temporal_payloads: list[dict] = []

    async def search_text(self, payload: dict) -> list[dict]:
        self.text_payloads.append(payload)
        return [{"frame_id": "L26_V002_f0001", "score": 8.0}]

    async def search_temporal_videos(self, payload: dict) -> dict:
        self.temporal_payloads.append(payload)
        return {
            "status": "success",
            "data": {
                "query_plan": {
                    "video_context": "A cooking show",
                    "events": [
                        {
                            "event_id": "E1",
                            "description": "A cook wraps fish in a green leaf",
                            "retrieval_prompts": ["hands wrapping fish in a green leaf"],
                        }
                    ],
                },
                "candidates": [
                    {"video_id": "L26_V001", "summary_score": 1.0},
                    {"video_id": "L26_V002", "summary_score": 0.5},
                ],
            },
        }


class _Reranker:
    client = object()

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        self.queries.append(query)
        return hits


@pytest.mark.asyncio
async def test_selected_temporal_video_uses_scoped_kis_candidates() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )

    response = await service.search_temporal_events(
        "A cooking video\nE1: A cook wraps fish in a green leaf",
        batch_ids=["L26"],
        video_ids=["L26_V002"],
        text_weight=0.2,
        visual_weight=0.8,
    )

    data = response["data"]
    assert data["mode"] == "temporal_events_kis_selected_video"
    assert data["selected_video"]["video_id"] == "L26_V002"
    assert data["events"][0]["event_id"] == "E1"
    assert data["events"][0]["frame_id"] == "L26_V002_f0001"
    assert data["events"][0]["native_frame_idx"] == 0
    assert data["kis"]["weights"] == {"text": 0.2, "visual": 0.8}
    assert visual.payloads[0]["video_ids"] == ["L26_V002"]
    assert semantic.text_payloads[0]["video_ids"] == ["L26_V002"]
    assert visual.payloads[0]["top_k"] == 20
    assert "interaction_queries" in semantic.text_payloads[0]


@pytest.mark.asyncio
async def test_selected_temporal_video_applies_controls_per_event() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )
    reranker = _Reranker()
    service.reranker = reranker
    calls: list[tuple[str, float, float, bool]] = []
    original_search_text = service.search_text

    async def record_search_text(*args, **kwargs):
        calls.append((args[0], kwargs["text_weight"], kwargs["visual_weight"], kwargs["use_rerank"]))
        return await original_search_text(*args, **kwargs)

    service.search_text = record_search_text  # type: ignore[method-assign]
    response = await service.search_temporal_events(
        "Cycling video\nE1: riders enter the bridge\nE2: riders leave the bridge",
        batch_ids=["L23"],
        video_ids=["L23_V001"],
        event_options=[
            {
                "event_id": "E1",
                "text_weight": 0.2,
                "visual_weight": 0.8,
                "use_rerank": True,
            },
            {
                "event_id": "E2",
                "text_weight": 0.7,
                "visual_weight": 0.3,
                "requires_after_previous": True,
            },
        ],
    )

    assert sorted(calls) == [
        ("riders enter the bridge", 0.2, 0.8, True),
        ("riders leave the bridge", 0.7, 0.3, False),
    ]
    assert reranker.queries == ["riders enter the bridge"]
    assert response["data"]["kis"]["queries"] == [
        {
            "event_id": "E1",
            "description": "riders enter the bridge",
            "visual_prompt": "riders enter the bridge",
            "weights": {"text": 0.2, "visual": 0.8},
            "requires_after_previous": False,
            "verify_camera_motion": False,
            "motion_weight": 0.7,
        },
        {
            "event_id": "E2",
            "description": "riders leave the bridge",
            "visual_prompt": "riders leave the bridge",
            "weights": {"text": 0.7, "visual": 0.3},
            "requires_after_previous": True,
            "verify_camera_motion": False,
            "motion_weight": 0.7,
        },
    ]


@pytest.mark.asyncio
async def test_temporal_video_search_uses_visual_and_caption_kis() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )

    response = await service.search_temporal_videos(
        "A cooking show\nE1: A cook wraps fish in a green leaf",
        batch_ids=["L26"],
        top_k_videos=2,
        summary_weight=0.4,
        kis_weight=0.6,
    )

    assert response["data"]["selected_video_id"] == "L26_V002"
    assert response["data"]["weights"] == {"summary": 0.4, "kis": 0.6}
    assert response["data"]["candidates"][0]["kis_score"] == 1.0
    assert visual.payloads[0]["visual_prompt"] == "A cook wraps fish in a green leaf"
    assert "interaction_queries" in semantic.text_payloads[0]
    assert semantic.temporal_payloads[0]["summary_weight"] == 1.0
    assert semantic.temporal_payloads[0]["event_weight"] == 0.0


@pytest.mark.asyncio
async def test_temporal_video_search_applies_kis_text_visual_weights() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )
    calls: list[dict[str, float]] = []
    original_search_text = service.search_text

    async def record_search_text(*args, **kwargs):
        calls.append(
            {
                "text_weight": kwargs["text_weight"],
                "visual_weight": kwargs["visual_weight"],
            }
        )
        return await original_search_text(*args, **kwargs)

    service.search_text = record_search_text  # type: ignore[method-assign]
    response = await service.search_temporal_videos(
        "A cooking show\nE1: A cook wraps fish in a green leaf",
        batch_ids=["L26"],
        top_k_videos=2,
        text_weight=0.3,
        visual_weight=0.7,
    )

    assert calls == [{"text_weight": 0.3, "visual_weight": 0.7}]
    assert response["data"]["kis"]["weights"] == {"text": 0.3, "visual": 0.7}


@pytest.mark.asyncio
async def test_temporal_video_search_uses_raw_event_wording_for_kis() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )

    await service.search_temporal_videos(
        "A cooking show\nE1: Exact wording from the user",
        batch_ids=["L26"],
        top_k_videos=2,
    )

    assert visual.payloads[0]["visual_prompt"] == "Exact wording from the user"


@pytest.mark.asyncio
async def test_temporal_video_search_reranks_each_event_only_when_enabled() -> None:
    visual = _VisualPipeline()
    semantic = _SemanticPipeline()
    service = SearchService(
        settings=Settings(),
        parser=_Parser(),
        dev1=visual,
        dev2=semantic,
    )
    reranker = _Reranker()
    service.reranker = reranker

    response = await service.search_temporal_videos(
        "A cooking show\nE1: A cook wraps fish in a green leaf",
        batch_ids=["L26"],
        top_k_videos=2,
        use_rerank=True,
    )

    assert reranker.queries == ["A cook wraps fish in a green leaf"]
    assert response["data"]["kis"]["gemini_rerank"] is True
