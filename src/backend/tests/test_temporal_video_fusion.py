from __future__ import annotations

import pytest

from backend.config import Settings
from backend.schemas.search import ParsedQuery, SearchHit
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
    )

    data = response["data"]
    assert data["mode"] == "temporal_events_kis_selected_video"
    assert data["selected_video"]["video_id"] == "L26_V002"
    assert data["events"][0]["event_id"] == "E1"
    assert data["events"][0]["frame_id"] == "L26_V002_f0001"
    assert data["events"][0]["native_frame_idx"] == 0
    assert visual.payloads[0]["video_ids"] == ["L26_V002"]
    assert semantic.text_payloads[0]["video_ids"] == ["L26_V002"]
    assert visual.payloads[0]["top_k"] == 20
    assert "interaction_queries" in semantic.text_payloads[0]


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
