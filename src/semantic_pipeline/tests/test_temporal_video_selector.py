from __future__ import annotations

import json
from pathlib import Path

from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query
from semantic_pipeline.retrieval.temporal_video_selector import (
    ElasticsearchVideoSelector,
    build_selector_documents,
    build_video_selection_query,
    iter_selector_actions,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_selector_documents_needs_no_caption_or_frame_artifacts(tmp_path: Path) -> None:
    pilot = tmp_path / "L24/L24_V033/pilot"
    _write_json(
        pilot / "video_summary.json",
        {
            "video_id": "L24_V033",
            "content_type": "performance",
            "content_profile": "performance",
            "profile_confidence": 0.9,
            "duration_ms": 1000,
            "summary_vi": "Lân đen trắng biểu diễn trên cột.",
            "summary_en": "Black and white lion dance.",
            "main_topics": ["múa lân"],
            "main_entities": ["lân đen trắng"],
            "main_locations": ["sân khấu"],
            "search_text": "lân đen trắng cột rồng",
        },
    )
    _write_json(
        pilot / "timeline.json",
        {
            "video_id": "L24_V033",
            "events": [{"event_id": "event-1"}],
            "segments": [
                {
                    "segment_id": "story-1",
                    "title": "Lân chào rồng",
                    "summary": "Con lân tiến đến đầu rồng.",
                    "topics": ["múa lân"],
                    "entities": ["rồng"],
                    "locations": ["sân khấu"],
                    "actions": ["cúi chào"],
                    "objects": ["đầu rồng"],
                    "visual_states": ["lân đứng trước rồng"],
                    "start_ms": 500,
                    "end_ms": 900,
                }
            ],
        },
    )

    video, segments = build_selector_documents(pilot)

    assert video["video_id"] == "L24_V033"
    assert video["segment_count"] == 1
    assert video["event_count"] == 1
    assert segments[0]["start_ms"] == 500
    assert "cúi chào" in segments[0]["search_text"]
    assert "đầu rồng" in segments[0]["search_text"]


def test_iter_selector_actions_indexes_summary_even_without_timeline(tmp_path: Path) -> None:
    pilot = tmp_path / "L22/L22_V001/pilot"
    _write_json(
        pilot / "video_summary.json",
        {
            "video_id": "L22_V001",
            "summary_vi": "Bản tin tổng hợp.",
            "search_text": "Tháp Rùa Hồ Hoàn Kiếm",
        },
    )

    actions = list(iter_selector_actions(tmp_path, batch_ids=["L22"]))

    assert len(actions) == 1
    assert actions[0]["_index"] == "semantic_videos_v1"
    assert actions[0]["_id"] == "L22_V001"


def test_video_selection_query_filters_the_whole_batch_by_video_prefix() -> None:
    parsed = parse_temporal_query("Múa lân đen trắng.\nE1: Lân chào rồng.")

    query = build_video_selection_query(
        parsed,
        fields=["summary_vi", "search_text"],
        batch_ids=["l24"],
    )

    filters = query["bool"]["filter"]
    assert filters == [
        {
            "bool": {
                "should": [{"prefix": {"video_id": "L24_"}}],
                "minimum_should_match": 1,
            }
        }
    ]
    expanded = query["bool"]["should"][-1]["multi_match"]["query"]
    assert "Lân chào rồng" in expanded


def test_video_selection_query_requires_a_strong_full_query_match_for_news_batches() -> None:
    parsed = parse_temporal_query("Japan food festival is the largest in the world")

    query = build_video_selection_query(
        parsed,
        fields=["summary_vi", "search_text"],
        batch_ids=["L21", "L22"],
    )

    should = query["bool"]["should"]
    assert len(should) == 1
    match = should[0]["multi_match"]
    assert match["query"] == parsed.shared_context
    assert match["minimum_should_match"] == "90%"


def test_video_selection_query_keeps_permissive_expansion_for_non_news_batches() -> None:
    parsed = parse_temporal_query("A cycling race.\nE1: Three cyclists ride in a line.")

    query = build_video_selection_query(
        parsed,
        fields=["summary_vi", "search_text"],
        batch_ids=["L23"],
    )

    assert query["bool"]["should"][-1]["multi_match"]["minimum_should_match"] == "15%"


class _FakeClient:
    def search(self, *, index, **kwargs):
        del kwargs
        if index == "semantic_videos_v1":
            hits = [
                {"_score": 8.0, "_source": {"video_id": "L24_V033", "summary_vi": "lân đen trắng"}},
                {"_score": 7.0, "_source": {"video_id": "L24_V011", "summary_vi": "lân bạc"}},
            ]
        else:
            hits = [
                {
                    "_score": 9.0,
                    "_source": {
                        "segment_id": "v033-story-1",
                        "video_id": "L24_V033",
                        "title": "Lân chào rồng",
                        "start_ms": 500,
                        "end_ms": 900,
                    },
                },
                {
                    "_score": 4.0,
                    "_source": {
                        "segment_id": "v011-story-1",
                        "video_id": "L24_V011",
                        "title": "Lân trên cột",
                        "start_ms": 100,
                        "end_ms": 400,
                    },
                },
                {
                    "_score": 8.0,
                    "_source": {
                        "segment_id": "v033-story-2",
                        "video_id": "L24_V033",
                        "title": "Rồng cử động đầu",
                        "start_ms": 900,
                        "end_ms": 1000,
                    },
                },
            ]
        return {"hits": {"hits": hits}}


def test_elasticsearch_selector_fuses_video_and_segment_hits_before_event_search() -> None:
    parsed = parse_temporal_query("Múa lân đen trắng.\nE1: Lân chào rồng.")

    result = ElasticsearchVideoSelector(_FakeClient()).select(parsed, batch_ids=["L24"], top_k=2)

    assert result["selected_video_id"] == "L24_V033"
    assert [item["video_id"] for item in result["candidates"]] == ["L24_V033", "L24_V011"]
    assert len(result["candidates"][0]["segment_evidence"]) == 2


class _OpposingScoreClient:
    def search(self, *, index, **kwargs):
        del kwargs
        if index == "semantic_videos_v1":
            hits = [
                {"_score": 10.0, "_source": {"video_id": "L23_V001"}},
                {"_score": 2.0, "_source": {"video_id": "L23_V002"}},
            ]
        else:
            hits = [
                {"_score": 1.0, "_source": {"video_id": "L23_V001"}},
                {"_score": 10.0, "_source": {"video_id": "L23_V002"}},
            ]
        return {"hits": {"hits": hits}}


class _CrossPolicyScoreClient:
    def search(self, *, index, **kwargs):
        del kwargs
        if index == "semantic_videos_v1":
            hits = [
                {"_score": 2.0, "_source": {"video_id": "L22_V030"}},
                {"_score": 100.0, "_source": {"video_id": "L26_V057"}},
            ]
        else:
            hits = []
        return {"hits": {"hits": hits}}


def test_elasticsearch_selector_applies_user_video_ranking_weights() -> None:
    parsed = parse_temporal_query("Cuoc dua xe dap.\nE1: Ba tay dua di thanh hang doc.")
    selector = ElasticsearchVideoSelector(_OpposingScoreClient())

    summary_first = selector.select(
        parsed,
        batch_ids=["L23"],
        top_k=2,
        summary_weight=0.8,
        event_weight=0.2,
    )
    event_first = selector.select(
        parsed,
        batch_ids=["L23"],
        top_k=2,
        summary_weight=0.2,
        event_weight=0.8,
    )

    assert summary_first["selected_video_id"] == "L23_V001"
    assert event_first["selected_video_id"] == "L23_V002"
    assert summary_first["weights"] == {"summary": 0.8, "event": 0.2}
    assert summary_first["candidates"][0]["summary_score"] == 1.0
    assert summary_first["candidates"][0]["event_score"] == 0.1


def test_elasticsearch_selector_normalizes_news_and_general_query_policies_separately() -> None:
    parsed = parse_temporal_query("Japan food festival is the largest in the world")

    result = ElasticsearchVideoSelector(_CrossPolicyScoreClient()).select(
        parsed,
        top_k=2,
        summary_weight=1.0,
        event_weight=0.0,
    )

    assert result["selected_video_id"] == "L22_V030"
    assert [item["summary_score"] for item in result["candidates"]] == [1.0, 1.0]
