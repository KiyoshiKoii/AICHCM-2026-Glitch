from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_pipeline.retrieval.temporal_event_search import (
    TemporalCorpus,
    TemporalEventSearch,
    discover_temporal_corpus,
)
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query


ROOT = Path(__file__).resolve().parents[3]


def test_parser_keeps_duplicate_source_labels_in_order() -> None:
    parsed = parse_temporal_query(
        "Video nấu ăn\nE1: khoảnh khắc đầu tiên cắt nấm\nE2: khoảnh khắc cuối cùng đặt lên đĩa\nE2: sau đó chảo được đặt lên bếp"
    )

    assert parsed.shared_context == "Video nấu ăn"
    assert [event.event_index for event in parsed.events] == [1, 2, 3]
    assert [event.source_label for event in parsed.events] == ["E1", "E2", "E2"]
    assert parsed.events[0].required_anchor == "action_start"
    assert parsed.events[1].required_anchor == "last_complete"


def test_summary_only_query_is_kept_as_video_context() -> None:
    parsed = parse_temporal_query("Bản tin về thời tiết nóng tại Barcelona")

    assert parsed.shared_context == "Bản tin về thời tiết nóng tại Barcelona"
    assert len(parsed.events) == 1


def _temporal_corpus(video_id: str, location: str, event_text: str, frame_number: int) -> TemporalCorpus:
    frame = {
        "frame_id": f"{video_id}_f{frame_number:04d}",
        "keyframe_n": frame_number,
        "native_frame_idx": frame_number * 100,
        "timestamp_ms": frame_number * 1_000,
        "visual_text": event_text,
        "asr_text": "",
        "ocr_text": "",
    }
    return TemporalCorpus(
        video={
            "video_id": video_id,
            "summary_vi": f"Bản tin về thời tiết nóng tại {location}",
            "summary_en": "",
            "search_text": f"Bản tin thời tiết nóng {location}",
            "main_locations": [location],
            "main_entities": [],
        },
        events=[
            {
                "event_id": f"{video_id}_event_1",
                "description_vi": event_text,
                "search_text": event_text,
                "start_ms": frame["timestamp_ms"],
                "temporal_anchors": [],
                "_frames": [frame],
            }
        ],
        frames=[frame],
    )


def test_video_context_exact_location_outranks_generic_hot_weather() -> None:
    search = TemporalEventSearch(
        [
            _temporal_corpus("L22_V001", "Barcelona", "Nắng nóng kỷ lục tại Barcelona", 91),
            _temporal_corpus("L22_V025", "Tây Ban Nha", "Trượt tuyết tránh nóng tại Tây Ban Nha", 20),
        ]
    )

    result = search.search("Bản tin về thời tiết nóng tại Barcelona", top_k_videos=2)

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["selected_video"]["matched_context_entities"] == ["Barcelona"]
    assert result["events"][0]["frame_id"] == "L22_V001_f0091"


def test_all_explicit_concepts_outrank_an_olympic_paris_only_story() -> None:
    search = TemporalEventSearch(
        [
            _temporal_corpus(
                "L22_V001",
                "Pháp",
                "Linh vật Olympic Paris 2024 bán chạy",
                160,
            ),
            _temporal_corpus(
                "L22_V002",
                "Paris",
                "Hoạt động văn hóa tại Olympic Paris",
                140,
            ),
        ]
    )

    result = search.search("linh vật Olympic Paris", top_k_videos=2)

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["events"][0]["frame_id"] == "L22_V001_f0160"


def test_real_l22_temporal_search_returns_one_video_for_all_events() -> None:
    pilot = ROOT / "data/processed/video_understanding/L22/L22_V001/pilot"
    if not (pilot / "timeline.json").is_file():
        pytest.skip("requires the locally generated L22_V001 pilot")
    corpora = discover_temporal_corpus(
        output_root=ROOT / "data/processed/video_understanding",
        caption_dir=ROOT / "data/metadata/caption",
        asr_dir=ROOT / "data/metadata/metadata_asr",
        map_dir=ROOT / "data/map-keyframes",
        keyframe_dir=ROOT / "data/keyframes",
        batch_ids=["L22"],
        video_ids=["L22_V001"],
    )
    result = TemporalEventSearch(corpora).search(
        "news about Barcelona temperature record\n"
        "E1: first time Barcelona temperature record is visible\n"
        "E2: first time extreme heat is shown"
    )

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["selected_video"]["total_events"] == 2
    assert result["events"]
    assert all(item["frame_id"].startswith("L22_V001_f") for item in result["events"])
    assert all(item["native_frame_idx"] >= 0 for item in result["events"])


def test_timeline_v2_contains_only_temporal_fields_inside_existing_pilot() -> None:
    path = ROOT / "data/processed/video_understanding/L22/L22_V001/pilot/timeline.json"
    if not path.is_file():
        pytest.skip("requires the locally generated L22_V001 pilot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "video-timeline-v2"
    assert isinstance(payload.get("events"), list)
    assert all("temporal_anchors" in event for event in payload["events"])
