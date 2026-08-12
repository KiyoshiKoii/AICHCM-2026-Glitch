from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_pipeline.retrieval.temporal_event_search import (
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
