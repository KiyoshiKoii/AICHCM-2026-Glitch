from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_pipeline.retrieval.hierarchical_video_search import (
    HierarchicalVideoSearch,
    build_hierarchical_documents,
    iter_elasticsearch_actions,
)
from semantic_pipeline.retrieval.hierarchical_index_definition import (
    frame_index_definition,
    segment_index_definition,
    video_index_definition,
)
from semantic_pipeline.video_understanding.pipeline import _build_search_text, _retrieval_coverage


ROOT = Path(__file__).resolve().parents[3]


def test_video_search_text_contains_every_story_summary() -> None:
    summary = {"summary_vi": "Bản tin", "summary_en": "News", "main_topics": ["thời sự"]}
    stories = [
        {
            "segment_id": "story-1",
            "title": "Tin Barcelona",
            "summary": "Nhiệt độ cao nhất trong 110 năm.",
            "topics": ["thời tiết"],
            "entities": [],
            "locations": ["Barcelona"],
        },
        {
            "segment_id": "story-2",
            "title": "Tin Bình Định",
            "summary": "Cháy xưởng sản xuất ván gỗ.",
            "topics": [],
            "entities": [],
            "locations": ["Bình Định"],
            "actions": ["đầu bếp cắt nấm"],
            "objects": ["nấm", "dao"],
            "visual_states": ["nấm nằm trong tô"],
        },
    ]
    search_text = _build_search_text(summary, stories)
    coverage = _retrieval_coverage(stories, search_text)

    assert "Nhiệt độ cao nhất trong 110 năm." in search_text
    assert "Cháy xưởng sản xuất ván gỗ." in search_text
    assert "đầu bếp cắt nấm" in search_text
    assert "nấm nằm trong tô" in search_text
    assert coverage["segments_in_search_text"] == 2
    assert not coverage["missing_segment_titles"]
    assert not coverage["missing_segment_summaries"]


def _actual_l22_documents():
    pilot = ROOT / "data/processed/video_understanding/L22/L22_V001/pilot"
    if not (pilot / "video_summary.json").is_file():
        pytest.skip("requires the locally generated L22_V001 pilot")
    return build_hierarchical_documents(
        pilot_dir=pilot,
        caption_path=ROOT / "data/metadata/caption/L22/L22_V001.json",
        asr_path=ROOT / "data/metadata/metadata_asr/L22_V001.json",
        map_path=ROOT / "data/map-keyframes/L22_V001.csv",
        keyframe_dir=ROOT / "data/keyframes/L22_V001",
    )


def test_l22_hierarchical_search_finds_barcelona_story_and_valid_frame() -> None:
    documents = _actual_l22_documents()
    result = HierarchicalVideoSearch(documents).search("nhiệt độ Barcelona cao nhất trong 110 năm")

    assert result["video"]["video_id"] == "L22_V001"
    assert "barcelona" in result["segments"][0]["title"].casefold()
    assert result["frames"]
    frame = result["frames"][0]
    assert result["segments"][0]["segment_id"] in frame["segment_ids"]
    assert frame["native_frame_idx"] >= 0


def test_l22_documents_generate_three_index_action_types() -> None:
    documents = _actual_l22_documents()
    actions = list(
        iter_elasticsearch_actions(
            documents,
            video_index="semantic_videos_v1",
            segment_index="semantic_segments_v1",
            frame_index="semantic_frames_v5",
        )
    )

    assert len(actions) == 1 + len(documents.segments) + len(documents.frames)
    assert actions[0]["_index"] == "semantic_videos_v1"
    assert {action["_index"] for action in actions[1:]} == {
        "semantic_segments_v1",
        "semantic_frames_v5",
    }


def test_hierarchical_mappings_cover_every_document_field() -> None:
    definitions = {
        "video": video_index_definition(),
        "segment": segment_index_definition(),
        "frame": frame_index_definition(),
    }
    assert all(definition["mappings"]["dynamic"] == "strict" for definition in definitions.values())
    assert "search_text" in definitions["video"]["mappings"]["properties"]
    assert "segment_id" in definitions["segment"]["mappings"]["properties"]
    assert "native_frame_idx" in definitions["frame"]["mappings"]["properties"]
