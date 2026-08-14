from __future__ import annotations

import pytest

from semantic_pipeline.video_understanding.llm_schemas import (
    VIDEO_SUMMARY_SCHEMA,
    WINDOW_SUMMARY_SCHEMA,
)
from semantic_pipeline.video_understanding.models import MicroScene, RuntimeFrame
from semantic_pipeline.video_understanding.summarizer import (
    NewsSummarizer,
    _fallback_candidates,
    _is_program_intro,
    _normalize_payload,
)
from semantic_pipeline.video_understanding.timeline_builder import _compact_frame


def test_window_schema_requires_object_with_events() -> None:
    assert WINDOW_SUMMARY_SCHEMA["type"] == "object"
    assert WINDOW_SUMMARY_SCHEMA["required"] == ["events"]
    assert WINDOW_SUMMARY_SCHEMA["properties"]["events"]["type"] == "array"


def test_video_schema_has_distinct_bilingual_fields() -> None:
    assert VIDEO_SUMMARY_SCHEMA["required"] == [
        "summary_vi",
        "summary_en",
        "main_topics",
        "main_entities",
        "main_locations",
        "main_actions",
        "main_objects",
        "main_visual_states",
        "chronological_outline",
    ]


def test_top_level_window_array_is_compatibility_wrapped() -> None:
    payload = _normalize_payload([{"title": "Sự kiện"}], "window")
    assert payload == {"events": [{"title": "Sự kiện"}]}


def test_top_level_video_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="video summary response must be an object"):
        _normalize_payload([], "video summary")


def test_non_object_window_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="window response must be an object"):
        _normalize_payload("not-json-object", "window")


def test_window_prompt_requires_vietnamese() -> None:
    prompt = NewsSummarizer._window_prompt(
        {"window_id": "w1"},
        domain_hint="food and cooking",
    )
    assert "natural Vietnamese" in prompt
    assert "JSON object with an events array" in prompt
    assert "food and cooking" in prompt
    assert "spatial relations" in prompt
    assert "Vietnamese news video" not in prompt


def _cooking_frame() -> RuntimeFrame:
    return RuntimeFrame(
        video_id="L26_V001",
        keyframe_n=1,
        frame_id="L26_V001_f0001",
        timestamp_ms=1_000,
        fps=25.0,
        native_frame_idx=25,
        raw_metadata={
            "caption": "A cook cuts a mushroom over a bowl.",
            "detailed_caption_vi": "Đầu bếp dùng dao cắt nấm phía trên một chiếc tô.",
            "detections": [
                {"object_id": "cook_0", "label": "cook", "action": "cutting"},
                {"object_id": "mushroom_0", "label": "mushroom", "action": ""},
                {"object_id": "bowl_0", "label": "bowl", "action": ""},
            ],
            "spatial_relations": [
                {
                    "subject_id": "mushroom_0",
                    "predicate": "above",
                    "object_id": "bowl_0",
                }
            ],
        },
    )


def test_compact_frame_preserves_resolved_spatial_relations() -> None:
    compact = _compact_frame(_cooking_frame())

    assert compact["spatial_relations"] == [
        {
            "subject_id": "mushroom_0",
            "subject_label": "mushroom",
            "predicate": "above",
            "object_id": "bowl_0",
            "object_label": "bowl",
        }
    ]


def test_metadata_fallback_keeps_generic_actions_objects_and_states() -> None:
    frame = _cooking_frame()
    scene = MicroScene(
        scene_id="L26_V001_scene_0001",
        scene_type="food_preparation",
        frame_indices=[1],
        start_ms=1_000,
        end_ms=1_000,
        representative_keyframe_n=1,
        asr_segment_indices=[],
    )

    candidate = _fallback_candidates([scene], [frame], {})[0]

    assert candidate.actions == ["cook: cutting"]
    assert {"cook", "mushroom", "bowl"}.issubset(candidate.objects)
    assert candidate.visual_states == ["mushroom above bowl"]
    assert not candidate.title.startswith("News story")


def test_program_intro_is_not_a_news_event() -> None:
    assert _is_program_intro(
        "Chương trình tin tức 60 giây",
        "Giới thiệu các nội dung chính trong ngày.",
    )
    assert not _is_program_intro(
        "Tin tức về vụ cháy",
        "Chương trình tin tức giới thiệu vụ cháy tại Bình Định.",
    )
