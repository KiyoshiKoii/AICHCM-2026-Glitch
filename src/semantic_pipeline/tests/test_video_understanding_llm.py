from __future__ import annotations

import pytest

from semantic_pipeline.video_understanding.llm_schemas import (
    VIDEO_SUMMARY_SCHEMA,
    WINDOW_SUMMARY_SCHEMA,
)
from semantic_pipeline.video_understanding.summarizer import (
    NewsSummarizer,
    _is_program_intro,
    _normalize_payload,
)


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
    prompt = NewsSummarizer._window_prompt({"window_id": "w1"})
    assert "natural Vietnamese" in prompt
    assert "JSON object with an events array" in prompt


def test_program_intro_is_not_a_news_event() -> None:
    assert _is_program_intro(
        "Chương trình tin tức 60 giây",
        "Giới thiệu các nội dung chính trong ngày.",
    )
    assert not _is_program_intro(
        "Tin tức về vụ cháy",
        "Chương trình tin tức giới thiệu vụ cháy tại Bình Định.",
    )
