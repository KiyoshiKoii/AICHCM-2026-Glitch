"""JSON schemas used to constrain Gemini video-understanding responses."""

from __future__ import annotations

NEWS_EVENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "scene_refs": {"type": "array", "items": {"type": "string"}},
        "asr_segment_refs": {"type": "array", "items": {"type": "integer"}},
        "uncertain": {"type": "boolean"},
    },
    "required": [
        "title",
        "summary",
        "topics",
        "entities",
        "locations",
        "scene_refs",
        "asr_segment_refs",
        "uncertain",
    ],
}

WINDOW_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "events": {
            "type": "array",
            "items": NEWS_EVENT_SCHEMA,
        }
    },
    "required": ["events"],
}

VIDEO_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary_vi": {"type": "string"},
        "summary_en": {"type": "string"},
        "main_topics": {"type": "array", "items": {"type": "string"}},
        "main_entities": {"type": "array", "items": {"type": "string"}},
        "main_locations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary_vi",
        "summary_en",
        "main_topics",
        "main_entities",
        "main_locations",
    ],
}
