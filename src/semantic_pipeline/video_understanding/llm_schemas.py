"""JSON schemas used to constrain evidence-grounded video summaries."""

from __future__ import annotations

VIDEO_EPISODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "visual_states": {"type": "array", "items": {"type": "string"}},
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
        "actions",
        "objects",
        "visual_states",
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
            "items": VIDEO_EPISODE_SCHEMA,
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
        "main_actions": {"type": "array", "items": {"type": "string"}},
        "main_objects": {"type": "array", "items": {"type": "string"}},
        "main_visual_states": {"type": "array", "items": {"type": "string"}},
        "chronological_outline": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary_vi",
        "summary_en",
        "main_topics",
        "main_entities",
        "main_locations",
        "main_actions",
        "main_objects",
        "main_visual_states",
        "chronological_outline",
    ],
}

# Backwards-compatible import for callers that still use the old news-specific name.
NEWS_EVENT_SCHEMA = VIDEO_EPISODE_SCHEMA
