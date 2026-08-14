"""Strict Elasticsearch mappings for video, event and frame retrieval layers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


VIDEO_INDEX_NAME = "semantic_videos_v1"
SEGMENT_INDEX_NAME = "semantic_segments_v1"
FRAME_INDEX_NAME = "semantic_frames_v5"
EVENT_INDEX_NAME = "semantic_events_v1"


def _settings() -> dict[str, Any]:
    return {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "folded_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    }


_VIDEO_DEFINITION = {
    "settings": _settings(),
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "document_type": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "content_type": {"type": "keyword"},
            "duration_ms": {"type": "integer"},
            "summary_vi": {"type": "text", "analyzer": "folded_text"},
            "summary_en": {"type": "text", "analyzer": "folded_text"},
            "main_topics": {"type": "text", "analyzer": "folded_text"},
            "main_entities": {"type": "text", "analyzer": "folded_text"},
            "main_locations": {"type": "text", "analyzer": "folded_text"},
            "search_text": {"type": "text", "analyzer": "folded_text"},
            "segment_count": {"type": "integer"},
            "event_count": {"type": "integer"},
            "content_profile": {"type": "keyword"},
            "profile_confidence": {"type": "float"},
        },
    },
}

_EVENT_DEFINITION = {
    "settings": _settings(),
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "document_type": {"type": "keyword"},
            "event_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "parent_segment_id": {"type": "keyword"},
            "content_profile": {"type": "keyword"},
            "event_type": {"type": "keyword"},
            "order": {"type": "integer"},
            "subject": {"type": "text", "analyzer": "folded_text"},
            "action": {"type": "text", "analyzer": "folded_text"},
            "object": {"type": "text", "analyzer": "folded_text"},
            "target": {"type": "text", "analyzer": "folded_text"},
            "attributes": {"type": "text", "analyzer": "folded_text"},
            "description_vi": {"type": "text", "analyzer": "folded_text"},
            "pre_state": {"type": "text", "analyzer": "folded_text"},
            "transition_state": {"type": "text", "analyzer": "folded_text"},
            "post_state": {"type": "text", "analyzer": "folded_text"},
            "start_ms": {"type": "integer"},
            "end_ms": {"type": "integer"},
            "keyframe_refs": {"type": "integer"},
            "asr_segment_refs": {"type": "integer"},
            "search_aliases": {"type": "text", "analyzer": "folded_text"},
            "search_text": {"type": "text", "analyzer": "folded_text"},
            "confidence": {"type": "float"},
            "uncertain": {"type": "boolean"},
            "temporal_anchors": {
                "type": "object",
                "dynamic": "strict",
                "properties": {
                    "anchor_type": {"type": "keyword"},
                    "frame_id": {"type": "keyword"},
                    "keyframe_n": {"type": "integer"},
                    "timestamp_ms": {"type": "integer"},
                    "native_frame_idx": {"type": "integer"},
                    "confidence": {"type": "float"},
                },
            },
        },
    },
}

_SEGMENT_DEFINITION = {
    "settings": _settings(),
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "document_type": {"type": "keyword"},
            "segment_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "folded_text"},
            "summary": {"type": "text", "analyzer": "folded_text"},
            "topics": {"type": "text", "analyzer": "folded_text"},
            "entities": {"type": "text", "analyzer": "folded_text"},
            "locations": {"type": "text", "analyzer": "folded_text"},
            "start_ms": {"type": "integer"},
            "end_ms": {"type": "integer"},
            "keyframe_refs": {"type": "integer"},
            "representative_keyframe_refs": {"type": "integer"},
            "asr_segment_refs": {"type": "integer"},
            "asr_text": {"type": "text", "analyzer": "folded_text"},
            "search_text": {"type": "text", "analyzer": "folded_text"},
        },
    },
}

_FRAME_DEFINITION = {
    "settings": _settings(),
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "document_type": {"type": "keyword"},
            "frame_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "keyframe_n": {"type": "integer"},
            "native_frame_idx": {"type": "integer"},
            "timestamp_ms": {"type": "integer"},
            "segment_ids": {"type": "keyword"},
            "event_ids": {"type": "keyword"},
            "is_segment_representative": {"type": "boolean"},
            "visual_text": {"type": "text", "analyzer": "folded_text"},
            "asr_text": {"type": "text", "analyzer": "folded_text"},
            "ocr_text": {"type": "text", "analyzer": "folded_text"},
        },
    },
}


def video_index_definition() -> dict[str, Any]:
    return deepcopy(_VIDEO_DEFINITION)


def segment_index_definition() -> dict[str, Any]:
    return deepcopy(_SEGMENT_DEFINITION)


def frame_index_definition() -> dict[str, Any]:
    return deepcopy(_FRAME_DEFINITION)


def event_index_definition() -> dict[str, Any]:
    return deepcopy(_EVENT_DEFINITION)
