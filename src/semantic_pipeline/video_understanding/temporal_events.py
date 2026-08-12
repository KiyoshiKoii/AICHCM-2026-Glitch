"""Compact temporal events derived from the canonical video timeline.

The event layer deliberately lives inside ``timeline.json``.  It gives the
temporal retrieval mode a stable, queryable representation without creating
candidate/event files beside the three published pilot artifacts.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


PROFILE_KEYWORDS = {
    "cooking_procedure": {
        "cook", "cooking", "kitchen", "ingredient", "mushroom", "tofu", "pan",
        "stove", "oil", "fry", "cut", "knife", "plate", "bowl", "food",
    },
    "performance": {
        "lion", "dragon", "pillar", "pole", "performer", "performance", "stage",
        "dance", "spinning", "jumping", "landing", "bowing", "audience",
    },
    "news": {
        "news", "studio", "anchor", "report", "headline", "broadcast", "journalist",
        "interview", "city", "minister", "government", "temperature",
    },
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _tokens(value: str) -> set[str]:
    return {item for item in TOKEN_RE.findall(_fold(value)) if len(item) > 1}


def _frame_text(frame: Any) -> str:
    raw = getattr(frame, "raw_metadata", {})
    values = [
        str(raw.get(field, ""))
        for field in (
            "caption",
            "detailed_caption",
            "caption_vi",
            "detailed_caption_vi",
            "ocr_text",
            "news_ticker_text",
        )
    ]
    for detection in raw.get("detections", []):
        if isinstance(detection, dict):
            values.extend(
                str(detection.get(field, ""))
                for field in ("label", "description", "description_vi", "action")
            )
            attributes = detection.get("attributes", [])
            if isinstance(attributes, list):
                values.extend(str(item) for item in attributes)
    return " ".join(values)


def classify_content_profile(frames: Iterable[Any], asr_segments: Iterable[Any]) -> tuple[str, float]:
    """Classify a video using sampled visual/ASR text without another artifact."""

    frame_list = list(frames)
    asr_text = " ".join(str(getattr(item, "text", "")) for item in asr_segments)
    visual_text = " ".join(_frame_text(frame) for frame in frame_list[:: max(1, len(frame_list) // 80)])
    corpus = _tokens(f"{visual_text} {asr_text}")
    scores = {
        profile: len(corpus & keywords) / max(1, len(keywords))
        for profile, keywords in PROFILE_KEYWORDS.items()
    }
    profile, score = max(scores.items(), key=lambda item: (item[1], item[0] == "news"))
    if score < 0.02:
        return "general_event", 0.25
    return profile, round(min(0.99, 0.45 + score), 3)


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted({int(item) for item in value if isinstance(item, int) or str(item).isdigit()})


def _anchor(anchor_type: str, frame: Any, confidence: float = 0.75) -> dict[str, Any]:
    return {
        "anchor_type": anchor_type,
        "frame_id": frame.frame_id,
        "keyframe_n": frame.keyframe_n,
        "timestamp_ms": frame.timestamp_ms,
        "native_frame_idx": frame.native_frame_idx,
        "confidence": confidence,
    }


def build_temporal_events(
    timeline_segments: list[dict[str, Any]],
    frames: list[Any],
    *,
    content_profile: str,
) -> list[dict[str, Any]]:
    """Create one compact event per evidence scene.

    Scenes are already the smallest evidence units produced by the pipeline.
    Keeping their boundaries makes first/last-frame queries deterministic and
    avoids duplicating raw captions or ASR transcripts in the published file.
    """

    frame_by_n = {frame.keyframe_n: frame for frame in frames}
    events: list[dict[str, Any]] = []
    for segment in timeline_segments:
        segment_id = str(segment.get("segment_id", ""))
        segment_title = str(segment.get("title", "")).strip()
        segment_summary = str(segment.get("summary", "")).strip()
        generic_title = not segment_title or segment_title.casefold().startswith("news story ")
        segment_terms = [
            segment_title,
            *([] if generic_title else [segment_summary]),
            *([] if generic_title else [str(item) for item in segment.get("topics", [])]),
            *([] if generic_title else [str(item) for item in segment.get("entities", [])]),
            *([] if generic_title else [str(item) for item in segment.get("locations", [])]),
        ]
        scenes = segment.get("scenes", [])
        if not isinstance(scenes, list):
            continue
        for scene_order, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                continue
            keyframes = _as_int_list(scene.get("keyframe_refs"))
            available = [frame_by_n[n] for n in keyframes if n in frame_by_n]
            if not available:
                continue
            available.sort(key=lambda frame: frame.timestamp_ms)
            first = available[0]
            last = available[-1]
            representative_n = scene.get("representative_keyframe_n")
            representative = frame_by_n.get(int(representative_n)) if representative_n is not None else None
            representative = representative or available[len(available) // 2]
            aliases = list(dict.fromkeys(
                item for item in [
                    *segment_terms,
                    str(scene.get("scene_type", "")),
                ] if item.strip()
            ))
            event_id = f"{segment_id}_event_{scene_order:04d}"
            events.append(
                {
                    "event_id": event_id,
                    "order": len(events) + 1,
                    "parent_segment_id": segment_id,
                    "content_profile": content_profile,
                    "event_type": str(scene.get("scene_type", "visual_transition")),
                    "subject": "",
                    "action": str(scene.get("scene_type", "visual_transition")),
                    "object": "",
                    "target": "",
                    "attributes": [],
                    "description_vi": segment_title or segment_summary or str(scene.get("scene_type", "")),
                    "pre_state": "Trạng thái trước evidence scene chưa được xác định tự động",
                    "transition_state": "Có chuyển tiếp hình ảnh hoặc nội dung trong scene",
                    "post_state": "Trạng thái sau evidence scene chưa được xác định tự động",
                    "start_ms": int(scene.get("start_ms", first.timestamp_ms)),
                    "end_ms": int(scene.get("end_ms", last.timestamp_ms)),
                    "temporal_anchors": [
                        _anchor("action_start", first),
                        _anchor("representative", representative, 0.8),
                        _anchor("action_end", last),
                    ],
                    "keyframe_refs": [frame.keyframe_n for frame in available],
                    "asr_segment_refs": _as_int_list(scene.get("asr_segment_refs")),
                    "search_aliases": aliases,
                    "confidence": 0.75,
                    "uncertain": True,
                }
            )
    events.sort(key=lambda item: (int(item.get("start_ms", 0)), int(item.get("end_ms", 0)), item["event_id"]))
    for order, event in enumerate(events, start=1):
        event["order"] = order
    return events


def event_search_text(event: dict[str, Any]) -> str:
    aliases = [
        str(value)
        for value in event.get("search_aliases", [])
        if str(value).strip() and len(str(value)) <= 300
    ]
    values = [
        event.get("description_vi", ""),
        event.get("event_type", ""),
        event.get("subject", ""),
        event.get("action", ""),
        event.get("object", ""),
        event.get("target", ""),
        *event.get("attributes", []),
        *aliases,
    ]
    return " ".join(str(value) for value in values if str(value).strip())
