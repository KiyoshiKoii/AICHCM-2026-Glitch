"""Domain-agnostic micro-scene segmentation and evidence packets."""

from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Any, Iterable

from .models import ASRSegment, MicroScene, RuntimeFrame


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
TRANSITION_TERMS = {
    "tiếp theo",
    "sau đó",
    "trong một diễn biến khác",
    "chuyển sang",
    "bản tin tiếp theo",
}


def _tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {item.casefold() for item in TOKEN_RE.findall(value) if len(item) > 1}


def _text_signature(frame: RuntimeFrame) -> set[str]:
    raw = frame.raw_metadata
    parts: list[str] = []
    for key in ("caption", "detailed_caption", "caption_vi", "detailed_caption_vi"):
        parts.append(str(raw.get(key, "")))
    parts.append(CLOCK_RE.sub("", str(raw.get("ocr_text", ""))))
    parts.append(str(raw.get("news_ticker_text", "")))
    for detection in raw.get("detections", []):
        if not isinstance(detection, dict):
            continue
        parts.extend(
            str(detection.get(key, ""))
            for key in ("label", "description", "description_vi", "action")
        )
        attributes = detection.get("attributes", [])
        if isinstance(attributes, list):
            parts.extend(str(item) for item in attributes)
    for relation in _resolved_spatial_relations(raw):
        parts.extend(
            str(relation.get(key, ""))
            for key in ("subject_label", "predicate", "object_label")
        )
    return _tokens(" ".join(parts))


def _resolved_spatial_relations(raw: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        str(item.get("object_id", "")): str(item.get("label", ""))
        for item in raw.get("detections", [])
        if isinstance(item, dict) and item.get("object_id")
    }
    result: list[dict[str, str]] = []
    for relation in raw.get("spatial_relations", []):
        if not isinstance(relation, dict):
            continue
        subject_id = str(relation.get("subject_id", "")).strip()
        predicate = str(relation.get("predicate", "")).strip()
        object_id = str(relation.get("object_id", "")).strip()
        if not subject_id or not predicate or not object_id:
            continue
        result.append(
            {
                "subject_id": subject_id,
                "subject_label": labels.get(subject_id, subject_id),
                "predicate": predicate,
                "object_id": object_id,
                "object_label": labels.get(object_id, object_id),
            }
        )
    return result


def _object_signature(frame: RuntimeFrame) -> set[str]:
    result: set[str] = set()
    for detection in frame.raw_metadata.get("detections", []):
        if not isinstance(detection, dict):
            continue
        label = detection.get("label")
        action = detection.get("action")
        if label:
            result.add(str(label).casefold())
        if action:
            result.add(f"action:{str(action).casefold()}")
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _scene_type(frame: RuntimeFrame) -> str:
    text = " ".join(
        str(frame.raw_metadata.get(key, ""))
        for key in ("caption", "detailed_caption", "caption_vi", "detailed_caption_vi")
    ).casefold()
    action_text = " ".join(
        str(item.get("action", ""))
        for item in frame.raw_metadata.get("detections", [])
        if isinstance(item, dict)
    ).casefold()
    if any(
        term in text
        for term in ("kitchen", "ingredient", "knife", "pan", "stove", "chảo", "dao", "nấu")
    ):
        return "food_preparation"
    if any(
        term in text
        for term in (
            "lion dance",
            "dragon dance",
            "múa lân",
            "múa rồng",
            "performer",
            "stage performance",
        )
    ):
        return "performance"
    if any(
        term in text
        for term in ("cyclist", "bicycle race", "peloton", "vận động viên", "cuộc đua")
    ):
        return "sport_activity"
    if any(
        term in text
        for term in ("news desk", "news studio", "trường quay", "phát thanh viên", "presenter")
    ):
        return "presenter_or_studio"
    if any(term in text for term in ("map", "bản đồ", "graphic", "đồ họa", "screen displays")):
        return "graphic_or_map"
    if any(term in text for term in ("interview", "phỏng vấn", "microphone", "micrô")):
        return "interview_or_dialogue"
    if any(term in text for term in ("logo", "opening", "intro", "sunset cityscape")):
        return "title_or_intro"
    if action_text:
        return "observable_activity"
    return "visual_scene"


def adaptive_continuity_guard(frames: list[RuntimeFrame]) -> float:
    gaps = [
        (frames[index].timestamp_ms - frames[index - 1].timestamp_ms) / 1000
        for index in range(1, len(frames))
    ]
    if not gaps:
        return 8.0
    return max(8.0, min(15.0, 2.5 * median(gaps)))


def _asr_transition(frame: RuntimeFrame, segments: dict[int, ASRSegment]) -> bool:
    text = " ".join(segments[index].text.casefold() for index in frame.nearby_asr_segment_indices)
    return any(term in text for term in TRANSITION_TERMS)


def build_micro_scenes(
    frames: list[RuntimeFrame],
    segments: dict[int, ASRSegment],
    max_scene_seconds: float = 20.0,
) -> list[MicroScene]:
    if not frames:
        return []
    guard = adaptive_continuity_guard(frames)
    scenes: list[MicroScene] = []
    current: list[RuntimeFrame] = [frames[0]]
    current_text = _text_signature(frames[0])
    current_objects = _object_signature(frames[0])
    boundary_reason = "start"

    def close_scene(reason: str) -> None:
        nonlocal current, current_text, current_objects, boundary_reason
        frame_positions = [item.keyframe_n for item in current]
        representative = max(
            current,
            key=lambda item: (
                not bool(item.raw_metadata.get("quality_flags")),
                len(_object_signature(item)),
                len(_text_signature(item)),
            ),
        )
        asr_indices = sorted({index for item in current for index in item.asr_segment_indices})
        scenes.append(
            MicroScene(
                scene_id=f"{current[0].video_id}_scene_{len(scenes) + 1:04d}",
                scene_type=_scene_type(representative),
                frame_indices=frame_positions,
                start_ms=current[0].timestamp_ms,
                end_ms=current[-1].timestamp_ms,
                representative_keyframe_n=representative.keyframe_n,
                asr_segment_indices=asr_indices,
                boundary_reason=reason or boundary_reason,
            )
        )
        current = []
        current_text = set()
        current_objects = set()
        boundary_reason = ""

    for frame in frames[1:]:
        gap = (frame.timestamp_ms - current[-1].timestamp_ms) / 1000
        frame_text = _text_signature(frame)
        frame_objects = _object_signature(frame)
        text_similarity = _jaccard(current_text, frame_text)
        object_similarity = _jaccard(current_objects, frame_objects)
        type_change = _scene_type(frame) != _scene_type(current[-1])
        split = False
        reason = ""
        if gap > guard:
            split, reason = True, "timestamp_gap"
        elif (frame.timestamp_ms - current[0].timestamp_ms) / 1000 >= max_scene_seconds:
            split, reason = True, "max_scene_duration"
        elif type_change and text_similarity < 0.35:
            split, reason = True, "visual_type_change"
        elif text_similarity < 0.18 and object_similarity < 0.25:
            split, reason = True, "semantic_change"
        elif _asr_transition(frame, segments):
            split, reason = True, "asr_transition"
        if split:
            close_scene(reason)
            current.append(frame)
            current_text = _text_signature(frame)
            current_objects = _object_signature(frame)
        else:
            current.append(frame)
            current_text |= frame_text
            current_objects |= frame_objects
    close_scene("end")
    return scenes


def _compact_frame(frame: RuntimeFrame) -> dict[str, Any]:
    raw = frame.raw_metadata
    detections = []
    for detection in raw.get("detections", []):
        if not isinstance(detection, dict):
            continue
        detections.append(
            {
                "object_id": detection.get("object_id", ""),
                "label": detection.get("label", ""),
                "description": detection.get("description", ""),
                "description_vi": detection.get("description_vi", ""),
                "attributes": detection.get("attributes", []),
                "action": detection.get("action", ""),
            }
        )
    return {
        "keyframe_n": frame.keyframe_n,
        "frame_id": frame.frame_id,
        "timestamp_ms": frame.timestamp_ms,
        "caption": raw.get("caption", ""),
        "detailed_caption": raw.get("detailed_caption", ""),
        "caption_vi": raw.get("caption_vi", ""),
        "detailed_caption_vi": raw.get("detailed_caption_vi", ""),
        "ocr_text": CLOCK_RE.sub("", str(raw.get("ocr_text", ""))),
        "news_ticker_text": raw.get("news_ticker_text", ""),
        "detections": detections,
        "spatial_relations": _resolved_spatial_relations(raw),
        "quality_flags": raw.get("quality_flags", []),
    }


def build_evidence_windows(
    frames: list[RuntimeFrame],
    scenes: list[MicroScene],
    segments: dict[int, ASRSegment],
    window_seconds: int = 75,
    overlap_seconds: int = 15,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    duration_ms = frames[-1].timestamp_ms
    windows: list[dict[str, Any]] = []
    start_ms = 0
    step_ms = (window_seconds - overlap_seconds) * 1000
    window_index = 0
    while start_ms <= duration_ms:
        end_ms = start_ms + window_seconds * 1000
        selected_scenes = [
            scene
            for scene in scenes
            if scene.end_ms >= start_ms and scene.start_ms <= end_ms
        ]
        frame_numbers = sorted({n for scene in selected_scenes for n in scene.frame_indices})
        frame_by_n = {frame.keyframe_n: frame for frame in frames}
        selected_frames = [frame_by_n[n] for n in frame_numbers]
        asr_indices = sorted(
            {
                index
                for frame in selected_frames
                for index in frame.nearby_asr_segment_indices
                if segments[index].end_ms >= start_ms and segments[index].start_ms <= end_ms
            }
        )
        if selected_scenes or asr_indices:
            windows.append(
                {
                    "window_id": f"window_{window_index + 1:04d}",
                    "start_ms": start_ms,
                    "end_ms": min(end_ms, duration_ms),
                    "scene_ids": [scene.scene_id for scene in selected_scenes],
                    "frames": [_compact_frame(frame) for frame in selected_frames],
                    "asr_segments": [
                        {
                            "index": index,
                            "start_ms": segments[index].start_ms,
                            "end_ms": segments[index].end_ms,
                            "text": segments[index].text,
                        }
                        for index in asr_indices
                    ],
                }
            )
        window_index += 1
        if end_ms >= duration_ms:
            break
        start_ms += step_ms
    return windows
