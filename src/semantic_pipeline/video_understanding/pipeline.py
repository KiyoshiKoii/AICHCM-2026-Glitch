"""Orchestration for evidence-grounded per-video understanding artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asr_alignment import align_asr, quality_flags_for_segments
from .loaders import (
    build_runtime_frames,
    load_asr_segments,
    load_caption_records,
    load_keyframe_map,
    sha256_file,
)
from .publisher import publish_pilot
from .quality import hard_gates_pass, validate_artifacts
from .summarizer import NewsSummarizer
from .timeline_builder import build_evidence_windows, build_micro_scenes
from .temporal_events import build_temporal_events, classify_content_profile, event_search_text


DEFAULT_CAPTION_DIR = Path("data/metadata/caption")
DEFAULT_ASR_DIR = Path("data/metadata/metadata_asr")
DEFAULT_MAP_DIR = Path("data/map-keyframes")
DEFAULT_KEYFRAME_DIR = Path("data/keyframes")
DEFAULT_OUTPUT_ROOT = Path("data/processed/video_understanding")
VIDEO_ID_RE = re.compile(r"^(?P<batch>L\d{2})_V\d{3}$")
EVIDENCE_STOPWORDS = {
    "anh",
    "cac",
    "cho",
    "cua",
    "dang",
    "duoc",
    "kien",
    "mot",
    "nhung",
    "nguoi",
    "su",
    "tai",
    "thanh",
    "thoi",
    "tin",
    "trong",
    "va",
    "viec",
}


def batch_id_from_video_id(video_id: str) -> str:
    """Validate a BTC video ID and return its batch component."""

    match = VIDEO_ID_RE.fullmatch(video_id)
    if match is None:
        raise ValueError("video_id must use the BTC format Lxx_Vyyy, for example L22_V001")
    return match.group("batch")


def _story_timeline(
    candidates: list[Any],
    scenes: list[Any],
    frames: list[Any],
    segments: dict[int, Any],
) -> list[dict[str, Any]]:
    if not frames:
        return []
    video_id = frames[0].video_id
    frame_by_n = {frame.keyframe_n: frame for frame in frames}
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        selected = [scene_by_id[scene_id] for scene_id in candidate.scene_ids if scene_id in scene_by_id]
        if not selected:
            continue
        selected.sort(key=lambda scene: scene.start_ms)
        scene_payload: list[dict[str, Any]] = []
        for scene in selected:
            frame_refs = [frame_by_n[n] for n in scene.frame_indices if n in frame_by_n]
            representative = frame_by_n.get(scene.representative_keyframe_n)
            scene_payload.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_type": scene.scene_type,
                    "start_ms": scene.start_ms,
                    "end_ms": scene.end_ms,
                    "start_native_frame_idx": frame_refs[0].native_frame_idx,
                    "end_native_frame_idx": frame_refs[-1].native_frame_idx,
                    "keyframe_refs": [frame.keyframe_n for frame in frame_refs],
                    "asr_segment_refs": scene.asr_segment_indices,
                    "representative_keyframe_n": scene.representative_keyframe_n,
                    "representative_native_frame_idx": representative.native_frame_idx if representative else None,
                }
            )
        output.append(
            {
                "segment_id": f"{video_id}_story_{index:04d}",
                "segment_type": "news_story",
                "title": candidate.title,
                "summary": candidate.summary,
                "topics": candidate.topics,
                "entities": candidate.entities,
                "locations": candidate.locations,
                "uncertain": candidate.uncertain,
                "start_ms": selected[0].start_ms,
                "end_ms": selected[-1].end_ms,
                "start_native_frame_idx": frame_by_n[selected[0].frame_indices[0]].native_frame_idx,
                "end_native_frame_idx": frame_by_n[selected[-1].frame_indices[-1]].native_frame_idx,
                "scenes": scene_payload,
                "asr_segment_refs": sorted(
                    {item for scene in selected for item in scene.asr_segment_indices}
                ),
            }
        )
    return output


def _merge_duplicate_candidates(candidates: list[Any]) -> list[Any]:
    """Collapse duplicate LLM cards produced by overlapping evidence windows."""

    result: list[Any] = []
    for candidate in candidates:
        if not candidate.scene_ids and not candidate.asr_segment_indices:
            continue
        current_refs = set(candidate.scene_ids)
        current_asr = set(candidate.asr_segment_indices)
        duplicate = None
        for existing in result:
            shared_scene = len(current_refs & set(existing.scene_ids))
            shared_asr = len(current_asr & set(existing.asr_segment_indices))
            if _same_news_story(existing, candidate) or _shared_evidence_duplicate(
                existing,
                candidate,
                shared_scene=shared_scene,
                shared_asr=shared_asr,
            ):
                duplicate = existing
                break
        if duplicate is None:
            result.append(candidate)
            continue
        duplicate.scene_ids = sorted(set(duplicate.scene_ids) | current_refs)
        duplicate.asr_segment_indices = sorted(set(duplicate.asr_segment_indices) | current_asr)
        if len(candidate.summary) > len(duplicate.summary):
            duplicate.summary = candidate.summary
        duplicate.topics = list(dict.fromkeys(duplicate.topics + candidate.topics))[:20]
        duplicate.entities = list(dict.fromkeys(duplicate.entities + candidate.entities))[:20]
        duplicate.locations = list(dict.fromkeys(duplicate.locations + candidate.locations))[:20]
    return sorted(result, key=lambda item: min(item.scene_ids) if item.scene_ids else "")


def _story_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W_]+", value or "", flags=re.UNICODE)
        if len(token) > 2
    }


def _same_news_story(left: Any, right: Any) -> bool:
    """Detect duplicate cards emitted by adjacent/overlapping LLM windows."""

    left_text = _story_tokens(f"{left.title} {left.summary}")
    right_text = _story_tokens(f"{right.title} {right.summary}")
    union = left_text | right_text
    similarity = len(left_text & right_text) / len(union) if union else 0.0
    left_title = _story_tokens(left.title)
    right_title = _story_tokens(right.title)
    title_union = left_title | right_title
    title_similarity = len(left_title & right_title) / len(title_union) if title_union else 0.0
    anchors = set(left.entities + left.locations) & set(right.entities + right.locations)
    return bool(anchors) and (similarity >= 0.30 or title_similarity >= 0.45)


def _prune_candidate_evidence(
    candidates: list[Any],
    scenes: list[Any],
    frames: list[Any],
    segments: dict[int, Any],
) -> list[Any]:
    """Discard LLM scene references with no lexical support for their event."""

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    frame_by_n = {frame.keyframe_n: frame for frame in frames}
    result: list[Any] = []
    for candidate in candidates:
        anchors = _evidence_tokens(
            " ".join(
                [candidate.title, *candidate.topics, *candidate.entities, *candidate.locations]
            )
        )
        if not anchors:
            result.append(candidate)
            continue
        retained: list[str] = []
        for scene_id in candidate.scene_ids:
            scene = scene_by_id.get(scene_id)
            if scene is None:
                continue
            evidence = " ".join(
                [
                    *[
                        _frame_evidence_text(frame_by_n[keyframe])
                        for keyframe in scene.frame_indices
                        if keyframe in frame_by_n
                    ],
                    *[
                        str(segments[index].text)
                        for index in scene.asr_segment_indices
                        if index in segments
                    ],
                ]
            )
            if anchors & _evidence_tokens(evidence):
                retained.append(scene_id)
        if retained:
            candidate.scene_ids = retained
            candidate.asr_segment_indices = sorted(
                {
                    index
                    for scene_id in retained
                    for index in scene_by_id[scene_id].asr_segment_indices
                }
            )
        result.append(candidate)
    return result


def _interval_gap_ms(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    """Return zero for overlapping intervals, otherwise their distance."""

    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0


def _map_asr_refs_to_scenes(
    asr_indices: list[int],
    scenes: list[Any],
    segments: dict[int, Any],
    *,
    max_gap_ms: int,
) -> list[str]:
    """Map ASR-only evidence to the smallest set of nearby visual scenes.

    The normal alignment path puts an ASR index directly on the scene that
    contains its midpoint.  The interval fallback covers sparse keyframes or
    a Gemini response whose ASR reference came from a neighbouring window.
    """

    scene_refs: set[str] = set()
    ordered_scenes = sorted(scenes, key=lambda scene: (scene.start_ms, scene.end_ms))
    for index in asr_indices:
        segment = segments.get(index)
        if segment is None:
            continue
        direct = [scene for scene in ordered_scenes if index in scene.asr_segment_indices]
        if direct:
            scene_refs.update(scene.scene_id for scene in direct)
            continue

        overlapping = [
            scene
            for scene in ordered_scenes
            if _interval_gap_ms(
                segment.start_ms,
                segment.end_ms,
                scene.start_ms,
                scene.end_ms,
            )
            == 0
        ]
        if overlapping:
            scene_refs.update(scene.scene_id for scene in overlapping)
            continue

        nearest = min(
            ordered_scenes,
            key=lambda scene: (
                _interval_gap_ms(
                    segment.start_ms,
                    segment.end_ms,
                    scene.start_ms,
                    scene.end_ms,
                ),
                scene.start_ms,
            ),
            default=None,
        )
        if nearest is not None and _interval_gap_ms(
            segment.start_ms,
            segment.end_ms,
            nearest.start_ms,
            nearest.end_ms,
        ) <= max_gap_ms:
            scene_refs.add(nearest.scene_id)

    return [
        scene.scene_id
        for scene in ordered_scenes
        if scene.scene_id in scene_refs
    ]


def _normalize_story_evidence(
    candidates: list[Any],
    scenes: list[Any],
    segments: dict[int, Any],
    *,
    max_asr_scene_gap_ms: int = 10_000,
) -> tuple[list[Any], dict[str, int]]:
    """Ensure every retained story has valid visual and temporal evidence."""

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    stats = {
        "stories_received": len(candidates),
        "asr_only_stories_received": 0,
        "asr_only_stories_mapped": 0,
        "asr_only_stories_dropped": 0,
        "stories_dropped_without_evidence": 0,
    }
    normalized: list[Any] = []
    for candidate in candidates:
        valid_scene_ids = list(dict.fromkeys(
            scene_id for scene_id in candidate.scene_ids if scene_id in scene_by_id
        ))
        valid_asr_indices = list(dict.fromkeys(
            index for index in candidate.asr_segment_indices if index in segments
        ))
        had_no_valid_scenes = not valid_scene_ids
        if had_no_valid_scenes and valid_asr_indices:
            stats["asr_only_stories_received"] += 1
            valid_scene_ids = _map_asr_refs_to_scenes(
                valid_asr_indices,
                scenes,
                segments,
                max_gap_ms=max_asr_scene_gap_ms,
            )
            if valid_scene_ids:
                stats["asr_only_stories_mapped"] += 1
            else:
                stats["asr_only_stories_dropped"] += 1

        if not valid_scene_ids:
            stats["stories_dropped_without_evidence"] += 1
            continue

        candidate.scene_ids = valid_scene_ids
        candidate.asr_segment_indices = valid_asr_indices
        normalized.append(candidate)

    return normalized, stats


def _story_start_ms(
    candidate: Any,
    scenes: list[Any],
    segments: dict[int, Any],
) -> int:
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    scene_starts = [
        scene_by_id[scene_id].start_ms
        for scene_id in candidate.scene_ids
        if scene_id in scene_by_id
    ]
    if scene_starts:
        return min(scene_starts)
    asr_starts = [segments[index].start_ms for index in candidate.asr_segment_indices if index in segments]
    return min(asr_starts, default=2**63 - 1)


def _sort_story_candidates(
    candidates: list[Any],
    scenes: list[Any],
    segments: dict[int, Any],
) -> list[Any]:
    return sorted(
        candidates,
        key=lambda candidate: (
            _story_start_ms(candidate, scenes, segments),
            candidate.source_window_id or "",
            candidate.title.casefold(),
        ),
    )


def _frame_evidence_text(frame: Any) -> str:
    raw = frame.raw_metadata
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
    return " ".join(values)


def _evidence_tokens(value: str) -> set[str]:
    return {token for token in _story_tokens(value) if token not in EVIDENCE_STOPWORDS}


def _shared_evidence_duplicate(
    left: Any,
    right: Any,
    *,
    shared_scene: int,
    shared_asr: int,
) -> bool:
    """Avoid merging unrelated cards that happen to share an anchor scene."""

    if not (shared_scene or shared_asr):
        return False
    left_title = _story_tokens(left.title)
    right_title = _story_tokens(right.title)
    union = left_title | right_title
    title_similarity = len(left_title & right_title) / len(union) if union else 0.0
    shared_anchors = set(left.entities + left.locations) & set(right.entities + right.locations)
    return title_similarity >= 0.30 or bool(shared_anchors)


def build_video(
    *,
    video_id: str,
    caption_dir: Path = DEFAULT_CAPTION_DIR,
    asr_dir: Path = DEFAULT_ASR_DIR,
    map_dir: Path = DEFAULT_MAP_DIR,
    keyframe_dir: Path = DEFAULT_KEYFRAME_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    use_llm: bool = False,
    require_llm: bool = False,
    force_publish: bool = False,
) -> dict[str, Any]:
    batch_id = batch_id_from_video_id(video_id)
    caption_path = caption_dir / batch_id / f"{video_id}.json"
    asr_path = asr_dir / f"{video_id}.json"
    map_path = map_dir / f"{video_id}.csv"
    image_dir = keyframe_dir / video_id
    source_paths = (caption_path, asr_path, map_path)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    captions = load_caption_records(caption_path, video_id)
    keyframe_map = load_keyframe_map(map_path)
    asr_segments, asr_available = load_asr_segments(asr_path, video_id)
    frames = build_runtime_frames(captions, keyframe_map, video_id, image_dir)
    segments = align_asr(frames, asr_segments)
    scenes = build_micro_scenes(frames, segments)
    windows = build_evidence_windows(frames, scenes, segments)
    summarizer = NewsSummarizer(use_llm=use_llm, require_llm=require_llm)
    candidates = summarizer.summarize_windows(
        windows,
        scenes,
        frames,
        segments,
        require_llm=require_llm,
    )
    candidates = _prune_candidate_evidence(candidates, scenes, frames, segments)
    candidates, evidence_normalization = _normalize_story_evidence(
        candidates,
        scenes,
        segments,
    )
    candidates = _merge_duplicate_candidates(candidates)
    candidates = _sort_story_candidates(candidates, scenes, segments)
    summary = summarizer.summarize_video(
        candidates,
        video_id=video_id,
        duration_ms=frames[-1].timestamp_ms,
    )
    timeline_segments = _story_timeline(candidates, scenes, frames, segments)
    content_profile, profile_confidence = classify_content_profile(frames, asr_segments)
    temporal_events = build_temporal_events(
        timeline_segments,
        frames,
        content_profile=content_profile,
    )
    search_text = _build_search_text(summary, timeline_segments)
    search_text = " ".join(
        dict.fromkeys(
            [
                search_text,
                content_profile,
                *[event_search_text(event) for event in temporal_events],
            ]
        )
    )
    generation_id = f"{video_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    validation = validate_artifacts(frames, segments, scenes, candidates, temporal_events)
    validation["evidence_normalization"] = evidence_normalization
    validation["retrieval_coverage"] = _retrieval_coverage(timeline_segments, search_text)
    validation.update(
        {
            "generation_id": generation_id,
            "video_id": video_id,
            "profile": f"{content_profile}-v1",
            "summarization_mode": summarizer.mode,
            "model": summarizer.model,
            "prompt_version": summarizer.prompt_version,
            "input_hashes": source_hashes,
            "asr_available": asr_available,
            "asr_quality_flags": {
                "segments_with_unk": sum(
                    bool(quality_flags_for_segments([segment.index], segments))
                    for segment in asr_segments
                )
            },
        }
    )
    validation["inputs_unchanged"] = all(sha256_file(path) == source_hashes[str(path)] for path in source_paths)
    if not hard_gates_pass(validation):
        validation["status"] = "rejected"
        raise ValueError(json.dumps(validation, ensure_ascii=False, indent=2))

    timeline = {
        "schema_version": "video-timeline-v2",
        "generation_id": generation_id,
        "video_id": video_id,
        "content_type": content_profile,
        "content_profile": content_profile,
        "profile_confidence": profile_confidence,
        "duration_ms": frames[-1].timestamp_ms,
        "fps": frames[0].fps,
        "segments": timeline_segments,
        "events": temporal_events,
    }
    video_summary = {
        "schema_version": "video-summary-v2",
        "generation_id": generation_id,
        "video_id": video_id,
        "content_type": content_profile,
        "content_profile": content_profile,
        "profile_confidence": profile_confidence,
        **summary,
        "segment_refs": [item["segment_id"] for item in timeline_segments],
        "event_refs": [item["event_id"] for item in temporal_events],
        "search_text": search_text,
    }
    pilot_dir = output_root / batch_id / video_id / "pilot"
    published = publish_pilot(
        timeline=timeline,
        video_summary=video_summary,
        validation_report=validation,
        pilot_dir=pilot_dir,
        force=force_publish,
    )
    validation["published"] = published
    return {
        "video_id": video_id,
        "published": published,
        "pilot_dir": str(pilot_dir),
        "summarization_mode": summarizer.mode,
        "quality_score": validation["quality"]["overall_score"],
        "stories": len(timeline_segments),
        "scenes": len(scenes),
        "frames": len(frames),
        "asr_segments": len(asr_segments),
    }


def _build_search_text(summary: dict[str, Any], timeline_segments: list[dict[str, Any]]) -> str:
    """Build bilingual retrieval text without exact duplicate fragments."""

    fragments = [
        str(summary.get("summary_vi", "")).strip(),
        *[
            fragment
            for item in timeline_segments
            for fragment in (
                str(item.get("title", "")).strip(),
                str(item.get("summary", "")).strip(),
                *[str(value).strip() for value in item.get("topics", [])],
                *[str(value).strip() for value in item.get("entities", [])],
                *[str(value).strip() for value in item.get("locations", [])],
            )
        ],
        *[str(item).strip() for item in summary.get("main_topics", [])],
        *[str(item).strip() for item in summary.get("main_entities", [])],
        *[str(item).strip() for item in summary.get("main_locations", [])],
        str(summary.get("summary_en", "")).strip(),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for fragment in fragments:
        normalized = " ".join(fragment.casefold().split())
        if fragment and normalized not in seen:
            seen.add(normalized)
            unique.append(fragment)
    return " ".join(unique)


def _retrieval_coverage(timeline_segments: list[dict[str, Any]], search_text: str) -> dict[str, Any]:
    """Verify each persisted news segment remains discoverable at video level."""

    searchable = " ".join(search_text.casefold().split())
    missing_titles = [
        str(item.get("segment_id", ""))
        for item in timeline_segments
        if str(item.get("title", "")).strip()
        and " ".join(str(item["title"]).casefold().split()) not in searchable
    ]
    missing_summaries = [
        str(item.get("segment_id", ""))
        for item in timeline_segments
        if str(item.get("summary", "")).strip()
        and " ".join(str(item["summary"]).casefold().split()) not in searchable
    ]
    return {
        "timeline_segments": len(timeline_segments),
        "segments_in_search_text": len(timeline_segments) - len(set(missing_titles) | set(missing_summaries)),
        "missing_segment_titles": missing_titles,
        "missing_segment_summaries": missing_summaries,
    }
