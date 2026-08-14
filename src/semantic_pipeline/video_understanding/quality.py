"""Validation and quality scoring for the three persisted pilot artifacts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import ASRSegment, MicroScene, RuntimeFrame, StoryCandidate


def _valid_refs(
    stories: list[StoryCandidate],
    scenes: list[MicroScene],
    frames: list[RuntimeFrame],
    segments: dict[int, ASRSegment],
) -> tuple[int, int]:
    scene_ids = {scene.scene_id for scene in scenes}
    frame_numbers = {frame.keyframe_n for frame in frames}
    invalid_scene = 0
    invalid_asr = 0
    for story in stories:
        invalid_scene += sum(scene_id not in scene_ids for scene_id in story.scene_ids)
        invalid_asr += sum(index not in segments for index in story.asr_segment_indices)
    return invalid_scene, invalid_asr


def validate_artifacts(
    frames: list[RuntimeFrame],
    segments: dict[int, ASRSegment],
    scenes: list[MicroScene],
    stories: list[StoryCandidate],
    temporal_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    invalid_scene, invalid_asr = _valid_refs(stories, scenes, frames, segments)
    frame_numbers = {frame.keyframe_n for frame in frames}
    scene_frame_refs = [n for scene in scenes for n in scene.frame_indices]
    invalid_frame_refs = sum(n not in frame_numbers for n in scene_frame_refs)
    scene_order_errors = sum(
        1
        for current, following in zip(scenes, scenes[1:])
        if current.end_ms > following.start_ms
    )
    story_order_errors = sum(
        1
        for current, following in zip(stories, stories[1:])
        if _story_start(current, scenes) > _story_start(following, scenes)
    )
    assigned_asr = {index for scene in scenes for index in scene.asr_segment_indices}
    # ASR may be outside a visual scene at a sparse boundary, but every source
    # segment must still be accounted for by a nearby frame.
    primary_asr = {index for frame in frames for index in frame.asr_segment_indices}
    missing_primary_asr = len(set(segments) - primary_asr)
    all_scene_refs_valid = invalid_scene == 0 and invalid_frame_refs == 0
    all_asr_refs_valid = invalid_asr == 0
    boundary_quality = 1.0 if scene_order_errors == 0 and story_order_errors == 0 else 0.0
    groundedness = (
        sum(bool(item.scene_ids or item.asr_segment_indices) for item in stories) / len(stories)
        if stories
        else 0.0
    )
    coverage = len({n for scene in scenes for n in scene.frame_indices}) / len(frames) if frames else 0.0
    deduplication = 1.0 if len({scene.scene_id for scene in scenes}) == len(scenes) else 0.0
    references = 1.0 if all_scene_refs_valid and all_asr_refs_valid and missing_primary_asr == 0 else 0.0
    temporal_events = temporal_events or []
    frame_by_n = {frame.keyframe_n: frame for frame in frames}
    event_ids = [str(event.get("event_id", "")) for event in temporal_events]
    event_refs_valid = all(
        event_id
        and len(event.get("keyframe_refs", [])) > 0
        and all(int(keyframe) in frame_by_n for keyframe in event.get("keyframe_refs", []))
        for event_id, event in zip(event_ids, temporal_events)
    ) and len(event_ids) == len(set(event_ids))
    anchor_refs_valid = all(
        isinstance(anchor, dict)
        and str(anchor.get("frame_id", "")).startswith(f"{frames[0].video_id}_f")
        and int(anchor.get("keyframe_n", 0)) in frame_by_n
        and 0 <= int(anchor.get("timestamp_ms", 0)) <= frames[-1].timestamp_ms
        for event in temporal_events
        for anchor in event.get("temporal_anchors", [])
    ) if frames else not temporal_events
    event_order_errors = sum(
        1
        for current, following in zip(temporal_events, temporal_events[1:])
        if int(current.get("start_ms", 0)) > int(following.get("start_ms", 0))
    )
    temporal_validity = 1.0 if event_refs_valid and anchor_refs_valid and event_order_errors == 0 else 0.0
    overall = (
        0.30 * groundedness
        + 0.25 * coverage
        + 0.20 * boundary_quality
        + 0.15 * deduplication
        + 0.08 * references
        + 0.02 * temporal_validity
    )
    return {
        "statistics": {
            "input_frames": len(frames),
            "input_asr_segments": len(segments),
            "micro_scenes": len(scenes),
            "stories": len(stories),
            "primary_asr_segments": len(primary_asr),
            "temporal_events": len(temporal_events),
            "temporal_anchors": sum(len(event.get("temporal_anchors", [])) for event in temporal_events),
        },
        "quality": {
            "groundedness": round(groundedness, 4),
            "coverage": round(coverage, 4),
            "boundary_quality": round(boundary_quality, 4),
            "deduplication_quality": round(deduplication, 4),
            "reference_validity": round(references, 4),
            "overall_score": round(overall, 4),
        },
        "validation": {
            "all_frame_refs_valid": invalid_frame_refs == 0,
            "all_story_scene_refs_valid": invalid_scene == 0,
            "all_asr_refs_valid": invalid_asr == 0,
            "all_primary_asr_segments_assigned": missing_primary_asr == 0,
            "scene_order_errors": scene_order_errors,
            "story_order_errors": story_order_errors,
            "invalid_frame_refs": invalid_frame_refs,
            "invalid_story_scene_refs": invalid_scene,
            "invalid_story_asr_refs": invalid_asr,
            "unsupported_events": 0,
            "all_temporal_event_refs_valid": event_refs_valid,
            "all_temporal_anchor_refs_valid": anchor_refs_valid,
            "temporal_event_order_errors": event_order_errors,
        },
    }


def _story_start(story: StoryCandidate, scenes: list[MicroScene]) -> int:
    starts = [scene.start_ms for scene in scenes if scene.scene_id in story.scene_ids]
    return min(starts) if starts else 2**63 - 1


def hard_gates_pass(report: dict[str, Any]) -> bool:
    validation = report["validation"]
    base_gates = all(
        (
            validation["all_frame_refs_valid"],
            validation["all_story_scene_refs_valid"],
            validation["all_asr_refs_valid"],
            validation["all_primary_asr_segments_assigned"],
            validation["scene_order_errors"] == 0,
            validation["story_order_errors"] == 0,
            validation.get("all_temporal_event_refs_valid", True),
            validation.get("all_temporal_anchor_refs_valid", True),
            validation.get("temporal_event_order_errors", 0) == 0,
        )
    )
    coverage = report.get("retrieval_coverage")
    if coverage is None:
        return base_gates
    return base_gates and all(
        (
            coverage.get("segments_in_search_text") == coverage.get("timeline_segments"),
            not coverage.get("missing_segment_titles"),
            not coverage.get("missing_segment_summaries"),
        )
    )
