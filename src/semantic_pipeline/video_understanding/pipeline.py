"""Orchestration for the fixed L22_V001 video-understanding pilot."""

from __future__ import annotations

import hashlib
import json
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
from .publisher import publish_fixed_pilot
from .quality import hard_gates_pass, validate_artifacts
from .summarizer import NewsSummarizer
from .timeline_builder import build_evidence_windows, build_micro_scenes


DEFAULT_CAPTION_DIR = Path("data/metadata/caption")
DEFAULT_ASR_DIR = Path("data/metadata/metadata_asr")
DEFAULT_MAP_DIR = Path("data/map-keyframes")
DEFAULT_KEYFRAME_DIR = Path("data/keyframes")
DEFAULT_OUTPUT_ROOT = Path("data/processed/video_understanding")
PILOT_VIDEO_ID = "L22_V001"


def _story_timeline(
    candidates: list[Any],
    scenes: list[Any],
    frames: list[Any],
    segments: dict[int, Any],
) -> list[dict[str, Any]]:
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
                "segment_id": f"L22_V001_story_{index:04d}",
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
            if shared_scene or shared_asr:
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


def build_video(
    *,
    video_id: str = PILOT_VIDEO_ID,
    caption_dir: Path = DEFAULT_CAPTION_DIR,
    asr_dir: Path = DEFAULT_ASR_DIR,
    map_dir: Path = DEFAULT_MAP_DIR,
    keyframe_dir: Path = DEFAULT_KEYFRAME_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    use_llm: bool = False,
    require_llm: bool = False,
    force_publish: bool = False,
) -> dict[str, Any]:
    if video_id != PILOT_VIDEO_ID:
        raise ValueError(f"Pilot is intentionally restricted to {PILOT_VIDEO_ID}")
    caption_path = caption_dir / "L22" / f"{video_id}.json"
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
    candidates = _merge_duplicate_candidates(candidates)
    summary = summarizer.summarize_video(
        candidates,
        video_id=video_id,
        duration_ms=frames[-1].timestamp_ms,
    )
    timeline_segments = _story_timeline(candidates, scenes, frames, segments)
    generation_id = f"{video_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    validation = validate_artifacts(frames, segments, scenes, candidates)
    validation.update(
        {
            "generation_id": generation_id,
            "video_id": video_id,
            "profile": "news-v1",
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
        "schema_version": "video-timeline-v1",
        "generation_id": generation_id,
        "video_id": video_id,
        "content_type": "news",
        "duration_ms": frames[-1].timestamp_ms,
        "fps": frames[0].fps,
        "segments": timeline_segments,
    }
    video_summary = {
        "schema_version": "video-summary-v1",
        "generation_id": generation_id,
        "video_id": video_id,
        "content_type": "news",
        **summary,
        "segment_refs": [item["segment_id"] for item in timeline_segments],
        "search_text": " ".join(
            [
                summary.get("summary_vi", ""),
                summary.get("summary_en", ""),
                " ".join(summary.get("main_topics", [])),
                " ".join(summary.get("main_entities", [])),
                " ".join(item["title"] for item in timeline_segments),
            ]
        ).strip(),
    }
    pilot_dir = output_root / "L22" / video_id / "pilot"
    published = publish_fixed_pilot(
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
