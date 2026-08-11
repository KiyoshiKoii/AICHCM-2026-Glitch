"""Evidence-preserving video → story → keyframe retrieval.

The module builds Elasticsearch-ready documents in memory from the three fixed
video-understanding artifacts and source metadata.  Its local lexical search is
used for deterministic tests and for diagnosing retrieval before an index is
rebuilt; it never writes another derived JSON file.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from semantic_pipeline.video_understanding.asr_alignment import align_asr
from semantic_pipeline.video_understanding.loaders import (
    build_runtime_frames,
    load_asr_segments,
    load_caption_records,
    load_keyframe_map,
)


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
FIRST_MOMENT_RE = re.compile(r"\b(đầu tiên|lần đầu|first)\b", re.IGNORECASE)


@dataclass(frozen=True)
class HierarchicalDocuments:
    video: dict[str, Any]
    segments: list[dict[str, Any]]
    frames: list[dict[str, Any]]


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _tokens(value: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(_fold(value)) if len(token) > 1}


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _segment_keyframes(segment: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(keyframe)
            for scene in segment.get("scenes", [])
            if isinstance(scene, dict)
            for keyframe in scene.get("keyframe_refs", [])
            if isinstance(keyframe, int) or (isinstance(keyframe, str) and keyframe.isdigit())
        }
    )


def _segment_representatives(segment: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(scene["representative_keyframe_n"])
            for scene in segment.get("scenes", [])
            if isinstance(scene, dict) and scene.get("representative_keyframe_n") is not None
        }
    )


def build_hierarchical_documents(
    *,
    pilot_dir: Path,
    caption_path: Path,
    asr_path: Path,
    map_path: Path,
    keyframe_dir: Path,
) -> HierarchicalDocuments:
    """Build video, story and frame documents without persisting copies."""

    summary = _read_object(pilot_dir / "video_summary.json")
    timeline = _read_object(pilot_dir / "timeline.json")
    video_id = str(summary.get("video_id", "")).strip()
    if not video_id or timeline.get("video_id") != video_id:
        raise ValueError("video summary and timeline must share a non-blank video_id")
    timeline_segments = timeline.get("segments", [])
    if not isinstance(timeline_segments, list):
        raise ValueError("timeline segments must be an array")

    captions = load_caption_records(caption_path, video_id)
    keyframe_map = load_keyframe_map(map_path)
    asr_segments, _ = load_asr_segments(asr_path, video_id)
    frames = build_runtime_frames(captions, keyframe_map, video_id, keyframe_dir)
    aligned_asr = align_asr(frames, asr_segments)
    asr_by_index = {item.index: item.text for item in asr_segments}

    video_document = {
        "document_type": "video",
        "video_id": video_id,
        "content_type": str(summary.get("content_type", "")),
        "duration_ms": int(summary.get("duration_ms", 0)),
        "summary_vi": str(summary.get("summary_vi", "")),
        "summary_en": str(summary.get("summary_en", "")),
        "main_topics": _strings(summary.get("main_topics")),
        "main_entities": _strings(summary.get("main_entities")),
        "main_locations": _strings(summary.get("main_locations")),
        "search_text": str(summary.get("search_text", "")),
        "segment_count": len(timeline_segments),
    }

    segment_documents: list[dict[str, Any]] = []
    segment_by_keyframe: dict[int, list[str]] = {}
    representatives_by_segment: dict[str, set[int]] = {}
    for raw in timeline_segments:
        if not isinstance(raw, dict):
            continue
        segment_id = str(raw.get("segment_id", "")).strip()
        if not segment_id:
            raise ValueError("timeline segment has no segment_id")
        keyframes = _segment_keyframes(raw)
        representatives = _segment_representatives(raw)
        references = [int(item) for item in raw.get("asr_segment_refs", []) if isinstance(item, int)]
        asr_text = " ".join(asr_by_index[index] for index in references if index in asr_by_index)
        topics = _strings(raw.get("topics"))
        entities = _strings(raw.get("entities"))
        locations = _strings(raw.get("locations"))
        title = str(raw.get("title", "")).strip()
        description = str(raw.get("summary", "")).strip()
        search_text = " ".join([title, description, *topics, *entities, *locations, asr_text]).strip()
        segment_documents.append(
            {
                "document_type": "segment",
                "segment_id": segment_id,
                "video_id": video_id,
                "title": title,
                "summary": description,
                "topics": topics,
                "entities": entities,
                "locations": locations,
                "start_ms": int(raw.get("start_ms", 0)),
                "end_ms": int(raw.get("end_ms", 0)),
                "keyframe_refs": keyframes,
                "representative_keyframe_refs": representatives,
                "asr_segment_refs": references,
                "asr_text": asr_text,
                "search_text": search_text,
            }
        )
        representatives_by_segment[segment_id] = set(representatives)
        for keyframe in keyframes:
            segment_by_keyframe.setdefault(keyframe, []).append(segment_id)

    frame_documents: list[dict[str, Any]] = []
    for frame in frames:
        raw = frame.raw_metadata
        detection_text: list[str] = []
        for detection in raw.get("detections", []):
            if isinstance(detection, dict):
                detection_text.extend(
                    str(detection.get(field, ""))
                    for field in ("label", "description", "description_vi", "action")
                )
                detection_text.extend(_strings(detection.get("attributes")))
        segment_ids = segment_by_keyframe.get(frame.keyframe_n, [])
        frame_asr = " ".join(aligned_asr[index].text for index in frame.asr_segment_indices if index in aligned_asr)
        visual_text = " ".join(
            str(raw.get(field, ""))
            for field in (
                "caption",
                "detailed_caption",
                "caption_vi",
                "detailed_caption_vi",
                "ocr_text",
                "news_ticker_text",
            )
        )
        frame_documents.append(
            {
                "document_type": "frame",
                "frame_id": frame.frame_id,
                "video_id": video_id,
                "keyframe_n": frame.keyframe_n,
                "native_frame_idx": frame.native_frame_idx,
                "timestamp_ms": frame.timestamp_ms,
                "segment_ids": segment_ids,
                "is_segment_representative": any(
                    frame.keyframe_n in representatives_by_segment.get(segment_id, set())
                    for segment_id in segment_ids
                ),
                "visual_text": " ".join([visual_text, *detection_text]).strip(),
                "asr_text": frame_asr,
                "ocr_text": str(raw.get("ocr_text", "")),
            }
        )
    return HierarchicalDocuments(video_document, segment_documents, frame_documents)


def iter_elasticsearch_actions(
    documents: HierarchicalDocuments,
    *,
    video_index: str,
    segment_index: str,
    frame_index: str,
) -> Iterable[dict[str, Any]]:
    """Yield idempotent bulk actions for separate video, story and frame indexes."""

    yield {"_op_type": "index", "_index": video_index, "_id": documents.video["video_id"], "_source": documents.video}
    for segment in documents.segments:
        yield {"_op_type": "index", "_index": segment_index, "_id": segment["segment_id"], "_source": segment}
    for frame in documents.frames:
        yield {"_op_type": "index", "_index": frame_index, "_id": frame["frame_id"], "_source": frame}


class HierarchicalVideoSearch:
    """Deterministic lexical reference implementation for the hierarchical flow."""

    def __init__(self, documents: HierarchicalDocuments) -> None:
        self.documents = documents
        self._segments = {item["segment_id"]: item for item in documents.segments}

    @staticmethod
    def _score(query: set[str], fields: Iterable[tuple[str, float]]) -> float:
        if not query:
            return 0.0
        score = 0.0
        for text, weight in fields:
            matched = len(query & _tokens(text))
            score += weight * matched / len(query)
        return score

    def search(self, query: str, *, top_segments: int = 3, top_frames: int = 10) -> dict[str, Any]:
        query_tokens = _tokens(query)
        if not query_tokens:
            raise ValueError("query must contain searchable text")
        video = self.documents.video
        video_score = self._score(
            query_tokens,
            ((video["search_text"], 1.0), (video["summary_vi"], 1.5), (video["summary_en"], 1.0)),
        )
        ranked_segments = sorted(
            (
                {
                    **segment,
                    "score": self._score(
                        query_tokens,
                        (
                            (segment["title"], 3.0),
                            (" ".join(segment["topics"]), 2.0),
                            (" ".join(segment["entities"] + segment["locations"]), 2.5),
                            (segment["summary"], 1.8),
                            (segment["asr_text"], 1.5),
                        ),
                    ),
                }
                for segment in self.documents.segments
            ),
            key=lambda item: (-item["score"], item["start_ms"], item["segment_id"]),
        )[:top_segments]
        selected_segment_ids = {item["segment_id"] for item in ranked_segments if item["score"] > 0}
        if not selected_segment_ids and ranked_segments:
            selected_segment_ids.add(ranked_segments[0]["segment_id"])

        ranked_frames = sorted(
            (
                {
                    **frame,
                    "score": self._score(
                        query_tokens,
                        (
                            (frame["visual_text"], 3.0),
                            (frame["asr_text"], 2.5),
                            (frame["ocr_text"], 1.5),
                        ),
                    )
                    + 0.1 * max(
                        (segment["score"] for segment in ranked_segments if segment["segment_id"] in frame["segment_ids"]),
                        default=0.0,
                    )
                    + (0.05 if frame["is_segment_representative"] else 0.0),
                }
                for frame in self.documents.frames
                if selected_segment_ids & set(frame["segment_ids"])
            ),
            key=lambda item: (-item["score"], item["timestamp_ms"], item["keyframe_n"]),
        )[:top_frames]
        if FIRST_MOMENT_RE.search(query) and ranked_frames:
            threshold = ranked_frames[0]["score"] * 0.70
            qualified = [item for item in ranked_frames if item["score"] >= threshold]
            ranked_frames = sorted(qualified, key=lambda item: (item["timestamp_ms"], item["keyframe_n"]))

        return {
            "query": query,
            "video": {"video_id": video["video_id"], "score": round(video_score, 6)},
            "segments": ranked_segments,
            "frames": ranked_frames,
        }
