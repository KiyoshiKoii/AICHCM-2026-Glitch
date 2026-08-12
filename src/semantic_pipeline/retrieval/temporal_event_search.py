"""Same-video, ordered event retrieval for BTC E1..En questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from semantic_pipeline.retrieval.hierarchical_video_search import build_hierarchical_documents
from semantic_pipeline.retrieval.temporal_query_parser import (
    ParsedTemporalQuery,
    TemporalEventQuery,
    parse_temporal_query,
    query_tokens,
)
from semantic_pipeline.video_understanding.temporal_events import build_temporal_events, event_search_text


TEMPORAL_STOPWORDS = {
    "first", "earliest", "start", "starts", "started", "begin", "begins", "beginning",
    "last", "latest", "final", "end", "ends", "ended", "complete", "completed",
    "completely", "fully", "moment", "momentos", "khoanh", "khac", "dau", "tien",
    "lan", "cuoi", "cung", "bat", "dau", "hoan", "toan", "sau", "do", "then",
    "visible", "shown", "show", "see", "seen", "thay",
    "is", "are", "the", "a", "an", "about", "story", "time", "people", "facing",
}


@dataclass(frozen=True)
class TemporalCorpus:
    video: dict[str, Any]
    events: list[dict[str, Any]]
    frames: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _event_documents(
    timeline: dict[str, Any],
    frames: list[dict[str, Any]],
    video_id: str,
) -> list[dict[str, Any]]:
    raw_events = timeline.get("events")
    if isinstance(raw_events, list) and raw_events:
        return [dict(item) for item in raw_events if isinstance(item, dict) and item.get("event_id")]
    # Compatibility for pilots generated before video-timeline-v2.
    generated = build_temporal_events(
        timeline.get("segments", []),
        [_RuntimeFrameProxy(item) for item in frames],
        content_profile=str(timeline.get("content_profile", "general_event")),
    )
    return generated


class _RuntimeFrameProxy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.frame_id = payload["frame_id"]
        self.keyframe_n = int(payload["keyframe_n"])
        self.timestamp_ms = int(payload["timestamp_ms"])
        self.native_frame_idx = int(payload["native_frame_idx"])
        self.raw_metadata = {
            "caption": payload.get("visual_text", ""),
            "asr_text": payload.get("asr_text", ""),
            "ocr_text": payload.get("ocr_text", ""),
        }


def discover_temporal_corpus(
    *,
    output_root: Path,
    caption_dir: Path,
    asr_dir: Path,
    map_dir: Path,
    keyframe_dir: Path,
    batch_ids: Iterable[str] = (),
    video_ids: Iterable[str] = (),
) -> list[TemporalCorpus]:
    wanted_batches = {item.upper() for item in batch_ids}
    wanted_videos = {item.upper() for item in video_ids}
    corpora: list[TemporalCorpus] = []
    for pilot_dir in sorted(output_root.glob("L*/L*_V*/pilot")):
        video_id = pilot_dir.parent.name.upper()
        batch_id = pilot_dir.parent.parent.name.upper()
        if wanted_batches and batch_id not in wanted_batches:
            continue
        if wanted_videos and video_id not in wanted_videos and video_id.rsplit("_", 1)[-1] not in wanted_videos:
            continue
        try:
            documents = build_hierarchical_documents(
                pilot_dir=pilot_dir,
                caption_path=caption_dir / batch_id / f"{video_id}.json",
                asr_path=asr_dir / f"{video_id}.json",
                map_path=map_dir / f"{video_id}.csv",
                keyframe_dir=keyframe_dir / video_id,
            )
            timeline = _read_json(pilot_dir / "timeline.json")
            events = _event_documents(timeline, documents.frames, video_id)
            frame_by_n = {int(frame["keyframe_n"]): frame for frame in documents.frames}
            for event in events:
                event["video_id"] = video_id
                event.setdefault("keyframe_refs", [])
                event["search_text"] = event_search_text(event)
                event["_frames"] = [
                    frame_by_n[n]
                    for n in event.get("keyframe_refs", [])
                    if isinstance(n, int) and n in frame_by_n
                ]
            corpora.append(TemporalCorpus(documents.video, events, documents.frames))
        except (OSError, ValueError, json.JSONDecodeError):
            # A partially generated batch should not make valid videos
            # unsearchable.  Validation remains visible in each pilot report.
            continue
    if not corpora:
        raise FileNotFoundError(f"No valid temporal pilots found under {output_root}")
    return corpora


def _score(query: set[str], text: str) -> float:
    useful = query - TEMPORAL_STOPWORDS
    if not useful:
        useful = query
    matched = useful & query_tokens(text)
    return len(matched) / len(useful) if useful else 0.0


def _anchor_for(event: dict[str, Any], query: TemporalEventQuery) -> dict[str, Any] | None:
    anchors = [item for item in event.get("temporal_anchors", []) if isinstance(item, dict)]
    preferred = {
        "action_start": {"action_start", "first_contact"},
        "first_contact": {"first_contact", "action_start"},
        "state_complete": {"state_complete", "first_complete", "representative"},
        "last_complete": {"last_complete", "action_end"},
        "action_end": {"action_end", "last_complete"},
        "representative": {"representative"},
        "first_visible": {"first_visible", "action_start", "representative"},
    }.get(query.required_anchor, {"representative"})
    if query.required_anchor == "first_visible":
        frames = sorted(
            event.get("_frames", []),
            key=lambda item: (item["timestamp_ms"], item["keyframe_n"]),
        )
        best_score = max(
            (
                _score(
                    set(query.tokens),
                    " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
                )
                for frame in frames
            ),
            default=0.0,
        )
        if best_score > 0:
            threshold = best_score * 0.7
            for frame in frames:
                score = _score(
                    set(query.tokens),
                    " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
                )
                if score >= threshold:
                    return {
                        "anchor_type": "first_visible",
                        "frame_id": frame["frame_id"],
                        "keyframe_n": frame["keyframe_n"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "native_frame_idx": frame["native_frame_idx"],
                        "confidence": round(max(0.55, score), 6),
                    }
    for anchor in anchors:
        if anchor.get("anchor_type") in preferred:
            return anchor
    frames = sorted(event.get("_frames", []), key=lambda item: (item["timestamp_ms"], item["keyframe_n"]))
    if not frames:
        return None
    selected = frames[-1] if query.required_anchor in {"last_complete", "action_end"} else frames[0]
    return {
        "anchor_type": query.required_anchor,
        "frame_id": selected["frame_id"],
        "keyframe_n": selected["keyframe_n"],
        "timestamp_ms": selected["timestamp_ms"],
        "native_frame_idx": selected["native_frame_idx"],
        "confidence": 0.55,
    }


class TemporalEventSearch:
    def __init__(self, corpora: Iterable[TemporalCorpus]) -> None:
        self.corpora = tuple(corpora)

    def _rank_event(self, event: dict[str, Any], query: TemporalEventQuery) -> float:
        query_set = set(query.tokens)
        title_score = _score(query_set, str(event.get("description_vi", "")))
        event_score = _score(query_set, event.get("search_text", ""))
        frame_scores = [
            _score(
                query_set,
                " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
            )
            for frame in event.get("_frames", [])
        ]
        frame_score = max(frame_scores, default=0.0)
        return 0.55 * title_score + 0.25 * event_score + 0.20 * frame_score

    def search(
        self,
        query: str | ParsedTemporalQuery,
        *,
        top_k_videos: int = 10,
    ) -> dict[str, Any]:
        parsed = parse_temporal_query(query) if isinstance(query, str) else query
        context_tokens = query_tokens(parsed.shared_context)
        ranked_videos: list[tuple[float, TemporalCorpus, list[dict[str, Any]]]] = []
        for corpus in self.corpora:
            video = corpus.video
            video_text = " ".join(
                [
                    str(video.get("summary_vi", "")),
                    str(video.get("summary_en", "")),
                    str(video.get("search_text", "")),
                    *[event.get("search_text", "") for event in corpus.events],
                ]
            )
            context_score = _score(context_tokens, video_text) if context_tokens else 0.0
            selected_events: list[dict[str, Any]] = []
            event_scores: list[float] = []
            used_event_ids: set[str] = set()
            for event_query in parsed.events:
                ranked = sorted(
                    (
                        (
                            self._rank_event(event, event_query)
                            - (0.15 if event.get("event_id") in used_event_ids else 0.0),
                            event,
                        )
                        for event in corpus.events
                    ),
                    key=lambda item: (-item[0], int(item[1].get("start_ms", 0)), str(item[1].get("event_id", ""))),
                )
                score, event = ranked[0] if ranked else (0.0, None)
                if ranked and event_query.required_anchor in {"action_start", "first_contact", "first_visible"}:
                    close_matches = [
                        item for item in ranked
                        if item[0] >= max(0.30, score * 0.80)
                    ]
                    if close_matches:
                        score, event = min(
                            close_matches,
                            key=lambda item: (
                                int(item[1].get("start_ms", 0)),
                                -item[0],
                                str(item[1].get("event_id", "")),
                            ),
                        )
                if event is None or score < 0.30:
                    event_scores.append(0.0)
                    continue
                anchor = _anchor_for(event, event_query)
                if anchor is None:
                    event_scores.append(0.0)
                    continue
                selected_events.append(
                    {
                        "event_index": event_query.event_index,
                        "source_label": event_query.source_label,
                        "event_id": event["event_id"],
                        "event_text": event.get("description_vi", ""),
                        "anchor_type": anchor["anchor_type"],
                        "frame_id": anchor["frame_id"],
                        "keyframe_n": anchor["keyframe_n"],
                        "native_frame_idx": anchor["native_frame_idx"],
                        "timestamp_ms": anchor["timestamp_ms"],
                        "score": round(score, 6),
                        "confidence": anchor.get("confidence", event.get("confidence", 0.0)),
                        "uncertain": bool(event.get("uncertain", True)),
                        "reason_vi": event.get("description_vi", ""),
                    }
                )
                used_event_ids.add(str(event.get("event_id", "")))
                event_scores.append(score)
            coverage = sum(score > 0 for score in event_scores) / max(1, len(parsed.events))
            ordering_bonus = 0.0
            timestamps = [item["timestamp_ms"] for item in selected_events]
            if timestamps and timestamps == sorted(timestamps):
                ordering_bonus = 0.1
            total = 0.3 * context_score + 0.5 * coverage + 0.2 * (sum(event_scores) / max(1, len(event_scores))) + ordering_bonus
            total = min(1.0, max(0.0, total))
            ranked_videos.append((total, corpus, selected_events))
        ranked_videos.sort(key=lambda item: (-item[0], str(item[1].video.get("video_id", ""))))
        selected = ranked_videos[: max(1, top_k_videos)]
        if not selected:
            return {"query": parsed.shared_context, "selected_video": None, "videos": [], "events": []}
        best_score, best_corpus, best_events = selected[0]
        best_events.sort(key=lambda item: item["event_index"])
        return {
            "query": query if isinstance(query, str) else parsed.shared_context,
            "mode": "temporal_event_search",
            "selected_video": {
                "video_id": best_corpus.video["video_id"],
                "score": round(best_score, 6),
                "matched_events": len(best_events),
                "total_events": len(parsed.events),
                "event_coverage": round(len(best_events) / max(1, len(parsed.events)), 6),
            },
            "events": best_events,
            "videos": [
                {
                    "video_id": corpus.video["video_id"],
                    "score": round(score, 6),
                    "matched_events": len(events),
                    "total_events": len(parsed.events),
                }
                for score, corpus, events in selected
            ],
        }
