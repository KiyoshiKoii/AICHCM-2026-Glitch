"""Elasticsearch-first video selection for temporal event queries.

The selector deliberately stops at ``video_id``.  It searches compact video
and timeline-segment documents across a batch, then lets the temporal event
pipeline load frames for only the selected video.  This keeps E1..En matching
out of the batch-wide hot path.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from semantic_pipeline.retrieval.elasticsearch_backend import require_elasticsearch
from semantic_pipeline.retrieval.hierarchical_index_definition import (
    SEGMENT_INDEX_NAME,
    VIDEO_INDEX_NAME,
    segment_index_definition,
    video_index_definition,
)
from semantic_pipeline.retrieval.temporal_query_parser import ParsedTemporalQuery

try:
    from elasticsearch.helpers import streaming_bulk
except ImportError:  # pragma: no cover - covered through require_elasticsearch.
    streaming_bulk = None


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _segment_search_text(segment: dict[str, Any]) -> str:
    return " ".join(
        [
            str(segment.get("title", "")),
            str(segment.get("summary", "")),
            *_strings(segment.get("topics")),
            *_strings(segment.get("entities")),
            *_strings(segment.get("locations")),
            *_strings(segment.get("actions")),
            *_strings(segment.get("objects")),
            *_strings(segment.get("visual_states")),
            str(segment.get("asr_text", "")),
        ]
    ).strip()


def build_selector_documents(pilot_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build only video/segment documents; captions and frame maps are untouched."""

    summary = _read_object(pilot_dir / "video_summary.json")
    video_id = str(summary.get("video_id", "")).strip().upper()
    if not video_id:
        raise ValueError(f"{pilot_dir}: video summary has no video_id")

    timeline_path = pilot_dir / "timeline.json"
    timeline = _read_object(timeline_path) if timeline_path.is_file() else {}
    timeline_video_id = str(timeline.get("video_id", "")).strip().upper()
    if timeline_video_id and timeline_video_id != video_id:
        raise ValueError(f"{pilot_dir}: summary/timeline video_id mismatch")
    raw_segments = timeline.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError(f"{timeline_path}: segments must be an array")

    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        segment_id = str(raw.get("segment_id", "")).strip()
        if not segment_id:
            continue
        segments.append(
            {
                "document_type": "segment",
                "segment_id": segment_id,
                "video_id": video_id,
                "title": str(raw.get("title", "")),
                "summary": str(raw.get("summary", "")),
                "topics": _strings(raw.get("topics")),
                "entities": _strings(raw.get("entities")),
                "locations": _strings(raw.get("locations")),
                "start_ms": int(raw.get("start_ms", 0)),
                "end_ms": int(raw.get("end_ms", 0)),
                "keyframe_refs": [
                    int(item)
                    for item in raw.get("keyframe_refs", [])
                    if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
                ],
                "representative_keyframe_refs": [
                    int(item)
                    for item in raw.get("representative_keyframe_refs", [])
                    if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
                ],
                "asr_segment_refs": [
                    int(item)
                    for item in raw.get("asr_segment_refs", [])
                    if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
                ],
                "asr_text": str(raw.get("asr_text", "")),
                "search_text": _segment_search_text(raw),
            }
        )

    raw_events = timeline.get("events", [])
    event_count = len(raw_events) if isinstance(raw_events, list) else 0
    video = {
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
        "segment_count": len(segments),
        "event_count": event_count,
        "content_profile": str(summary.get("content_profile", summary.get("content_type", ""))),
        "profile_confidence": float(summary.get("profile_confidence", 0.0)),
    }
    return video, segments


def iter_selector_actions(
    output_root: Path,
    *,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
    video_index: str = VIDEO_INDEX_NAME,
    segment_index: str = SEGMENT_INDEX_NAME,
) -> Iterable[dict[str, Any]]:
    wanted_batches = {item.upper() for item in batch_ids}
    wanted_videos = {item.upper() for item in video_ids}
    for pilot_dir in sorted(output_root.glob("L*/L*_V*/pilot")):
        batch_id = pilot_dir.parent.parent.name.upper()
        video_id = pilot_dir.parent.name.upper()
        if wanted_batches and batch_id not in wanted_batches:
            continue
        if wanted_videos and video_id not in wanted_videos:
            continue
        if not (pilot_dir / "video_summary.json").is_file():
            continue
        video, segments = build_selector_documents(pilot_dir)
        yield {
            "_op_type": "index",
            "_index": video_index,
            "_id": video["video_id"],
            "_source": video,
        }
        for segment in segments:
            yield {
                "_op_type": "index",
                "_index": segment_index,
                "_id": segment["segment_id"],
                "_source": segment,
            }


def ensure_selector_indices(
    client: Any,
    *,
    video_index: str = VIDEO_INDEX_NAME,
    segment_index: str = SEGMENT_INDEX_NAME,
) -> dict[str, bool]:
    created: dict[str, bool] = {}
    for name, definition in (
        (video_index, video_index_definition()),
        (segment_index, segment_index_definition()),
    ):
        exists = bool(client.indices.exists(index=name))
        if not exists:
            client.indices.create(index=name, **definition)
        created[name] = not exists
    return created


def bulk_ingest_selector(
    client: Any,
    *,
    output_root: Path,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
    video_index: str = VIDEO_INDEX_NAME,
    segment_index: str = SEGMENT_INDEX_NAME,
    chunk_size: int = 500,
    streaming_bulk_fn: Any | None = None,
) -> dict[str, int]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    helper = streaming_bulk_fn or streaming_bulk
    if helper is None:
        require_elasticsearch()
        raise AssertionError("unreachable")
    indexed = 0
    failures: list[dict[str, Any]] = []
    actions = iter_selector_actions(
        output_root,
        batch_ids=batch_ids,
        video_ids=video_ids,
        video_index=video_index,
        segment_index=segment_index,
    )
    for succeeded, item in helper(
        client,
        actions,
        chunk_size=chunk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if succeeded:
            indexed += 1
        elif len(failures) < 10:
            failures.append(item)
    if failures:
        raise RuntimeError(f"Selector bulk ingestion failed; first errors: {failures}")
    client.indices.refresh(index=f"{video_index},{segment_index}")
    return {
        "indexed": indexed,
        "videos_in_index": int(client.count(index=video_index)["count"]),
        "segments_in_index": int(client.count(index=segment_index)["count"]),
    }


def _id_filter(batch_ids: Sequence[str], video_ids: Sequence[str]) -> list[dict[str, Any]]:
    exact_videos = [item.upper() for item in video_ids if item.strip()]
    if exact_videos:
        return [{"terms": {"video_id": exact_videos}}]
    batches = [item.upper() for item in batch_ids if item.strip()]
    if not batches:
        return []
    clauses = [{"prefix": {"video_id": f"{batch}_"}} for batch in batches]
    return [{"bool": {"should": clauses, "minimum_should_match": 1}}]


def build_video_selection_query(
    parsed: ParsedTemporalQuery,
    *,
    fields: Sequence[str],
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
) -> dict[str, Any]:
    expanded_parts = [parsed.shared_context]
    for event in parsed.events:
        expanded_parts.extend(
            [
                event.text,
                *event.retrieval_prompts,
                *event.target_predicates,
                *event.context_predicates,
            ]
        )
    # Deterministic and Gemini parsers intentionally preserve source text in
    # several fields.  Repeating those strings in one BM25 query would make a
    # generic event (for example "touches the ground") outweigh the actual
    # video description.  Keep every distinct expansion exactly once.
    unique_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in expanded_parts:
        normalized = " ".join(part.split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen_parts:
            seen_parts.add(key)
            unique_parts.append(normalized)
    expanded = " ".join(unique_parts)
    should: list[dict[str, Any]] = []
    if parsed.shared_context:
        should.append(
            {
                "multi_match": {
                    "query": parsed.shared_context,
                    "fields": list(fields),
                    "type": "best_fields",
                    "boost": 8.0,
                }
            }
        )
    should.append(
        {
            "multi_match": {
                "query": expanded,
                "fields": list(fields),
                "type": "best_fields",
                "minimum_should_match": "15%",
            }
        }
    )
    query: dict[str, Any] = {"bool": {"should": should, "minimum_should_match": 1}}
    filters = _id_filter(batch_ids, video_ids)
    if filters:
        query["bool"]["filter"] = filters
    return query


class ElasticsearchVideoSelector:
    """Select videos without resolving any E1..En temporal anchors."""

    VIDEO_FIELDS = (
        "summary_vi^4",
        "summary_en^2",
        "main_entities^3",
        "main_locations^3",
        "main_topics^2",
        "search_text",
    )
    SEGMENT_FIELDS = (
        "title^4",
        "summary^3",
        "entities^3",
        "locations^3",
        "topics^2",
        "asr_text^1.5",
        "search_text",
    )

    def __init__(
        self,
        client: Any,
        *,
        video_index: str = VIDEO_INDEX_NAME,
        segment_index: str = SEGMENT_INDEX_NAME,
    ) -> None:
        self.client = client
        self.video_index = video_index
        self.segment_index = segment_index

    def select(
        self,
        parsed: ParsedTemporalQuery,
        *,
        batch_ids: Sequence[str] = (),
        video_ids: Sequence[str] = (),
        top_k: int = 5,
        per_index_k: int = 100,
        summary_weight: float = 0.95,
        event_weight: float = 0.05,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if summary_weight < 0 or event_weight < 0:
            raise ValueError("video selection weights must be non-negative")
        weight_total = summary_weight + event_weight
        if weight_total <= 0:
            raise ValueError("summary_weight and event_weight cannot both be zero")
        normalized_summary_weight = summary_weight / weight_total
        normalized_event_weight = event_weight / weight_total
        size = max(top_k, per_index_k)
        video_response = self.client.search(
            index=self.video_index,
            size=size,
            query=build_video_selection_query(
                parsed,
                fields=self.VIDEO_FIELDS,
                batch_ids=batch_ids,
                video_ids=video_ids,
            ),
            source=["video_id", "summary_vi", "content_profile"],
        )
        segment_response = self.client.search(
            index=self.segment_index,
            size=size,
            query=build_video_selection_query(
                parsed,
                fields=self.SEGMENT_FIELDS,
                batch_ids=batch_ids,
                video_ids=video_ids,
            ),
            source=["segment_id", "video_id", "title", "summary", "start_ms", "end_ms"],
        )

        evidence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        video_ranks: dict[str, int] = {}
        video_sources: dict[str, dict[str, Any]] = {}
        video_scores: dict[str, float] = {}
        video_hits = video_response.get("hits", {}).get("hits", [])
        for rank, hit in enumerate(video_hits, 1):
            source = hit.get("_source", {})
            video_id = str(source.get("video_id", "")).upper()
            if not video_id:
                continue
            video_ranks[video_id] = rank
            video_sources[video_id] = source
            video_scores[video_id] = float(hit.get("_score") or 0.0)

        segment_hits = segment_response.get("hits", {}).get("hits", [])
        segment_scores: defaultdict[str, list[float]] = defaultdict(list)
        for rank, hit in enumerate(segment_hits, 1):
            source = hit.get("_source", {})
            video_id = str(source.get("video_id", "")).upper()
            if not video_id:
                continue
            segment_scores[video_id].append(float(hit.get("_score") or 0.0))
            if len(evidence[video_id]) < 3:
                evidence[video_id].append(
                    {
                        "segment_id": source.get("segment_id"),
                        "title": source.get("title", ""),
                        "summary": source.get("summary", ""),
                        "start_ms": int(source.get("start_ms", 0)),
                        "end_ms": int(source.get("end_ms", 0)),
                        "elasticsearch_score": float(hit.get("_score") or 0.0),
                        "rank": rank,
                    }
                )

        max_video_score = max(video_scores.values(), default=1.0) or 1.0
        max_segment_score = max(
            (score for values in segment_scores.values() for score in values),
            default=1.0,
        ) or 1.0
        candidate_ids = set(video_scores) | set(segment_scores)
        scores = {
            video_id: (
                normalized_summary_weight * video_scores.get(video_id, 0.0) / max_video_score
                + normalized_event_weight
                * max(segment_scores.get(video_id, [0.0]))
                / max_segment_score
            )
            for video_id in candidate_ids
        }
        ranked_ids = sorted(candidate_ids, key=lambda item: (-scores[item], item))[:top_k]
        candidates = [
            {
                "video_id": video_id,
                "score": round(scores[video_id], 8),
                "video_score": round(video_scores.get(video_id, 0.0), 6),
                "best_segment_score": round(max(segment_scores.get(video_id, [0.0])), 6),
                "summary_score": round(video_scores.get(video_id, 0.0) / max_video_score, 8),
                "event_score": round(
                    max(segment_scores.get(video_id, [0.0])) / max_segment_score,
                    8,
                ),
                "video_rank": video_ranks.get(video_id),
                "summary_vi": video_sources.get(video_id, {}).get("summary_vi", ""),
                "content_profile": video_sources.get(video_id, {}).get("content_profile", ""),
                "segment_evidence": evidence.get(video_id, []),
            }
            for video_id in ranked_ids
        ]
        return {
            "mode": "elasticsearch_video_then_events",
            "selected_video_id": ranked_ids[0] if ranked_ids else None,
            "candidates": candidates,
            "weights": {
                "summary": round(normalized_summary_weight, 8),
                "event": round(normalized_event_weight, 8),
            },
            "video_hits": len(video_hits),
            "segment_hits": len(segment_hits),
        }
