"""Indexing and lexical search for compact Gemini metadata.

This module has no dependency on a running Elasticsearch node until a client
is constructed.  The pure ingestion and query builders are intentionally easy
to test without Docker.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_pipeline.core.compact_metadata import CompactVisualRecord
from semantic_pipeline.core.frame_id import parse_frame_id
from semantic_pipeline.retrieval.index_definition import (
    DEFAULT_ALIAS_NAME,
    DEFAULT_INDEX_NAME,
    index_definition,
)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import streaming_bulk
except ImportError:  # pragma: no cover - exercised only in incomplete setups.
    Elasticsearch = None  # type: ignore[assignment]
    streaming_bulk = None


DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
DEFAULT_CAPTION_DIR = Path("data/metadata/caption")
DEFAULT_YOUTUBE_METADATA = Path("data/metadata/metadata_youtube.jsonl")


class ElasticsearchUnavailable(RuntimeError):
    """Raised when the optional client package is not installed."""


@dataclass(frozen=True)
class VideoContext:
    title: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()
    series_name: str = ""
    broadcast_slot: str = ""
    channel_name: str = ""
    source_network: str = ""
    episode_date: str | None = None


def require_elasticsearch() -> None:
    if Elasticsearch is None:
        raise ElasticsearchUnavailable(
            "Missing dependency 'elasticsearch'. Install requirements.txt first."
        )


def create_client(url: str | None = None):
    """Create the official client from an explicit URL or environment variable."""

    require_elasticsearch()
    resolved_url = url or os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL)
    return Elasticsearch(resolved_url, request_timeout=30)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            values.append(text)
    return tuple(values)


def load_video_contexts(path: str | Path = DEFAULT_YOUTUBE_METADATA) -> dict[str, VideoContext]:
    """Load only compact, useful program fields keyed by ``video_id``."""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        return {}

    contexts: dict[str, VideoContext] = {}
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid YouTube metadata JSON at line {line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"YouTube metadata line {line_number} must be an object")
        video_id = _text(item.get("video_id"))
        if not video_id:
            raise ValueError(f"YouTube metadata line {line_number} has no video_id")

        search_fields = item.get("search_fields")
        search_fields = search_fields if isinstance(search_fields, dict) else {}
        filters = item.get("filters")
        filters = filters if isinstance(filters, dict) else {}
        payload = item.get("payload")
        payload = payload if isinstance(payload, dict) else {}

        keyword_values = _text_values(search_fields.get("keywords")) + _text_values(
            search_fields.get("keywords_common")
        )
        contexts[video_id] = VideoContext(
            title=_text(search_fields.get("title")) or _text(payload.get("title")),
            description=_text(search_fields.get("description")) or _text(payload.get("description")),
            keywords=tuple(dict.fromkeys(keyword_values)),
            series_name=_text(search_fields.get("series_name")) or _text(filters.get("series_name")),
            broadcast_slot=_text(search_fields.get("broadcast_slot")) or _text(filters.get("broadcast_slot")),
            channel_name=_text(search_fields.get("channel_name")) or _text(filters.get("channel_name")),
            source_network=_text(search_fields.get("source_network")) or _text(filters.get("source_network")),
            episode_date=_text(search_fields.get("episode_date")) or _text(filters.get("episode_date")) or None,
        )
    return contexts


def iter_caption_records(
    caption_dir: str | Path = DEFAULT_CAPTION_DIR,
    *,
    require_provenance: bool = False,
) -> Iterator[CompactVisualRecord]:
    """Yield validated records from all per-video Gemini checkpoints."""

    root = Path(caption_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Caption directory does not exist: {root}")
    files = sorted(root.glob("L*/L*_V*.json"))
    if not files:
        raise FileNotFoundError(f"No per-video caption JSON files found under {root}")

    seen_frame_ids: set[str] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid caption JSON: {path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Caption artifact must be a JSON array: {path}")
        for position, raw_record in enumerate(payload):
            try:
                record = CompactVisualRecord.model_validate(raw_record)
            except Exception as exc:
                raise ValueError(f"Invalid caption record {path}[{position}]") from exc
            if require_provenance:
                missing = [
                    name
                    for name, value in (
                        ("visual_source_frame_id", record.visual_source_frame_id),
                        ("ocr_source_frame_id", record.ocr_source_frame_id),
                    )
                    if not value
                ]
                if missing:
                    raise ValueError(
                        f"Caption record {path}[{position}] is missing provenance: {missing}"
                    )
                if record.ocr_source_frame_id != record.frame_id:
                    raise ValueError(
                        f"Caption record {path}[{position}] must source OCR from itself; "
                        f"got {record.ocr_source_frame_id!r} for {record.frame_id!r}"
                    )
            if record.frame_id in seen_frame_ids:
                raise ValueError(f"Duplicate frame_id across caption artifacts: {record.frame_id}")
            seen_frame_ids.add(record.frame_id)
            yield record


def _relation_document(record: CompactVisualRecord, relation: Any) -> dict[str, Any]:
    detections = {item.object_id: item for item in record.detections}
    subject = detections.get(relation.subject_id)
    object_ = detections.get(relation.object_id)
    if subject is None or object_ is None:
        raise ValueError(
            f"Spatial relation in {record.frame_id} references an unknown detection: "
            f"{relation.subject_id!r}, {relation.object_id!r}"
        )

    def text_fields(prefix: str, detection: Any) -> dict[str, Any]:
        return {
            f"{prefix}_label": detection.label,
            f"{prefix}_description": detection.description,
            f"{prefix}_description_vi": detection.description_vi,
            f"{prefix}_attributes": list(detection.attributes),
            f"{prefix}_action": detection.action,
        }

    document = {
        "subject_id": relation.subject_id,
        "predicate": relation.predicate,
        "object_id": relation.object_id,
        **text_fields("subject", subject),
        **text_fields("object", object_),
    }
    return document


def build_frame_document(record: CompactVisualRecord, context: VideoContext | None = None) -> dict[str, Any]:
    """Build one strict-mapping Elasticsearch document from a compact record."""

    ref = parse_frame_id(record.frame_id)
    video = context or VideoContext()
    visual_source_frame_id = record.visual_source_frame_id or record.frame_id
    ocr_source_frame_id = record.ocr_source_frame_id or record.frame_id
    document = {
        "frame_id": ref.frame_id,
        "video_id": ref.video_name,
        "program_code": ref.video_name.split("_", 1)[0],
        "frame_number": (
            record.native_frame_index
            if record.native_frame_index is not None
            else ref.frame_index
        ),
        "visual_source_frame_id": visual_source_frame_id,
        "ocr_source_frame_id": ocr_source_frame_id,
        "quality_flags": list(record.quality_flags),
        "is_visual_representative": visual_source_frame_id == record.frame_id,
        "caption": record.caption,
        "detailed_caption": record.detailed_caption,
        "caption_vi": record.caption_vi,
        "detailed_caption_vi": record.detailed_caption_vi,
        "ocr_text": record.ocr_text,
        "news_ticker_text": record.news_ticker_text,
        "detections": [item.model_dump(mode="json") for item in record.detections],
        "spatial_relations": [
            _relation_document(record, relation) for relation in record.spatial_relations
        ],
        "video_title": video.title,
        "video_description": video.description,
        "video_keywords": " ".join(video.keywords),
        "series_name": video.series_name,
        "broadcast_slot": video.broadcast_slot,
        "channel_name": video.channel_name,
        "source_network": video.source_network,
        "episode_date": video.episode_date,
    }
    if record.keyframe_n is not None:
        document["keyframe_number"] = record.keyframe_n
    if record.asr_available or record.has_asr or record.asr_text:
        document.update(
            {
                "asr_available": record.asr_available,
                "has_asr": record.has_asr,
                "asr_text": record.asr_text,
                "asr_segment_indices": list(record.asr_segment_indices),
                "asr_start_ms": record.asr_start_ms,
                "asr_end_ms": record.asr_end_ms,
                "asr_quality_flags": list(record.asr_quality_flags),
            }
        )
    return document


def iter_bulk_actions(
    caption_dir: str | Path = DEFAULT_CAPTION_DIR,
    youtube_metadata: str | Path = DEFAULT_YOUTUBE_METADATA,
    index_name: str = DEFAULT_INDEX_NAME,
    require_provenance: bool = False,
) -> Iterator[dict[str, Any]]:
    """Build idempotent bulk actions, using ``frame_id`` as ``_id``."""

    contexts = load_video_contexts(youtube_metadata)
    for record in iter_caption_records(caption_dir, require_provenance=require_provenance):
        source = build_frame_document(record, contexts.get(parse_frame_id(record.frame_id).video_name))
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": record.frame_id,
            "_source": source,
        }


def ensure_index(client: Any, index_name: str = DEFAULT_INDEX_NAME) -> bool:
    """Create a physical, versioned index if it does not already exist."""

    if bool(client.indices.exists(index=index_name)):
        return False
    client.indices.create(index=index_name, **index_definition())
    return True


def _not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or getattr(
        getattr(exc, "meta", None), "status", None
    ) == 404


def activate_alias(
    client: Any,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
    alias_name: str = DEFAULT_ALIAS_NAME,
) -> None:
    """Atomically point the stable read alias at one physical index."""

    try:
        current = client.indices.get_alias(name=alias_name)
    except Exception as exc:
        if not _not_found(exc):
            raise
        current = {}
    actions = [
        {"remove": {"index": old_index, "alias": alias_name}}
        for old_index in current
        if old_index != index_name
    ]
    actions.append(
        {
            "add": {
                "index": index_name,
                "alias": alias_name,
                "is_write_index": True,
            }
        }
    )
    client.indices.update_aliases(actions=actions)


def bulk_ingest(
    client: Any,
    *,
    caption_dir: str | Path = DEFAULT_CAPTION_DIR,
    youtube_metadata: str | Path = DEFAULT_YOUTUBE_METADATA,
    index_name: str = DEFAULT_INDEX_NAME,
    chunk_size: int = 500,
    require_provenance: bool = False,
    streaming_bulk_fn: Any | None = None,
) -> dict[str, int]:
    """Index every current caption checkpoint and refresh the physical index."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    helper = streaming_bulk_fn or streaming_bulk
    if helper is None:
        require_elasticsearch()
        raise AssertionError("unreachable")

    indexed = 0
    failures: list[dict[str, Any]] = []
    actions = iter_bulk_actions(
        caption_dir,
        youtube_metadata,
        index_name,
        require_provenance=require_provenance,
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
        raise RuntimeError(f"Bulk ingestion failed; first errors: {failures}")

    client.indices.refresh(index=index_name)
    total = int(client.count(index=index_name)["count"])
    return {"indexed": indexed, "documents_in_index": total}


def _clean_keywords(keywords: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result[:20]


def _clean_object_queries(
    object_queries: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, str]]:
    """Keep paired English/Vietnamese phrases for one object constraint."""

    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in object_queries or ():
        if not isinstance(item, Mapping):
            continue
        english = _text(item.get("english_phrase"))
        vietnamese = _text(item.get("vietnamese_phrase"))
        if not english or not vietnamese:
            continue
        key = (english.casefold(), vietnamese.casefold())
        if key not in seen:
            seen.add(key)
            result.append((english, vietnamese))
    return result[:5]


def _clean_scope_ids(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        normalized = _text(value).upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


OBJECT_COVERAGE_BOOST = 2.0
SPATIAL_PREDICATES = frozenset({"left_of", "right_of", "above", "below", "overlapping"})
INTERACTION_BOOST = 20.0
INTERACTION_COVERAGE_BOOST = 30.0


def _object_nested_clause(english_phrase: str, vietnamese_phrase: str) -> dict[str, Any]:
    """Match one object's attributes inside the same nested detection."""
    return {
        "nested": {
            "path": "detections",
            "score_mode": "max",
            "query": {
                "bool": {
                    "should": [
                        {
                            "combined_fields": {
                                "query": english_phrase,
                                "fields": [
                                    "detections.description",
                                    "detections.attributes",
                                    "detections.action",
                                    "detections.label",
                                ],
                                "operator": "and",
                            }
                        },
                        {
                            "match": {
                                "detections.description_vi": {
                                    "query": vietnamese_phrase,
                                    "operator": "and",
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
    }


def _clean_spatial_queries(
    spatial_queries: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, str, str, str, str]]:
    """Normalize bilingual subject/predicate/object constraints for one relation."""

    result: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in spatial_queries or ():
        if not isinstance(item, Mapping):
            continue
        subject_en = _text(item.get("subject_english_phrase") or item.get("subject_phrase"))
        subject_vi = _text(item.get("subject_vietnamese_phrase"))
        predicate = _text(item.get("predicate")).casefold()
        object_en = _text(item.get("object_english_phrase") or item.get("object_phrase"))
        object_vi = _text(item.get("object_vietnamese_phrase"))
        if not subject_en or not object_en or predicate not in SPATIAL_PREDICATES:
            continue
        key = (subject_en.casefold(), subject_vi.casefold(), predicate, object_en.casefold(), object_vi.casefold())
        if key not in seen:
            seen.add(key)
            result.append((subject_en, subject_vi, predicate, object_en, object_vi))
    return result[:5]


def _clean_interaction_queries(
    interaction_queries: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, str, str, str, str, str]]:
    """Normalize subject-action-object constraints for one relation row."""

    result: list[tuple[str, str, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for item in interaction_queries or ():
        if not isinstance(item, Mapping):
            continue
        subject_en = _text(item.get("subject_english_phrase"))
        subject_vi = _text(item.get("subject_vietnamese_phrase"))
        action_en = _text(item.get("action_english_phrase"))
        action_vi = _text(item.get("action_vietnamese_phrase"))
        object_en = _text(item.get("object_english_phrase"))
        object_vi = _text(item.get("object_vietnamese_phrase"))
        if not subject_en or not action_en or not object_en:
            continue
        key = (
            subject_en.casefold(), subject_vi.casefold(), action_en.casefold(),
            action_vi.casefold(), object_en.casefold(), object_vi.casefold(),
        )
        if key not in seen:
            seen.add(key)
            result.append((subject_en, subject_vi, action_en, action_vi, object_en, object_vi))
    return result[:5]


def _relation_entity_clause(
    english_phrase: str,
    vietnamese_phrase: str,
    fields: Sequence[str],
    vietnamese_fields: Sequence[str],
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [
        {
            "combined_fields": {
                "query": english_phrase,
                "fields": list(fields),
                "operator": "and",
            }
        }
    ]
    if vietnamese_phrase:
        clauses.append(
            {
                "multi_match": {
                    "query": vietnamese_phrase,
                    "fields": list(vietnamese_fields),
                    "operator": "and",
                }
            }
        )
    return {"bool": {"should": clauses, "minimum_should_match": 1}}


def _relation_action_clause(action_en: str, action_vi: str, prefix: str) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [
        {
            "combined_fields": {
                "query": action_en,
                "fields": [
                    f"spatial_relations.{prefix}_action",
                    f"spatial_relations.{prefix}_description",
                    f"spatial_relations.{prefix}_attributes",
                ],
                "operator": "and",
            }
        }
    ]
    if action_vi:
        clauses.append(
            {
                "match": {
                    f"spatial_relations.{prefix}_description_vi": {
                        "query": action_vi,
                        "operator": "and",
                    }
                }
            }
        )
    return {"bool": {"should": clauses, "minimum_should_match": 1}}


def _interaction_nested_clause(
    subject_en: str,
    subject_vi: str,
    action_en: str,
    action_vi: str,
    object_en: str,
    object_vi: str,
) -> dict[str, Any]:
    """Match subject, action, and object inside one denormalized relation row."""

    return {
        "nested": {
            "path": "spatial_relations",
            "score_mode": "max",
            "boost": INTERACTION_BOOST,
            "query": {
                "bool": {
                    "must": [
                        _relation_entity_clause(
                            subject_en,
                            subject_vi,
                            [
                                "spatial_relations.subject_label",
                                "spatial_relations.subject_description",
                                "spatial_relations.subject_attributes",
                                "spatial_relations.subject_action",
                            ],
                            ["spatial_relations.subject_description_vi"],
                        ),
                        _relation_action_clause(action_en, action_vi, "subject"),
                        _relation_entity_clause(
                            object_en,
                            object_vi,
                            [
                                "spatial_relations.object_label",
                                "spatial_relations.object_description",
                                "spatial_relations.object_attributes",
                                "spatial_relations.object_action",
                            ],
                            ["spatial_relations.object_description_vi"],
                        ),
                    ]
                }
            },
        }
    }


def _spatial_nested_clause(
    subject_en: str,
    subject_vi: str,
    predicate: str,
    object_en: str,
    object_vi: str,
) -> dict[str, Any]:
    return {
        "nested": {
            "path": "spatial_relations",
            "score_mode": "max",
            "boost": 4.0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"spatial_relations.predicate": predicate}},
                        _relation_entity_clause(
                            subject_en,
                            subject_vi,
                            [
                                "spatial_relations.subject_label",
                                "spatial_relations.subject_description",
                                "spatial_relations.subject_attributes",
                                "spatial_relations.subject_action",
                            ],
                            ["spatial_relations.subject_description_vi"],
                        ),
                        _relation_entity_clause(
                            object_en,
                            object_vi,
                            [
                                "spatial_relations.object_label",
                                "spatial_relations.object_description",
                                "spatial_relations.object_attributes",
                                "spatial_relations.object_action",
                            ],
                            ["spatial_relations.object_description_vi"],
                        ),
                    ]
                }
            },
        }
    }


def build_lexical_query(
    keywords: Sequence[str],
    *,
    object_queries: Sequence[Mapping[str, Any]] | None = None,
    ocr_queries: Sequence[str] | None = None,
    program_queries: Sequence[str] | None = None,
    batch_ids: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    spatial_queries: Sequence[Mapping[str, Any]] | None = None,
    interaction_queries: Sequence[Mapping[str, Any]] | None = None,
    exclude_quality_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a weighted, field-aware BM25 query from parsed query keywords.

    A parsed query contains translations and near-synonyms, so each item is a
    recall-oriented ``should`` clause rather than one over-constrained bag of
    terms. Detailed captions are the primary visual retrieval evidence;
    concise captions only provide a small secondary boost because they are
    intentionally lossy and can be generic in large Gemini batches. Captions
    use English stemming; OCR and program metadata preserve multilingual
    literal matching.
    """

    terms = _clean_keywords(keywords)
    objects = _clean_object_queries(object_queries)
    ocr_terms = _clean_keywords(ocr_queries or ())
    program_terms = _clean_keywords(program_queries or ())
    batches = _clean_scope_ids(batch_ids)
    videos = _clean_scope_ids(video_ids)
    spatial = _clean_spatial_queries(spatial_queries)
    interactions = _clean_interaction_queries(interaction_queries)
    if not (terms or objects or ocr_terms or program_terms or batches or videos or spatial or interactions):
        return {"match_none": {}}

    should: list[dict[str, Any]] = []
    for term in terms:
        should.extend(
            [
                {
                    "combined_fields": {
                        "query": term,
                        "fields": ["detailed_caption^4", "caption"],
                        "operator": "and",
                    }
                },
                {
                    "combined_fields": {
                        "query": term,
                        "fields": ["detailed_caption_vi^4", "caption_vi"],
                        "operator": "and",
                    }
                },
                {
                    "multi_match": {
                        "query": term,
                        "fields": ["ocr_text^5", "news_ticker_text^6"],
                        "type": "best_fields",
                        "operator": "and",
                    }
                },
                {
                    "multi_match": {
                        "query": term,
                        "fields": [
                            "video_title^1.5",
                            "video_keywords^1.25",
                            "video_description^0.5",
                        ],
                        "type": "best_fields",
                        "operator": "and",
                    }
                },
                {
                    "nested": {
                        "path": "detections",
                        "score_mode": "max",
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "combined_fields": {
                                            "query": term,
                                            "fields": [
                                                "detections.description",
                                                "detections.attributes",
                                                "detections.action",
                                                "detections.label",
                                            ],
                                            "operator": "and",
                                        }
                                    },
                                    {
                                        "match": {
                                            "detections.description_vi": {
                                                "query": term,
                                                "operator": "and",
                                            }
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                    }
                },
            ]
        )
        if len(term.split()) > 1:
            should.extend(
                [
                    {
                        "multi_match": {
                            "query": term,
                            "fields": ["ocr_text^8", "news_ticker_text^10"],
                            "type": "phrase",
                        }
                    },
                    {
                        "multi_match": {
                            "query": term,
                            "fields": ["detailed_caption^2", "caption"],
                            "type": "phrase",
                        }
                    },
                    {
                        "multi_match": {
                            "query": term,
                            "fields": ["detailed_caption_vi^2", "caption_vi"],
                            "type": "phrase",
                        }
                    },
                ]
            )

    # Each clause searches one nested document, so all visual constraints in
    # an object phrase must belong to the same detected object.
    for english_phrase, vietnamese_phrase in objects:
        should.append(_object_nested_clause(english_phrase, vietnamese_phrase))

    for phrase in ocr_terms:
        should.append(
            {
                "multi_match": {
                    "query": phrase,
                    "fields": ["ocr_text", "news_ticker_text"],
                    "type": "phrase",
                }
            }
        )

    for phrase in program_terms:
        should.append(
            {
                "multi_match": {
                    "query": phrase,
                    "fields": [
                        "video_title",
                        "video_keywords",
                        "video_description",
                        "series_name",
                        "broadcast_slot",
                        "channel_name",
                        "source_network",
                    ],
                    "type": "phrase",
                }
            }
        )
    for spatial_query in spatial:
        should.append(_spatial_nested_clause(*spatial_query))
    for interaction_query in interactions:
        should.append(_interaction_nested_clause(*interaction_query))
    filters: list[dict[str, Any]] = []
    if batches:
        filters.append({"terms": {"program_code": [batch.lower() for batch in batches]}})
    if videos:
        exact_videos = [video for video in videos if "_" in video]
        suffix_videos = [video for video in videos if "_" not in video]
        video_should: list[dict[str, Any]] = []
        if exact_videos:
            video_should.append({"terms": {"video_id": exact_videos}})
        video_should.extend(
            {"wildcard": {"video_id": {"value": f"*_{suffix}"}}}
            for suffix in suffix_videos
        )
        filters.append(
            video_should[0]
            if len(video_should) == 1
            else {"bool": {"should": video_should, "minimum_should_match": 1}}
        )
    excluded_flags = [_text(value) for value in (exclude_quality_flags or ()) if _text(value)]
    if excluded_flags:
        filters.append({"bool": {"must_not": [{"terms": {"quality_flags": excluded_flags}}]}})

    base_query = {"bool": {"should": should, "minimum_should_match": 1}}
    if filters:
        base_query["bool"]["filter"] = filters
    if not objects and not interactions:
        return base_query

    # Keep partial matches for recall, but explicitly reward object coverage:
    # each independently matched object contributes one additive boost. This
    # makes a 3/3 object match outrank an otherwise similar 1/3 match.
    coverage_functions = [
        {
            "filter": _object_nested_clause(english_phrase, vietnamese_phrase),
            "weight": OBJECT_COVERAGE_BOOST,
        }
        for english_phrase, vietnamese_phrase in objects
    ]
    coverage_functions.extend(
        {
            "filter": _interaction_nested_clause(*interaction_query),
            "weight": INTERACTION_COVERAGE_BOOST,
        }
        for interaction_query in interactions
    )
    return {
        "function_score": {
            "query": base_query,
            "functions": coverage_functions,
            "score_mode": "sum",
            "boost_mode": "sum",
        }
    }


class ElasticsearchTextSearch:
    """Small adapter that implements the existing internal semantic API shape."""

    def __init__(self, client: Any | None = None, *, index: str | None = None, url: str | None = None) -> None:
        self.client = client or create_client(url)
        self.index = index or os.getenv("ELASTICSEARCH_INDEX", DEFAULT_ALIAS_NAME)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def document_count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def search(
        self,
        keywords: Sequence[str],
        *,
        object_queries: Sequence[Mapping[str, Any]] | None = None,
        ocr_queries: Sequence[str] | None = None,
        program_queries: Sequence[str] | None = None,
        batch_ids: Sequence[str] | None = None,
        video_ids: Sequence[str] | None = None,
        spatial_queries: Sequence[Mapping[str, Any]] | None = None,
        interaction_queries: Sequence[Mapping[str, Any]] | None = None,
        exclude_quality_flags: Sequence[str] | None = None,
        collapse_visual_duplicates: bool = False,
        top_k: int = 200,
    ) -> list[dict[str, Any]]:
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")
        query = build_lexical_query(
            keywords,
            object_queries=object_queries,
            ocr_queries=ocr_queries,
            program_queries=program_queries,
            batch_ids=batch_ids,
            video_ids=video_ids,
            spatial_queries=spatial_queries,
            interaction_queries=interaction_queries,
            exclude_quality_flags=exclude_quality_flags,
        )
        if "match_none" in query:
            return []
        response = self.client.search(
            index=self.index,
            size=top_k,
            query=query,
            **(
                {"collapse": {"field": "visual_source_frame_id"}}
                if collapse_visual_duplicates
                else {}
            ),
            source=[
                "frame_id",
                "video_id",
                "frame_number",
                "visual_source_frame_id",
                "ocr_source_frame_id",
                "quality_flags",
                "is_visual_representative",
                "caption",
                "detailed_caption",
                "caption_vi",
                "detailed_caption_vi",
                "ocr_text",
                "news_ticker_text",
            ],
            highlight={
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fields": {
                    "caption": {},
                    "detailed_caption": {},
                    "caption_vi": {},
                    "detailed_caption_vi": {},
                    "ocr_text": {},
                    "news_ticker_text": {},
                },
            },
        )
        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append(
                {
                    "frame_id": source["frame_id"],
                    "score": float(hit.get("_score") or 0.0),
                    "video_name": source["video_id"],
                    "frame_index": int(source["frame_number"]),
                    "metadata": {
                        "visual_source_frame_id": source.get(
                            "visual_source_frame_id", source["frame_id"]
                        ),
                        "ocr_source_frame_id": source.get(
                            "ocr_source_frame_id", source["frame_id"]
                        ),
                        "quality_flags": source.get("quality_flags", []),
                        "is_visual_representative": source.get(
                            "is_visual_representative", True
                        ),
                        "caption": source.get("caption", ""),
                        "detailed_caption": source.get("detailed_caption", ""),
                        "caption_vi": source.get("caption_vi", ""),
                        "detailed_caption_vi": source.get("detailed_caption_vi", ""),
                        "ocr_text": source.get("ocr_text", ""),
                        "news_ticker_text": source.get("news_ticker_text", ""),
                        "highlights": hit.get("highlight", {}),
                    },
                }
            )
        return results
