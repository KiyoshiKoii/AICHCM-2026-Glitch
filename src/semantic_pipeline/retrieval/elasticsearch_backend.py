"""Indexing and lexical search for compact Gemini metadata.

This module has no dependency on a running Elasticsearch node until a client
is constructed.  The pure ingestion and query builders are intentionally easy
to test without Docker.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
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


def iter_caption_records(caption_dir: str | Path = DEFAULT_CAPTION_DIR) -> Iterator[CompactVisualRecord]:
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
            if record.frame_id in seen_frame_ids:
                raise ValueError(f"Duplicate frame_id across caption artifacts: {record.frame_id}")
            seen_frame_ids.add(record.frame_id)
            yield record


def build_frame_document(record: CompactVisualRecord, context: VideoContext | None = None) -> dict[str, Any]:
    """Build one strict-mapping Elasticsearch document from a compact record."""

    ref = parse_frame_id(record.frame_id)
    video = context or VideoContext()
    return {
        "frame_id": ref.frame_id,
        "video_id": ref.video_name,
        "program_code": ref.video_name.split("_", 1)[0],
        "frame_number": ref.frame_index,
        "caption": record.caption,
        "detailed_caption": record.detailed_caption,
        "ocr_text": record.ocr_text,
        "news_ticker_text": record.news_ticker_text,
        "video_title": video.title,
        "video_description": video.description,
        "video_keywords": " ".join(video.keywords),
        "series_name": video.series_name,
        "broadcast_slot": video.broadcast_slot,
        "channel_name": video.channel_name,
        "source_network": video.source_network,
        "episode_date": video.episode_date,
    }


def iter_bulk_actions(
    caption_dir: str | Path = DEFAULT_CAPTION_DIR,
    youtube_metadata: str | Path = DEFAULT_YOUTUBE_METADATA,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Iterator[dict[str, Any]]:
    """Build idempotent bulk actions, using ``frame_id`` as ``_id``."""

    contexts = load_video_contexts(youtube_metadata)
    for record in iter_caption_records(caption_dir):
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
    actions = iter_bulk_actions(caption_dir, youtube_metadata, index_name)
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


def build_lexical_query(keywords: Sequence[str]) -> dict[str, Any]:
    """Build a weighted, field-aware BM25 query from parsed query keywords.

    A parsed query contains translations and near-synonyms, so each item is a
    recall-oriented ``should`` clause rather than one over-constrained bag of
    terms.  Captions use English stemming; OCR and program metadata preserve
    multilingual literal matching.
    """

    terms = _clean_keywords(keywords)
    if not terms:
        return {"match_none": {}}

    should: list[dict[str, Any]] = []
    for term in terms:
        should.extend(
            [
                {
                    "combined_fields": {
                        "query": term,
                        "fields": ["caption^4", "detailed_caption^2"],
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
                            "fields": ["caption^2", "detailed_caption"],
                            "type": "phrase",
                        }
                    },
                ]
            )
    return {"bool": {"should": should, "minimum_should_match": 1}}


class ElasticsearchTextSearch:
    """Small adapter that implements the existing internal semantic API shape."""

    def __init__(self, client: Any | None = None, *, index: str | None = None, url: str | None = None) -> None:
        self.client = client or create_client(url)
        self.index = index or os.getenv("ELASTICSEARCH_INDEX", DEFAULT_ALIAS_NAME)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def document_count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def search(self, keywords: Sequence[str], *, top_k: int = 200) -> list[dict[str, Any]]:
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")
        query = build_lexical_query(keywords)
        if "match_none" in query:
            return []
        response = self.client.search(
            index=self.index,
            size=top_k,
            query=query,
            source=[
                "frame_id",
                "video_id",
                "frame_number",
                "caption",
                "detailed_caption",
                "ocr_text",
                "news_ticker_text",
            ],
            highlight={
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fields": {
                    "caption": {},
                    "detailed_caption": {},
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
                        "caption": source.get("caption", ""),
                        "detailed_caption": source.get("detailed_caption", ""),
                        "ocr_text": source.get("ocr_text", ""),
                        "news_ticker_text": source.get("news_ticker_text", ""),
                        "highlights": hit.get("highlight", {}),
                    },
                }
            )
        return results
