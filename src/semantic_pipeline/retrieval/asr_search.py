"""Timestamp-preserving Elasticsearch retrieval over ASR transcripts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from semantic_pipeline.retrieval.elasticsearch_backend import (
    DEFAULT_ELASTICSEARCH_URL,
    create_client,
    require_elasticsearch,
)

try:
    from elasticsearch.helpers import streaming_bulk
except ImportError:  # pragma: no cover - handled by require_elasticsearch.
    streaming_bulk = None


ASR_INDEX_NAME = "semantic_asr_segments_v1"
DEFAULT_ASR_DIR = Path("data/metadata/metadata_asr")
DEFAULT_WINDOW_MS = 20_000
DEFAULT_MAX_GAP_MS = 2_500
DEFAULT_OVERLAP_SEGMENTS = 1


def asr_index_definition() -> dict[str, Any]:
    """Keep accented Vietnamese tokens while retaining a folded fallback."""

    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "vi_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    },
                    "folded_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "document_type": {"type": "keyword"},
                "asr_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "start_ms": {"type": "integer"},
                "end_ms": {"type": "integer"},
                "start_segment_index": {"type": "integer"},
                "end_segment_index": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "vi_text",
                    "fields": {
                        "folded": {"type": "text", "analyzer": "folded_text"},
                    },
                },
            },
        },
    }


def _normalized_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def load_asr_segments(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    video_id = _normalized_text(payload.get("video_name")).upper() or path.stem.upper()
    if not video_id:
        raise ValueError(f"{path}: missing video_name")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError(f"{path}: segments must be an array")

    segments: list[dict[str, Any]] = []
    for source_index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        text = _normalized_text(raw.get("text"))
        if not text:
            continue
        try:
            start_ms = max(0, round(float(raw.get("start")) * 1_000))
            end_ms = max(start_ms, round(float(raw.get("end")) * 1_000))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid timestamp at segments[{source_index}]") from exc
        segments.append(
            {
                "source_index": source_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
            }
        )
    segments.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["source_index"]))
    return video_id, segments


def build_asr_documents(
    video_id: str,
    segments: Sequence[dict[str, Any]],
    *,
    window_ms: int = DEFAULT_WINDOW_MS,
    max_gap_ms: int = DEFAULT_MAX_GAP_MS,
    overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS,
) -> list[dict[str, Any]]:
    """Merge short Whisper segments into searchable, slightly overlapping windows."""

    if window_ms < 1 or max_gap_ms < 0 or overlap_segments < 0:
        raise ValueError("invalid ASR window settings")
    documents: list[dict[str, Any]] = []
    start = 0
    while start < len(segments):
        end = start + 1
        while end < len(segments):
            previous = segments[end - 1]
            candidate = segments[end]
            gap_ms = candidate["start_ms"] - previous["end_ms"]
            duration_ms = candidate["end_ms"] - segments[start]["start_ms"]
            if gap_ms > max_gap_ms or duration_ms > window_ms:
                break
            end += 1

        window = segments[start:end]
        asr_id = f"{video_id}_asr_{len(documents) + 1:06d}"
        documents.append(
            {
                "document_type": "asr_segment",
                "asr_id": asr_id,
                "video_id": video_id,
                "start_ms": int(window[0]["start_ms"]),
                "end_ms": int(window[-1]["end_ms"]),
                "start_segment_index": int(window[0]["source_index"]),
                "end_segment_index": int(window[-1]["source_index"]),
                "text": " ".join(item["text"] for item in window),
            }
        )
        if end >= len(segments):
            break
        start = max(start + 1, end - overlap_segments)
    return documents


def iter_asr_documents(
    asr_dir: Path = DEFAULT_ASR_DIR,
    *,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
) -> Iterator[dict[str, Any]]:
    batches = {item.strip().upper() for item in batch_ids if item.strip()}
    videos = {item.strip().upper() for item in video_ids if item.strip()}
    files = sorted(asr_dir.glob("L*_V*.json"))
    if not files:
        raise FileNotFoundError(f"No per-video ASR files found under {asr_dir}")
    for path in files:
        file_video_id = path.stem.upper()
        if batches and file_video_id.split("_", 1)[0] not in batches:
            continue
        if videos and file_video_id not in videos:
            continue
        video_id, segments = load_asr_segments(path)
        yield from build_asr_documents(video_id, segments)


def iter_asr_actions(
    asr_dir: Path = DEFAULT_ASR_DIR,
    *,
    index_name: str = ASR_INDEX_NAME,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
) -> Iterable[dict[str, Any]]:
    for document in iter_asr_documents(asr_dir, batch_ids=batch_ids, video_ids=video_ids):
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": document["asr_id"],
            "_source": document,
        }


def ensure_asr_index(client: Any, index_name: str = ASR_INDEX_NAME) -> bool:
    if bool(client.indices.exists(index=index_name)):
        return False
    client.indices.create(index=index_name, **asr_index_definition())
    return True


def bulk_ingest_asr(
    client: Any,
    *,
    asr_dir: Path = DEFAULT_ASR_DIR,
    index_name: str = ASR_INDEX_NAME,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
    chunk_size: int = 1_000,
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
    for succeeded, item in helper(
        client,
        iter_asr_actions(
            asr_dir,
            index_name=index_name,
            batch_ids=batch_ids,
            video_ids=video_ids,
        ),
        chunk_size=chunk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if succeeded:
            indexed += 1
        elif len(failures) < 10:
            failures.append(item)
    if failures:
        raise RuntimeError(f"ASR bulk ingestion failed; first errors: {failures}")
    client.indices.refresh(index=index_name)
    return {
        "indexed": indexed,
        "documents_in_index": int(client.count(index=index_name)["count"]),
    }


def _scope_filter(batch_ids: Sequence[str], video_ids: Sequence[str]) -> list[dict[str, Any]]:
    batches = [item.strip().upper() for item in batch_ids if item.strip()]
    videos = [item.strip().upper() for item in video_ids if item.strip()]
    if videos:
        exact: list[str] = []
        suffixes: list[str] = []
        for video_id in videos:
            if "_" in video_id:
                exact.append(video_id)
            elif batches:
                exact.extend(f"{batch}_{video_id}" for batch in batches)
            else:
                suffixes.append(video_id)
        clauses: list[dict[str, Any]] = []
        if exact:
            clauses.append({"terms": {"video_id": exact}})
        clauses.extend({"wildcard": {"video_id": {"value": f"*_{suffix}"}}} for suffix in suffixes)
        return [{"bool": {"should": clauses, "minimum_should_match": 1}}]
    if batches:
        return [
            {
                "bool": {
                    "should": [{"prefix": {"video_id": f"{batch}_"}} for batch in batches],
                    "minimum_should_match": 1,
                }
            }
        ]
    return []


def build_asr_query(
    query: str,
    *,
    batch_ids: Sequence[str] = (),
    video_ids: Sequence[str] = (),
) -> dict[str, Any]:
    normalized = _normalized_text(query)
    if not normalized:
        raise ValueError("ASR query must not be blank")
    lexical = {
        "multi_match": {
            "query": normalized,
            "fields": ["text^3", "text.folded"],
            "type": "best_fields",
            "minimum_should_match": "35%",
        }
    }
    phrase = {
        "match_phrase": {
            "text": {
                "query": normalized,
                "slop": 4,
                "boost": 6.0,
            }
        }
    }
    body: dict[str, Any] = {
        "bool": {
            "must": [lexical],
            "should": [phrase],
        }
    }
    filters = _scope_filter(batch_ids, video_ids)
    if filters:
        body["bool"]["filter"] = filters
    return body


def _overlaps_existing(hit: dict[str, Any], selected: Sequence[dict[str, Any]]) -> bool:
    for existing in selected:
        if hit["video_id"] != existing["video_id"]:
            continue
        overlap = max(
            0,
            min(hit["end_ms"], existing["end_ms"])
            - max(hit["start_ms"], existing["start_ms"]),
        )
        shorter = max(1, min(
            hit["end_ms"] - hit["start_ms"],
            existing["end_ms"] - existing["start_ms"],
        ))
        if overlap / shorter >= 0.5:
            return True
    return False


class ElasticsearchASRSearch:
    def __init__(
        self,
        client: Any | None = None,
        *,
        index: str = ASR_INDEX_NAME,
        url: str = DEFAULT_ELASTICSEARCH_URL,
    ) -> None:
        self.client = client or create_client(url)
        self.index = index

    def search(
        self,
        query: str,
        *,
        batch_ids: Sequence[str] = (),
        video_ids: Sequence[str] = (),
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        response = self.client.search(
            index=self.index,
            size=min(max(top_k * 4, 100), 1_000),
            query=build_asr_query(query, batch_ids=batch_ids, video_ids=video_ids),
            source=[
                "asr_id",
                "video_id",
                "start_ms",
                "end_ms",
                "start_segment_index",
                "end_segment_index",
                "text",
            ],
        )
        raw_hits: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            video_id = str(source.get("video_id", "")).upper()
            if not video_id:
                continue
            raw_hits.append(
                {
                    **source,
                    "video_id": video_id,
                    "score": float(hit.get("_score") or 0.0),
                }
            )
        max_score = max((item["score"] for item in raw_hits), default=1.0) or 1.0
        selected: list[dict[str, Any]] = []
        for item in raw_hits:
            if _overlaps_existing(item, selected):
                continue
            item["score"] = round(item["score"] / max_score, 8)
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected
