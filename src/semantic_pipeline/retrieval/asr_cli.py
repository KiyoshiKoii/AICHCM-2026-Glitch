"""Create, ingest and query the timestamped ASR Elasticsearch index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from semantic_pipeline.retrieval.asr_search import (
    ASR_INDEX_NAME,
    DEFAULT_ASR_DIR,
    ElasticsearchASRSearch,
    bulk_ingest_asr,
    ensure_asr_index,
)
from semantic_pipeline.retrieval.elasticsearch_backend import (
    DEFAULT_ELASTICSEARCH_URL,
    create_client,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage timestamped ASR search in Elasticsearch")
    parser.add_argument("--url", default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL))
    parser.add_argument("--index", default=ASR_INDEX_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create and ingest the ASR index")
    bootstrap.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_DIR)
    bootstrap.add_argument("--batch-id", action="append", default=[])
    bootstrap.add_argument("--video-id", action="append", default=[])
    bootstrap.add_argument("--chunk-size", type=int, default=1_000)

    search = subparsers.add_parser("search", help="Search timestamped ASR passages")
    search.add_argument("query")
    search.add_argument("--batch-id", action="append", default=[])
    search.add_argument("--video-id", action="append", default=[])
    search.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = create_client(args.url)
        if args.command == "bootstrap":
            created = ensure_asr_index(client, args.index)
            result = bulk_ingest_asr(
                client,
                asr_dir=args.asr_dir,
                index_name=args.index,
                batch_ids=args.batch_id,
                video_ids=args.video_id,
                chunk_size=args.chunk_size,
            )
            print(json.dumps({"created": created, **result}, ensure_ascii=False, indent=2))
            return 0
        results = ElasticsearchASRSearch(client, index=args.index).search(
            args.query,
            batch_ids=args.batch_id,
            video_ids=args.video_id,
            top_k=args.top_k,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ASR search failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
