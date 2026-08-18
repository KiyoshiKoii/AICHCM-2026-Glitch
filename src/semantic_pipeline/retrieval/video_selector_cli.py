"""Bootstrap and query the Elasticsearch video-first selector indexes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_pipeline.retrieval.elasticsearch_backend import (  # noqa: E402
    DEFAULT_ELASTICSEARCH_URL,
    create_client,
)
from semantic_pipeline.retrieval.hierarchical_index_definition import (  # noqa: E402
    SEGMENT_INDEX_NAME,
    VIDEO_INDEX_NAME,
)
from semantic_pipeline.retrieval.temporal_query_expander import GeminiTemporalQueryParser  # noqa: E402
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query  # noqa: E402
from semantic_pipeline.retrieval.temporal_video_selector import (  # noqa: E402
    ElasticsearchVideoSelector,
    bulk_ingest_selector,
    ensure_selector_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage video-first Elasticsearch indexes")
    parser.add_argument("--url", default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL))
    parser.add_argument("--video-index", default=VIDEO_INDEX_NAME)
    parser.add_argument("--segment-index", default=SEGMENT_INDEX_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create and ingest video/segment indexes")
    bootstrap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/video_understanding"),
    )
    bootstrap.add_argument("--batch-id", action="append", default=[])
    bootstrap.add_argument("--video-id", action="append", default=[])
    bootstrap.add_argument("--chunk-size", type=int, default=500)

    search = subparsers.add_parser("search", help="Select video IDs without resolving E1..En")
    search.add_argument("query")
    search.add_argument("--batch-id", action="append", default=[])
    search.add_argument("--video-id", action="append", default=[])
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--without-gemini-query-parser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        client = create_client(args.url)
        if args.command == "bootstrap":
            created = ensure_selector_indices(
                client,
                video_index=args.video_index,
                segment_index=args.segment_index,
            )
            result = bulk_ingest_selector(
                client,
                output_root=args.output_root,
                batch_ids=args.batch_id,
                video_ids=args.video_id,
                video_index=args.video_index,
                segment_index=args.segment_index,
                chunk_size=args.chunk_size,
            )
            print(json.dumps({"created": created, **result}, ensure_ascii=False, indent=2))
            return 0

        parsed = parse_temporal_query(args.query)
        if not args.without_gemini_query_parser:
            parsed = GeminiTemporalQueryParser().parse(parsed)
        result = ElasticsearchVideoSelector(
            client,
            video_index=args.video_index,
            segment_index=args.segment_index,
        ).select(
            parsed,
            batch_ids=args.batch_id,
            video_ids=args.video_id,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"video selector failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
