"""Safe command-line management of the versioned Elasticsearch index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from semantic_pipeline.retrieval.elasticsearch_backend import (
    DEFAULT_CAPTION_DIR,
    DEFAULT_ELASTICSEARCH_URL,
    DEFAULT_YOUTUBE_METADATA,
    ElasticsearchTextSearch,
    activate_alias,
    bulk_ingest,
    create_client,
    ensure_index,
)
from semantic_pipeline.retrieval.index_definition import DEFAULT_ALIAS_NAME, DEFAULT_INDEX_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Gemini metadata search in Elasticsearch.")
    parser.add_argument("--url", default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL))
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="Check Elasticsearch and the active alias.")
    subparsers.add_parser("setup", help="Create the physical index if absent.")
    subparsers.add_parser("activate", help="Atomically point the alias at --index-name.")

    for name, help_text in (("ingest", "Bulk-index the current caption artifacts."), ("bootstrap", "Create, ingest, then activate.")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTION_DIR)
        command.add_argument("--youtube-metadata", type=Path, default=DEFAULT_YOUTUBE_METADATA)
        command.add_argument("--chunk-size", type=int, default=500)
        command.add_argument(
            "--require-provenance",
            action="store_true",
            help="Reject legacy caption records missing visual/OCR source IDs.",
        )

    search = subparsers.add_parser("search", help="Run one lexical search through the alias.")
    search.add_argument("keywords", nargs="+")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument(
        "--collapse-visual-duplicates",
        action="store_true",
        help="Collapse v4 results by visual_source_frame_id.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        client = create_client(args.url)
        if args.command == "health":
            backend = ElasticsearchTextSearch(client, index=args.alias)
            if not backend.ping():
                raise RuntimeError(f"Elasticsearch is not reachable at {args.url}")
            print(json.dumps({"status": "ok", "documents": backend.document_count()}, indent=2))
        elif args.command == "setup":
            created = ensure_index(client, args.index_name)
            print(f"Index {args.index_name}: {'created' if created else 'already exists'}")
        elif args.command == "activate":
            activate_alias(client, index_name=args.index_name, alias_name=args.alias)
            print(f"Alias {args.alias} now points to {args.index_name}")
        elif args.command in {"ingest", "bootstrap"}:
            created = ensure_index(client, args.index_name)
            result = bulk_ingest(
                client,
                caption_dir=args.caption_dir,
                youtube_metadata=args.youtube_metadata,
                index_name=args.index_name,
                chunk_size=args.chunk_size,
                require_provenance=args.require_provenance,
            )
            if args.command == "bootstrap":
                activate_alias(client, index_name=args.index_name, alias_name=args.alias)
            print(json.dumps({"created": created, "alias": args.alias if args.command == "bootstrap" else None, **result}, indent=2))
        elif args.command == "search":
            backend = ElasticsearchTextSearch(client, index=args.alias)
            print(
                json.dumps(
                    backend.search(
                        args.keywords,
                        top_k=args.top_k,
                        collapse_visual_duplicates=args.collapse_visual_duplicates,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except Exception as exc:
        parser.exit(status=1, message=f"Elasticsearch command failed: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
