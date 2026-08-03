"""Elasticsearch storage/search backend for Task 4.

The module keeps Elasticsearch optional: unit tests and the BM25 service can be
used without installing the client.  A real connection is only created when
``ElasticsearchBackend`` or the CLI is used.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import streaming_bulk
except ImportError:  # BM25-only development remains supported.
    Elasticsearch = None  # type: ignore[assignment]
    streaming_bulk = None

try:
    from code_classifier import classify_record
    from schemas import DEFAULT_METADATA_PATH, FrameMetadata, validate_metadata_file
except ImportError:
    from .code_classifier import classify_record
    from .schemas import DEFAULT_METADATA_PATH, FrameMetadata, validate_metadata_file

SEMANTIC_DIR = Path(__file__).resolve().parent
DEFAULT_DEFINITION_PATH = SEMANTIC_DIR / "elasticsearch_index.json"
DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
DEFAULT_INDEX_NAME = "semantic_frames_v5"
DEFAULT_ALIAS_NAME = "semantic_frames"
MUTATING_COMMANDS = {"setup", "activate", "ingest", "bootstrap"}

SEARCH_FIELDS = [
    "code.search_terms^4.0",
    "code.patterns^3.0",
    "ocr_text^2.0",
    "ocr_text.stemmed^1.5",
    "caption",
]
FILTER_FIELDS = {
    "time_of_day": "entities.time_of_day",
    "setting": "entities.setting",
    "locations": "entities.locations",
    "objects": "entities.objects",
    "actions": "entities.actions",
    "colors": "entities.colors",
    "code_language": "code.language",
    "code_patterns": "code.patterns.keyword",
}
SPATIAL_PREDICATES = {
    "left_of",
    "right_of",
    "above",
    "below",
    "overlapping",
}
SPATIAL_INDEX_POLICY = "label-triple-max-confidence-v1"


def require_elasticsearch_client() -> None:
    if Elasticsearch is None:
        raise RuntimeError(
            "Missing Python package 'elasticsearch'. Install project dependencies "
            "or run: python -m pip install elasticsearch==9.4.1"
        )


def create_client(
    url: str | None = None,
    *,
    username: str | None = None,
    password: str | None = None,
    ca_cert: str | Path | None = None,
):
    """Create an official Elasticsearch client from explicit args or env vars.

    Secure deployments use ``ELASTICSEARCH_USERNAME``,
    ``ELASTICSEARCH_PASSWORD`` and ``ELASTICSEARCH_CA_CERT``.  Credentials are
    deliberately rejected for plaintext HTTP, and TLS verification is never
    disabled.  The no-auth HTTP defaults remain backward-compatible with the
    existing loopback-only development Compose file.
    """
    require_elasticsearch_client()
    resolved_url = url or os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL)
    resolved_username = (
        username if username is not None else os.getenv("ELASTICSEARCH_USERNAME")
    )
    resolved_password = (
        password if password is not None else os.getenv("ELASTICSEARCH_PASSWORD")
    )
    resolved_ca_cert = (
        ca_cert if ca_cert is not None else os.getenv("ELASTICSEARCH_CA_CERT")
    )

    if bool(resolved_username) != bool(resolved_password):
        raise ValueError(
            "ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD must be set together"
        )

    uses_https = resolved_url.lower().startswith("https://")
    if (resolved_username or resolved_password) and not uses_https:
        raise ValueError("Elasticsearch credentials may only be sent over HTTPS")

    client_options: dict[str, Any] = {"request_timeout": 30}
    if resolved_ca_cert:
        if not uses_https:
            raise ValueError("ELASTICSEARCH_CA_CERT requires an HTTPS URL")
        ca_path = Path(resolved_ca_cert).expanduser()
        if not ca_path.is_file():
            raise ValueError(f"Elasticsearch CA certificate does not exist: {ca_path}")
        client_options["ca_certs"] = str(ca_path.resolve())
    if resolved_username and resolved_password:
        client_options["basic_auth"] = (resolved_username, resolved_password)

    return Elasticsearch(resolved_url, **client_options)


def load_index_definition(
    definition_path: str | Path = DEFAULT_DEFINITION_PATH,
) -> dict:
    definition = json.loads(Path(definition_path).read_text(encoding="utf-8"))
    if set(definition) != {"settings", "mappings"}:
        raise ValueError("index definition must contain exactly settings and mappings")
    if definition["mappings"].get("dynamic") != "strict":
        raise ValueError("index mapping must use dynamic='strict'")
    return definition


def ensure_index(
    client,
    index_name: str = DEFAULT_INDEX_NAME,
    definition_path: str | Path = DEFAULT_DEFINITION_PATH,
) -> bool:
    """Create a versioned physical index if absent; never delete existing data."""
    if bool(client.indices.exists(index=index_name)):
        return False
    definition = load_index_definition(definition_path)
    client.indices.create(index=index_name, **definition)
    return True


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or getattr(
        getattr(exc, "meta", None), "status", None
    ) == 404


def activate_alias(
    client,
    index_name: str = DEFAULT_INDEX_NAME,
    alias_name: str = DEFAULT_ALIAS_NAME,
) -> None:
    """Atomically point the stable search alias at one versioned index."""
    try:
        current = client.indices.get_alias(name=alias_name)
    except Exception as exc:
        if not _is_not_found(exc):
            raise
        current = {}

    actions = [
        {"remove": {"index": old_index, "alias": alias_name}}
        for old_index in current.keys()
        if old_index != index_name
    ]
    # Re-adding the same alias is harmless and ensures is_write_index is set.
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


def collapse_spatial_relations_for_index(relations: list[Any]) -> list[Any]:
    """Keep one representative instance for each query-equivalent label triple.

    Elasticsearch spatial filters query labels and predicate, not instance IDs.
    Repeated same-label objects therefore add quadratic nested documents without
    changing search results.  Raw instance relations remain in metadata files for
    geometry evaluation; only the indexed projection is collapsed here.
    """
    best: dict[tuple[str, str, str], Any] = {}
    for relation in relations:
        subject = relation.subject_label or f"@{relation.subject_id}"
        object_ = relation.object_label or f"@{relation.object_id}"
        predicate = getattr(relation.predicate, "value", relation.predicate)
        key = (subject.casefold(), str(predicate), object_.casefold())
        incumbent = best.get(key)
        if incumbent is None or (
            relation.confidence,
            relation.subject_id,
            relation.object_id,
        ) > (
            incumbent.confidence,
            incumbent.subject_id,
            incumbent.object_id,
        ):
            best[key] = relation
    return [best[key] for key in sorted(best)]


def iter_bulk_actions(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Iterator[dict]:
    """Stream validated documents without loading a second raw corpus copy."""
    for record in validate_metadata_file(metadata_path):
        # Real validated records are classified at the ingestion boundary too,
        # so bootstrapping directly from legacy metadata cannot silently index
        # every code.language as unknown. Lightweight test doubles are preserved.
        if isinstance(record, FrameMetadata):
            record = classify_record(record)
        indexed_relations = collapse_spatial_relations_for_index(
            record.spatial_relations
        )
        source = record.model_dump(mode="json")
        source["spatial_relations"] = [
            relation.model_dump(mode="json") for relation in indexed_relations
        ]
        source["processing"].update(
            {
                "spatial_index_policy": SPATIAL_INDEX_POLICY,
                "spatial_relations_raw_count": len(record.spatial_relations),
                "spatial_relations_indexed_count": len(indexed_relations),
            }
        )
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": record.frame_id,
            "_source": source,
        }


def bulk_ingest(
    client,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    index_name: str = DEFAULT_INDEX_NAME,
    chunk_size: int = 500,
    streaming_bulk_fn: Callable[..., Iterator[tuple[bool, dict]]] | None = None,
) -> dict:
    """Idempotently index metadata using frame_id as Elasticsearch _id."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    helper = streaming_bulk_fn or streaming_bulk
    if helper is None:
        require_elasticsearch_client()
        raise AssertionError("unreachable")

    indexed = 0
    failures: list[dict] = []
    actions = iter_bulk_actions(metadata_path, index_name)
    for ok, item in helper(
        client,
        actions,
        chunk_size=chunk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            indexed += 1
        elif len(failures) < 10:
            failures.append(item)

    if failures:
        raise RuntimeError(
            f"Elasticsearch bulk ingest failed; first errors: {failures}"
        )
    client.indices.refresh(index=index_name)
    count = int(client.count(index=index_name)["count"])
    return {"indexed": indexed, "documents_in_index": count, "failures": 0}


def build_search_query(keywords: list[str], filters: dict | None = None) -> dict:
    cleaned_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
    text = " ".join(cleaned_keywords)
    if not text:
        return {"match_none": {}}

    filter_clauses = []
    for name, value in (filters or {}).items():
        if name == "spatial_relations":
            if not isinstance(value, list) or not value:
                raise ValueError("Filter 'spatial_relations' must be a non-empty list")
            for relation in value:
                if not isinstance(relation, dict) or set(relation) != {
                    "subject",
                    "predicate",
                    "object",
                }:
                    raise ValueError(
                        "Each spatial relation filter must contain exactly "
                        "subject, predicate, and object"
                    )
                subject = relation["subject"]
                predicate = relation["predicate"]
                object_ = relation["object"]
                if not all(
                    isinstance(item, str) and item.strip()
                    for item in (subject, predicate, object_)
                ) or predicate not in SPATIAL_PREDICATES:
                    raise ValueError("Invalid spatial relation filter")
                filter_clauses.append(
                    {
                        "nested": {
                            "path": "spatial_relations",
                            "query": {
                                "bool": {
                                    "filter": [
                                        {
                                            "term": {
                                                "spatial_relations.subject_label": subject.strip().casefold()
                                            }
                                        },
                                        {
                                            "term": {
                                                "spatial_relations.predicate": predicate
                                            }
                                        },
                                        {
                                            "term": {
                                                "spatial_relations.object_label": object_.strip().casefold()
                                            }
                                        },
                                    ]
                                }
                            },
                        }
                    }
                )
            continue
        if name not in FILTER_FIELDS:
            raise ValueError(
                f"Unsupported filter {name!r}; allowed: {sorted(FILTER_FIELDS)}"
            )
        field = FILTER_FIELDS[name]
        if isinstance(value, str):
            if value.strip():
                filter_clauses.append({"term": {field: value}})
        elif isinstance(value, list) and value and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            # A KIS request listing several constraints means all-of. One
            # `terms` query would be OR/any-of and silently admit partial matches.
            filter_clauses.extend(
                {"term": {field: item.strip().casefold()}} for item in value
            )
        else:
            raise ValueError(f"Filter {name!r} must be a string or non-empty string list")

    # Dev 3 deliberately sends keyword phrases (for example "three layers").
    # Preserve that information as optional phrase boosts instead of flattening
    # everything into a bag of words. The broad must-clause still protects recall.
    phrase_boosts = [
        {
            "multi_match": {
                "query": keyword,
                "fields": [
                    "code.search_terms^5.0",
                    "code.patterns^4.0",
                    "ocr_text^3.0",
                    "caption^2.0",
                ],
                "type": "phrase",
                "boost": 2.0,
            }
        }
        for keyword in cleaned_keywords
        if len(keyword.split()) > 1
    ]

    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": text,
                        "fields": SEARCH_FIELDS,
                        "type": "most_fields",
                    }
                }
            ],
            "filter": filter_clauses,
            "should": phrase_boosts,
        }
    }


class ElasticsearchBackend:
    """Search adapter returning the same four fields as the existing BM25 API."""

    supports_filters = True

    def __init__(
        self,
        client=None,
        url: str | None = None,
        index: str | None = None,
    ):
        self.url = url or os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL)
        self.index = index or os.getenv("ELASTICSEARCH_INDEX", DEFAULT_ALIAS_NAME)
        self.client = client or create_client(self.url)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def document_count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def search(
        self,
        keywords: list[str],
        top_k: int = 200,
        filters: dict | None = None,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = build_search_query(keywords, filters)
        if "match_none" in query:
            return []

        response = self.client.search(
            index=self.index,
            size=top_k,
            query=query,
            source=["frame_id", "video_name", "frame_index"],
        )
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "frame_id": source["frame_id"],
                    "score": float(hit.get("_score") or 0.0),
                    "video_name": source["video_name"],
                    "frame_index": int(source["frame_index"]),
                }
            )
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local semantic Elasticsearch index.")
    parser.add_argument(
        "--url",
        default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL),
    )
    parser.add_argument(
        "--index-name",
        help=(
            "Versioned physical index. Required for setup/activate/ingest/bootstrap "
            "to prevent accidental alias rollback."
        ),
    )
    parser.add_argument("--alias", default=DEFAULT_ALIAS_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check connection and document count.")
    subparsers.add_parser("setup", help="Create the versioned physical index.")
    subparsers.add_parser("activate", help="Atomically move the stable alias.")

    ingest = subparsers.add_parser("ingest", help="Validate and bulk-index metadata.")
    ingest.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    ingest.add_argument("--chunk-size", type=int, default=500)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Create, ingest, then activate the search alias."
    )
    bootstrap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    bootstrap.add_argument("--chunk-size", type=int, default=500)

    search = subparsers.add_parser("search", help="Run one search through the alias.")
    search.add_argument("keywords", nargs="+")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--time-of-day")
    search.add_argument("--setting")
    search.add_argument("--location", dest="locations", action="append")
    search.add_argument("--object", dest="objects", action="append")
    search.add_argument("--action", dest="actions", action="append")
    search.add_argument("--color", dest="colors", action="append")
    search.add_argument("--code-language", choices=("unknown", "sql"))
    search.add_argument("--code-pattern", dest="code_patterns", action="append")
    search.add_argument(
        "--spatial",
        nargs=3,
        action="append",
        metavar=("SUBJECT", "PREDICATE", "OBJECT"),
        help="Exact spatial triple, e.g. --spatial person left_of car",
    )

    analyze = subparsers.add_parser("analyze", help="Inspect English analyzer tokens.")
    analyze.add_argument("text")
    analyze.add_argument("--analyzer", default="kis_english")
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.command in MUTATING_COMMANDS and not args.index_name:
        raise ValueError(
            f"--index-name is required for {args.command}; use an explicit "
            "version such as semantic_frames_v5"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_cli_args(args)
        client = create_client(args.url)
        if args.command == "health":
            if not client.ping():
                raise RuntimeError(f"Elasticsearch is not reachable at {args.url}")
            count = int(client.count(index=args.alias)["count"])
            print(f"Elasticsearch healthy: url={args.url}, alias={args.alias}, documents={count}")
        elif args.command == "setup":
            created = ensure_index(client, args.index_name)
            print(f"Index {args.index_name}: {'created' if created else 'already exists'}")
        elif args.command == "ingest":
            ensure_index(client, args.index_name)
            result = bulk_ingest(
                client, args.metadata, args.index_name, chunk_size=args.chunk_size
            )
            print(json.dumps(result, indent=2))
        elif args.command == "activate":
            activate_alias(client, args.index_name, args.alias)
            print(f"Alias {args.alias} now points to {args.index_name}")
        elif args.command == "bootstrap":
            created = ensure_index(client, args.index_name)
            result = bulk_ingest(
                client, args.metadata, args.index_name, chunk_size=args.chunk_size
            )
            activate_alias(client, args.index_name, args.alias)
            print(
                json.dumps(
                    {"created": created, "alias": args.alias, **result}, indent=2
                )
            )
        elif args.command == "search":
            backend = ElasticsearchBackend(client=client, index=args.alias)
            filters = {
                name: value
                for name, value in {
                    "time_of_day": args.time_of_day,
                    "setting": args.setting,
                    "locations": args.locations,
                    "objects": args.objects,
                    "actions": args.actions,
                    "colors": args.colors,
                    "code_language": args.code_language,
                    "code_patterns": args.code_patterns,
                    "spatial_relations": [
                        {"subject": item[0], "predicate": item[1], "object": item[2]}
                        for item in (args.spatial or [])
                    ]
                    or None,
                }.items()
                if value is not None
            }
            print(
                json.dumps(
                    backend.search(
                        args.keywords,
                        top_k=args.top_k,
                        filters=filters,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "analyze":
            response = client.indices.analyze(
                index=args.index_name or args.alias,
                analyzer=args.analyzer,
                text=args.text,
            )
            print([token["token"] for token in response["tokens"]])
    except Exception as exc:
        parser.exit(status=1, message=f"Elasticsearch command failed: {exc}\n")


if __name__ == "__main__":
    main()
