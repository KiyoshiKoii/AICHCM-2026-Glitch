"""Opt-in Elasticsearch scale benchmark for the Dev 2 semantic pipeline.

The 24 hand-labelled frames remain the retrieval-quality benchmark.  This
module answers a different question: can the current mapping, bulk pipeline,
and query path handle a larger document count without changing the production
alias?  It deterministically replicates validated sample records into an
explicit physical index and never deletes or activates an index.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

try:
    from baseline import DEFAULT_QUERY_PATH, QueryCase, load_query_cases, percentile
    from elasticsearch_backend import (
        DEFAULT_ELASTICSEARCH_URL,
        ElasticsearchBackend,
        create_client,
        ensure_index,
        streaming_bulk,
    )
    from schemas import FrameMetadata, validate_metadata_file
except ImportError:
    from .baseline import DEFAULT_QUERY_PATH, QueryCase, load_query_cases, percentile
    from .elasticsearch_backend import (
        DEFAULT_ELASTICSEARCH_URL,
        ElasticsearchBackend,
        create_client,
        ensure_index,
        streaming_bulk,
    )
    from .schemas import FrameMetadata, validate_metadata_file


SEMANTIC_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = SEMANTIC_DIR / "sample_frames" / "metadata_spatial.json"
DEFAULT_DOCUMENTS = 10_000
DEFAULT_MAX_P95_MS = 50.0


def iter_scaled_actions(
    records: Sequence[FrameMetadata],
    index_name: str,
    document_count: int,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic, schema-compatible copies without building a giant list."""
    if not records:
        raise ValueError("records must not be empty")
    if document_count < 1:
        raise ValueError("document_count must be at least 1")
    if not index_name.strip():
        raise ValueError("index_name must not be empty")

    for ordinal in range(document_count):
        template = records[ordinal % len(records)]
        source = template.model_dump(mode="json")
        video_stem = f"scale_{ordinal:08d}"
        frame_index = int(template.frame_index)
        source.update(
            {
                "frame_id": f"{video_stem}_f{frame_index:04d}",
                "video_name": f"{video_stem}.mp4",
            }
        )
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": source["frame_id"],
            "_source": source,
        }


def ingest_scaled_corpus(
    client: Any,
    records: Sequence[FrameMetadata],
    index_name: str,
    document_count: int,
    chunk_size: int = 500,
    streaming_bulk_fn: Callable[..., Iterator[tuple[bool, dict]]] | None = None,
) -> dict[str, Any]:
    """Create a new explicit index and stream documents into it.

    Existing non-empty indexes are refused.  This avoids silently mixing two
    experiments and avoids needing a destructive delete/recreate operation.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    created = ensure_index(client, index_name=index_name)
    existing_count = int(client.count(index=index_name)["count"])
    if not created and existing_count:
        raise ValueError(
            f"scale index {index_name!r} already contains {existing_count} documents; "
            "choose a new --index-name or use --skip-ingest to benchmark it"
        )

    helper = streaming_bulk_fn or streaming_bulk
    if helper is None:
        raise RuntimeError("the elasticsearch Python client is required")

    failures: list[dict] = []
    indexed = 0
    started = time.perf_counter()
    for ok, item in helper(
        client,
        iter_scaled_actions(records, index_name, document_count),
        chunk_size=chunk_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            indexed += 1
        elif len(failures) < 10:
            failures.append(item)
    ingest_seconds = time.perf_counter() - started
    if failures:
        raise RuntimeError(f"scale bulk ingest failed; first errors: {failures}")

    client.indices.refresh(index=index_name)
    actual_count = int(client.count(index=index_name)["count"])
    if actual_count != document_count:
        raise RuntimeError(
            f"expected {document_count} documents after ingest, found {actual_count}"
        )
    return {
        "created": created,
        "indexed_operations": indexed,
        "documents": actual_count,
        "seconds": ingest_seconds,
        "documents_per_second": actual_count / ingest_seconds,
    }


def _search_case(backend: ElasticsearchBackend, case: QueryCase, top_k: int) -> list[dict]:
    if case.filters is not None:
        return backend.search(list(case.keywords), top_k=top_k, filters=case.filters)
    return backend.search(list(case.keywords), top_k=top_k)


def measure_search_latency(
    backend: ElasticsearchBackend,
    cases: Sequence[QueryCase],
    latency_runs: int,
    top_k: int,
    warmup_runs: int = 3,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("cases must not be empty")
    if latency_runs < 1:
        raise ValueError("latency_runs must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")

    latencies_ms: list[float] = []
    non_empty_queries = 0
    for case in cases:
        warmup = []
        for _ in range(warmup_runs):
            warmup = _search_case(backend, case, top_k)
        non_empty_queries += bool(warmup)
        for _ in range(latency_runs):
            started = time.perf_counter()
            _search_case(backend, case, top_k)
            latencies_ms.append((time.perf_counter() - started) * 1000)

    return {
        "query_count": len(cases),
        "non_empty_query_count": non_empty_queries,
        "warmup_runs_per_query": warmup_runs,
        "samples": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def _store_size_bytes(client: Any, index_name: str) -> int | None:
    try:
        stats = client.indices.stats(index=index_name, metric="store")
        return int(stats["_all"]["total"]["store"]["size_in_bytes"])
    except (KeyError, TypeError, ValueError):
        return None


def run_scale_benchmark(
    *,
    url: str,
    index_name: str,
    metadata_path: Path,
    query_path: Path,
    document_count: int,
    chunk_size: int,
    latency_runs: int,
    top_k: int,
    warmup_runs: int,
    max_p95_ms: float,
    skip_ingest: bool,
) -> dict[str, Any]:
    if document_count < 1:
        raise ValueError("document_count must be at least 1")
    if max_p95_ms <= 0:
        raise ValueError("max_p95_ms must be positive")

    client = create_client(url)
    if not client.ping():
        raise RuntimeError(f"Elasticsearch is not reachable at {url}")
    records = validate_metadata_file(metadata_path)

    ingest_report: dict[str, Any] | None = None
    if skip_ingest:
        actual_count = int(client.count(index=index_name)["count"])
        if actual_count < 1:
            raise ValueError(f"scale index {index_name!r} is empty")
    else:
        ingest_report = ingest_scaled_corpus(
            client,
            records,
            index_name,
            document_count,
            chunk_size=chunk_size,
        )
        actual_count = int(ingest_report["documents"])

    cases = load_query_cases(query_path)
    backend = ElasticsearchBackend(client=client, index=index_name)
    search_report = measure_search_latency(
        backend, cases, latency_runs, top_k, warmup_runs=warmup_runs
    )
    gate = {
        "minimum_documents": DEFAULT_DOCUMENTS,
        "max_p95_ms": max_p95_ms,
        "documents_passed": actual_count >= DEFAULT_DOCUMENTS,
        "latency_passed": search_report["p95_ms"] <= max_p95_ms,
    }
    gate["passed"] = gate["documents_passed"] and gate["latency_passed"]
    return {
        "benchmark_version": "elasticsearch-scale-v1",
        "url": url,
        "index": index_name,
        "source_templates": len(records),
        "requested_documents": document_count,
        "actual_documents": actual_count,
        "store_size_bytes": _store_size_bytes(client, index_name),
        "ingest": ingest_report,
        "search": search_report,
        "pilot_gate": gate,
        "notes": [
            "Synthetic replication measures infrastructure scale, not retrieval quality.",
            "The stable semantic_frames alias is never changed by this benchmark.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and benchmark an isolated synthetic Elasticsearch corpus."
    )
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Unmeasured steady-state warm-ups per distinct query.",
    )
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-p95-ms", type=float, default=DEFAULT_MAX_P95_MS)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Benchmark an existing explicit index without modifying it.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run_scale_benchmark(
            url=args.url,
            index_name=args.index_name,
            metadata_path=args.metadata,
            query_path=args.queries,
            document_count=args.documents,
            chunk_size=args.chunk_size,
            latency_runs=args.latency_runs,
            top_k=args.top_k,
            warmup_runs=args.warmup_runs,
            max_p95_ms=args.max_p95_ms,
            skip_ingest=args.skip_ingest,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Scale benchmark failed: {exc}\n")

    ingest = report["ingest"]
    print(
        f"Scale benchmark | index={report['index']} | "
        f"documents={report['actual_documents']:,} | "
        f"p95={report['search']['p95_ms']:.3f} ms | "
        f"gate_passed={report['pilot_gate']['passed']}"
    )
    if ingest is not None:
        print(
            f"Bulk ingest={ingest['documents_per_second']:.1f} docs/s "
            f"in {ingest['seconds']:.3f} s"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
