"""Baseline retrieval evaluation for Task 4, Step 0.

The query set and metrics in this module are backend-agnostic.  Today they
measure the in-memory BM25 implementation; the same evaluator can later be
used for Elasticsearch so both backends are compared on identical labels.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

SEMANTIC_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_PATH = SEMANTIC_DIR / "evaluation_queries.json"
DEFAULT_METADATA_PATH = SEMANTIC_DIR / "sample_frames" / "metadata.json"
DEFAULT_TOP_KS = (1, 5, 10, 200)


class SearchBackend(Protocol):
    """Small common surface shared by BM25 and the future ES backend."""

    records: list[dict]

    supports_filters: bool

    def search(
        self,
        keywords: list[str],
        top_k: int = 200,
        filters: dict | None = None,
    ) -> list[dict]: ...


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    keywords: tuple[str, ...]
    relevant_frame_ids: frozenset[str]
    tags: tuple[str, ...] = ()
    filters: dict | None = None


def load_query_cases(
    query_path: str | Path = DEFAULT_QUERY_PATH,
    valid_frame_ids: set[str] | None = None,
) -> list[QueryCase]:
    """Load and strictly validate the hand-labelled evaluation set."""
    path = Path(query_path)
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    cases: list[QueryCase] = []
    seen_ids: set[str] = set()
    required = {"query_id", "query", "keywords", "relevant_frame_ids"}

    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"Query at position {index} must be a JSON object")
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Query at position {index} is missing {sorted(missing)}")

        query_id = raw["query_id"]
        keywords = raw["keywords"]
        relevant = raw["relevant_frame_ids"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Query at position {index} has an invalid query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        if not isinstance(raw["query"], str) or not raw["query"].strip():
            raise ValueError(f"{query_id}: query must be a non-empty string")
        if not isinstance(keywords, list) or not keywords or not all(
            isinstance(value, str) and value.strip() for value in keywords
        ):
            raise ValueError(f"{query_id}: keywords must be a non-empty string array")
        if not isinstance(relevant, list) or not relevant or not all(
            isinstance(value, str) and value.strip() for value in relevant
        ):
            raise ValueError(
                f"{query_id}: relevant_frame_ids must be a non-empty string array"
            )

        relevant_set = frozenset(relevant)
        if len(relevant_set) != len(relevant):
            raise ValueError(f"{query_id}: relevant_frame_ids contains duplicates")
        if valid_frame_ids is not None:
            unknown = relevant_set - valid_frame_ids
            if unknown:
                raise ValueError(
                    f"{query_id}: ground truth frames are absent from metadata: "
                    f"{sorted(unknown)}"
                )

        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{query_id}: tags must be a string array")
        filters = raw.get("filters")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError(f"{query_id}: filters must be an object when provided")

        cases.append(
            QueryCase(
                query_id=query_id,
                query=raw["query"],
                keywords=tuple(keywords),
                relevant_frame_ids=relevant_set,
                tags=tuple(tags),
                filters=filters,
            )
        )
        seen_ids.add(query_id)
    return cases


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(
    ranked_ids: Sequence[str], relevant_ids: set[str], k: int
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    for rank, frame_id in enumerate(ranked_ids[:k], start=1):
        if frame_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG, sufficient for the hand-labelled KIS baseline."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, frame_id in enumerate(ranked_ids[:k], start=1)
        if frame_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def percentile(values: Sequence[float], percent: float) -> float:
    """Nearest-rank percentile, deterministic for small benchmark samples."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def evaluate_backend(
    backend: SearchBackend,
    cases: Sequence[QueryCase],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    latency_runs: int = 20,
) -> dict:
    """Evaluate quality once and latency repeatedly for every query."""
    if not cases:
        raise ValueError("cases must not be empty")
    if latency_runs < 1:
        raise ValueError("latency_runs must be at least 1")
    normalised_top_ks = tuple(sorted(set(top_ks)))
    if not normalised_top_ks or normalised_top_ks[0] < 1:
        raise ValueError("top_ks must contain positive integers")

    max_k = normalised_top_ks[-1]
    quality_rows = []
    latencies_ms: list[float] = []
    supports_filters = bool(getattr(backend, "supports_filters", False))
    filtered_query_count = sum(case.filters is not None for case in cases)

    def search_case(case: QueryCase) -> list[dict]:
        if case.filters is not None and supports_filters:
            return backend.search(
                list(case.keywords), top_k=max_k, filters=case.filters
            )
        return backend.search(list(case.keywords), top_k=max_k)

    for case in cases:
        results = search_case(case)
        ranked_ids = [result["frame_id"] for result in results]
        row = {
            "query_id": case.query_id,
            "relevant_frame_ids": sorted(case.relevant_frame_ids),
            "retrieved_frame_ids": ranked_ids,
            "first_relevant_rank": next(
                (
                    rank
                    for rank, frame_id in enumerate(ranked_ids, start=1)
                    if frame_id in case.relevant_frame_ids
                ),
                None,
            ),
            "filters": case.filters,
            "filters_applied": case.filters is not None and supports_filters,
            "metrics": {},
        }
        for k in normalised_top_ks:
            row["metrics"][str(k)] = {
                "recall": recall_at_k(ranked_ids, set(case.relevant_frame_ids), k),
                "reciprocal_rank": reciprocal_rank_at_k(
                    ranked_ids, set(case.relevant_frame_ids), k
                ),
                "ndcg": ndcg_at_k(ranked_ids, set(case.relevant_frame_ids), k),
            }
        quality_rows.append(row)

        # Warm-up each distinct query before recording latency.
        search_case(case)
        for _ in range(latency_runs):
            started = time.perf_counter()
            search_case(case)
            latencies_ms.append((time.perf_counter() - started) * 1000)

    aggregate = {}
    for k in normalised_top_ks:
        key = str(k)
        aggregate[key] = {
            "recall": statistics.fmean(row["metrics"][key]["recall"] for row in quality_rows),
            "mrr": statistics.fmean(
                row["metrics"][key]["reciprocal_rank"] for row in quality_rows
            ),
            "ndcg": statistics.fmean(row["metrics"][key]["ndcg"] for row in quality_rows),
        }

    return {
        "query_count": len(cases),
        "filtered_query_count": filtered_query_count,
        "filters_applied_count": filtered_query_count if supports_filters else 0,
        "top_ks": list(normalised_top_ks),
        "quality": aggregate,
        "latency_ms": {
            "samples": len(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "queries": quality_rows,
    }


def parse_top_ks(value: str) -> tuple[int, ...]:
    try:
        top_ks = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k values must be integers") from exc
    if not top_ks or top_ks[0] < 1:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return top_ks


def run_benchmark(
    metadata_path: str | Path,
    query_path: str | Path,
    top_ks: Sequence[int],
    latency_runs: int,
) -> dict:
    # Import lazily so metric/schema unit tests do not require rank_bm25.
    try:
        from database import TextDatabase
    except ImportError:
        from .database import TextDatabase

    metadata_path = Path(metadata_path)
    metadata_size_mb = metadata_path.stat().st_size / (1024 * 1024)

    tracemalloc.start()
    started = time.perf_counter()
    backend = TextDatabase(metadata_path)
    index_build_ms = (time.perf_counter() - started) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    valid_frame_ids = {record["frame_id"] for record in backend.records}
    cases = load_query_cases(query_path, valid_frame_ids=valid_frame_ids)
    report = evaluate_backend(backend, cases, top_ks=top_ks, latency_runs=latency_runs)
    report.update(
        {
            "backend": "rank_bm25",
            "corpus_documents": len(backend.records),
            "metadata_path": str(metadata_path.resolve()),
            "query_path": str(Path(query_path).resolve()),
            "metadata_size_mb": metadata_size_mb,
            "index_build_ms": index_build_ms,
            # Python allocations only; this is a comparable baseline, not total RSS.
            "index_build_python_peak_mb": peak_bytes / (1024 * 1024),
        }
    )
    return report


def print_summary(report: dict) -> None:
    print(
        f"Backend={report['backend']} | documents={report['corpus_documents']} "
        f"| queries={report['query_count']}"
    )
    print(
        f"Index build={report['index_build_ms']:.3f} ms | "
        f"Python peak={report['index_build_python_peak_mb']:.3f} MB | "
        f"metadata={report['metadata_size_mb']:.3f} MB"
    )
    print("\nQuality")
    print(f"{'K':>5} {'Recall':>10} {'MRR':>10} {'NDCG':>10}")
    for k in report["top_ks"]:
        metrics = report["quality"][str(k)]
        print(
            f"{k:>5} {metrics['recall']:>10.4f} "
            f"{metrics['mrr']:>10.4f} {metrics['ndcg']:>10.4f}"
        )
    latency = report["latency_ms"]
    print(
        "\nLatency "
        f"mean={latency['mean']:.3f} ms | p50={latency['p50']:.3f} ms | "
        f"p95={latency['p95']:.3f} ms | samples={latency['samples']}"
    )

    missed = [row for row in report["queries"] if row["first_relevant_rank"] is None]
    if missed:
        print("\nQueries with no relevant frame retrieved:")
        for row in missed:
            print(f"- {row['query_id']}: expected {row['relevant_frame_ids']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the Task 4 Step 0 retrieval baseline."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--top-k", type=parse_top_ks, default=DEFAULT_TOP_KS)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = run_benchmark(
        metadata_path=args.metadata,
        query_path=args.queries,
        top_ks=args.top_k,
        latency_runs=args.latency_runs,
    )
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
