"""Run the Step 0 labelled-query benchmark against Elasticsearch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from baseline import (
        DEFAULT_QUERY_PATH,
        DEFAULT_TOP_KS,
        evaluate_backend,
        load_query_cases,
        parse_top_ks,
    )
    from elasticsearch_backend import (
        DEFAULT_ALIAS_NAME,
        DEFAULT_ELASTICSEARCH_URL,
        ElasticsearchBackend,
    )
except ImportError:
    from .baseline import (
        DEFAULT_QUERY_PATH,
        DEFAULT_TOP_KS,
        evaluate_backend,
        load_query_cases,
        parse_top_ks,
    )
    from .elasticsearch_backend import (
        DEFAULT_ALIAS_NAME,
        DEFAULT_ELASTICSEARCH_URL,
        ElasticsearchBackend,
    )


def print_summary(report: dict) -> None:
    print(
        f"Backend=elasticsearch | index={report['index']} | "
        f"documents={report['corpus_documents']} | queries={report['query_count']}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Elasticsearch with the same labels as BM25."
    )
    parser.add_argument("--url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--index", default=DEFAULT_ALIAS_NAME)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--top-k", type=parse_top_ks, default=DEFAULT_TOP_KS)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    backend = ElasticsearchBackend(url=args.url, index=args.index)
    if not backend.ping():
        parser.exit(status=1, message=f"Elasticsearch is not reachable at {args.url}\n")
    cases = load_query_cases(args.queries)
    report = evaluate_backend(
        backend, cases, top_ks=args.top_k, latency_runs=args.latency_runs
    )
    report.update(
        {
            "backend": "elasticsearch",
            "url": args.url,
            "index": args.index,
            "corpus_documents": backend.document_count(),
            "query_path": str(args.queries.resolve()),
        }
    )
    print_summary(report)
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
