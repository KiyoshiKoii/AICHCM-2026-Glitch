"""Tests for Task 4 Step 0: labelled queries and retrieval metrics."""

import json
import math
from pathlib import Path

import pytest

from baseline import (
    QueryCase,
    evaluate_backend,
    load_query_cases,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)

SEMANTIC_DIR = Path(__file__).resolve().parent.parent
QUERY_PATH = SEMANTIC_DIR / "evaluation_queries.json"
METADATA_PATH = SEMANTIC_DIR / "sample_frames" / "metadata.json"


class FakeBackend:
    records = [{"frame_id": "a"}, {"frame_id": "b"}, {"frame_id": "c"}]

    def search(self, keywords, top_k=200):
        return [
            {"frame_id": frame_id, "score": float(3 - index)}
            for index, frame_id in enumerate(["a", "b", "c"][:top_k])
        ]


class TestMetrics:
    def test_recall_at_k(self):
        ranked = ["a", "b", "c"]
        assert recall_at_k(ranked, {"b", "c"}, 1) == 0.0
        assert recall_at_k(ranked, {"b", "c"}, 2) == 0.5
        assert recall_at_k(ranked, {"b", "c"}, 3) == 1.0

    def test_reciprocal_rank_at_k(self):
        ranked = ["a", "b", "c"]
        assert reciprocal_rank_at_k(ranked, {"b"}, 1) == 0.0
        assert reciprocal_rank_at_k(ranked, {"b"}, 3) == 0.5

    def test_ndcg_at_k(self):
        assert ndcg_at_k(["b", "a", "c"], {"b"}, 3) == 1.0
        assert ndcg_at_k(["a", "b", "c"], {"b"}, 3) == pytest.approx(
            1 / math.log2(3)
        )

    def test_percentile_uses_nearest_rank(self):
        assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
        assert percentile([4.0, 1.0, 3.0, 2.0], 95) == 4.0

    def test_metrics_reject_invalid_inputs(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), 1)
        with pytest.raises(ValueError):
            reciprocal_rank_at_k(["a"], {"a"}, 0)
        with pytest.raises(ValueError):
            percentile([], 95)


class TestEvaluationDataset:
    def test_all_ground_truth_frames_exist_in_metadata(self):
        records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        frame_ids = {record["frame_id"] for record in records}
        cases = load_query_cases(QUERY_PATH, valid_frame_ids=frame_ids)
        assert len(cases) >= 10
        assert all(case.relevant_frame_ids <= frame_ids for case in cases)

    def test_query_ids_are_unique(self):
        cases = load_query_cases(QUERY_PATH)
        ids = [case.query_id for case in cases]
        assert len(ids) == len(set(ids))

    def test_sql_template_case_is_exact_and_unfiltered(self):
        cases = load_query_cases(QUERY_PATH)
        case = next(item for item in cases if item.query_id == "sql_query_template")
        assert case.keywords == ("SQL query",)
        assert case.filters is None
        assert case.relevant_frame_ids == {"vid03_f0004"}


class TestEvaluator:
    def test_aggregate_metrics_and_latency_schema(self):
        cases = [
            QueryCase(
                query_id="find_b",
                query="find b",
                keywords=("b",),
                relevant_frame_ids=frozenset({"b"}),
            )
        ]
        report = evaluate_backend(
            FakeBackend(), cases, top_ks=(1, 3), latency_runs=2
        )

        assert report["query_count"] == 1
        assert report["quality"]["1"]["recall"] == 0.0
        assert report["quality"]["3"]["recall"] == 1.0
        assert report["quality"]["3"]["mrr"] == 0.5
        assert report["latency_ms"]["samples"] == 2
        assert report["queries"][0]["first_relevant_rank"] == 2

    def test_forwards_filters_only_to_capable_backends(self):
        class FilterBackend:
            records = [{"frame_id": "a"}]
            supports_filters = True

            def __init__(self):
                self.calls = []

            def search(self, keywords, top_k=200, filters=None):
                self.calls.append(filters)
                return [{"frame_id": "a", "score": 1.0}]

        backend = FilterBackend()
        cases = [
            QueryCase(
                query_id="filtered",
                query="blue outdoor flower",
                keywords=("flower",),
                relevant_frame_ids=frozenset({"a"}),
                filters={"setting": "outdoor", "colors": ["blue"]},
            )
        ]
        report = evaluate_backend(backend, cases, top_ks=(1,), latency_runs=1)

        assert backend.calls == [cases[0].filters] * 3
        assert report["filtered_query_count"] == 1
        assert report["filters_applied_count"] == 1
        assert report["queries"][0]["filters_applied"] is True

    def test_current_bm25_backend_produces_complete_report(self):
        # Integration test: exercises the real rank_bm25 backend and fixtures.
        from baseline import run_benchmark

        report = run_benchmark(
            metadata_path=METADATA_PATH,
            query_path=QUERY_PATH,
            top_ks=(1, 5, 200),
            latency_runs=1,
        )
        assert report["backend"] == "rank_bm25"
        assert report["corpus_documents"] == 24
        assert report["query_count"] >= 10
        assert 0.0 <= report["quality"]["5"]["recall"] <= 1.0
        sql_row = next(
            row for row in report["queries"] if row["query_id"] == "sql_query_template"
        )
        assert sql_row["first_relevant_rank"] == 1
        assert sql_row["retrieved_frame_ids"][0] == "vid03_f0004"
