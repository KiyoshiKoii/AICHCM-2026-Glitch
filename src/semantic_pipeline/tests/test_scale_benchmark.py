from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_scale import ingest_scaled_corpus, iter_scaled_actions, measure_search_latency
from schemas import FrameMetadata


def _record() -> FrameMetadata:
    return FrameMetadata.model_validate(
        {
            "frame_id": "vid_f0001",
            "video_name": "vid.mp4",
            "frame_index": 1,
            "caption": "A red car is left of a person.",
            "entities": {"objects": ["car", "person"], "colors": ["red"]},
            "detections": [
                {
                    "object_id": "car_0",
                    "label": "car",
                    "bbox": [0.1, 0.1, 0.3, 0.3],
                    "confidence": 0.9,
                }
            ],
        }
    )


def test_iter_scaled_actions_is_deterministic_and_schema_compatible():
    actions = list(iter_scaled_actions([_record()], "scale_v1", 3))

    assert [row["_id"] for row in actions] == [
        "scale_00000000_f0001",
        "scale_00000001_f0001",
        "scale_00000002_f0001",
    ]
    assert all(row["_index"] == "scale_v1" for row in actions)
    for action in actions:
        FrameMetadata.model_validate(action["_source"])


@pytest.mark.parametrize("count", [0, -1])
def test_iter_scaled_actions_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="document_count"):
        list(iter_scaled_actions([_record()], "scale_v1", count))


class _Indices:
    def __init__(self):
        self.refreshed = False

    def exists(self, index):
        return False

    def create(self, index, **definition):
        return {"acknowledged": True}

    def refresh(self, index):
        self.refreshed = True


class _Client:
    def __init__(self):
        self.indices = _Indices()
        self.total = 0

    def count(self, index):
        return {"count": self.total}


def test_ingest_scaled_corpus_streams_and_verifies_count():
    client = _Client()

    def fake_bulk(client_arg, actions, **kwargs):
        assert client_arg is client
        for action in actions:
            client.total += 1
            yield True, {"index": {"_id": action["_id"]}}

    report = ingest_scaled_corpus(
        client,
        [_record()],
        "scale_v1",
        5,
        chunk_size=2,
        streaming_bulk_fn=fake_bulk,
    )

    assert report["documents"] == 5
    assert report["indexed_operations"] == 5
    assert report["documents_per_second"] > 0
    assert client.indices.refreshed is True


def test_existing_non_empty_index_is_refused(monkeypatch):
    client = _Client()
    client.total = 2
    monkeypatch.setattr("benchmark_scale.ensure_index", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="already contains 2"):
        ingest_scaled_corpus(
            client,
            [_record()],
            "scale_v1",
            5,
            streaming_bulk_fn=lambda *args, **kwargs: iter(()),
        )


def test_measure_search_latency_reports_all_percentiles():
    class Backend:
        def search(self, keywords, top_k=200, filters=None):
            return [{"frame_id": "x"}]

    cases = [
        SimpleNamespace(keywords=("car",), filters=None),
        SimpleNamespace(keywords=("car",), filters={"objects": ["car"]}),
    ]
    report = measure_search_latency(
        Backend(), cases, latency_runs=2, top_k=5, warmup_runs=2
    )

    assert report["query_count"] == 2
    assert report["non_empty_query_count"] == 2
    assert report["warmup_runs_per_query"] == 2
    assert report["samples"] == 4
    assert 0 <= report["min_ms"] <= report["p50_ms"] <= report["p95_ms"]
    assert report["p95_ms"] <= report["p99_ms"] <= report["max_ms"]
