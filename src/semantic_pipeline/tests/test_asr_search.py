from __future__ import annotations

import json

from semantic_pipeline.retrieval.asr_search import (
    ElasticsearchASRSearch,
    asr_index_definition,
    build_asr_documents,
    build_asr_query,
    load_asr_segments,
)


def test_load_and_window_asr_segments_preserves_time_range(tmp_path) -> None:
    path = tmp_path / "L26_V001.json"
    path.write_text(
        json.dumps(
            {
                "video_name": "L26_V001",
                "segments": [
                    {"start": 1.0, "end": 3.0, "text": "cho dầu"},
                    {"start": 3.2, "end": 5.0, "text": "vào chảo"},
                    {"start": 30.0, "end": 32.0, "text": "nêm gia vị"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    video_id, segments = load_asr_segments(path)
    documents = build_asr_documents(video_id, segments)

    assert video_id == "L26_V001"
    assert documents[0]["text"] == "cho dầu vào chảo"
    assert (documents[0]["start_ms"], documents[0]["end_ms"]) == (1000, 5000)
    assert documents[1]["text"] == "vào chảo"
    assert documents[2]["text"] == "nêm gia vị"


def test_index_keeps_accented_and_folded_vietnamese_fields() -> None:
    text_mapping = asr_index_definition()["mappings"]["properties"]["text"]

    assert text_mapping["analyzer"] == "vi_text"
    assert text_mapping["fields"]["folded"]["analyzer"] == "folded_text"


def test_query_applies_batch_scope_and_phrase_boost() -> None:
    query = build_asr_query("cho dầu vào chảo", batch_ids=["l26"])

    assert query["bool"]["should"][0]["match_phrase"]["text"]["boost"] == 6.0
    assert query["bool"]["filter"][0]["bool"]["should"] == [
        {"prefix": {"video_id": "L26_"}}
    ]


class _SearchClient:
    def __init__(self) -> None:
        self.request = None

    def search(self, **kwargs):
        self.request = kwargs
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 8.0,
                        "_source": {
                            "asr_id": "a",
                            "video_id": "L26_V001",
                            "start_ms": 1000,
                            "end_ms": 8000,
                            "text": "cho dầu vào chảo",
                        },
                    },
                    {
                        "_score": 7.0,
                        "_source": {
                            "asr_id": "b",
                            "video_id": "L26_V001",
                            "start_ms": 2000,
                            "end_ms": 7000,
                            "text": "dầu vào chảo",
                        },
                    },
                    {
                        "_score": 4.0,
                        "_source": {
                            "asr_id": "c",
                            "video_id": "L26_V002",
                            "start_ms": 5000,
                            "end_ms": 9000,
                            "text": "cho dầu vào chảo khác",
                        },
                    },
                ]
            }
        }


def test_search_deduplicates_overlapping_passages_and_normalizes_score() -> None:
    client = _SearchClient()

    hits = ElasticsearchASRSearch(client).search("cho dầu vào chảo", top_k=10)

    assert [hit["asr_id"] for hit in hits] == ["a", "c"]
    assert hits[0]["score"] == 1.0
    assert hits[1]["score"] == 0.5
    assert client.request["size"] == 100
