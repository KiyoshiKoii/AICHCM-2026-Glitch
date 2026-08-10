import json
from pathlib import Path

from semantic_pipeline.retrieval.elasticsearch_backend import (
    ElasticsearchTextSearch,
    build_lexical_query,
    iter_bulk_actions,
)


def _write_caption_artifact(root: Path) -> Path:
    batch_dir = root / "L21"
    batch_dir.mkdir(parents=True)
    (batch_dir / "L21_V001.json").write_text(
        json.dumps(
            [
                {
                    "frame_id": "L21_V001_f0001",
                    "caption": "A woman holds a red umbrella.",
                    "detailed_caption": "A woman is standing outside with a red umbrella.",
                    "caption_vi": "Một phụ nữ cầm ô màu đỏ.",
                    "detailed_caption_vi": "Một phụ nữ đứng ngoài trời và cầm một chiếc ô màu đỏ.",
                    "ocr_text": "THOI SU",
                    "news_ticker_text": "Tin moi nhat",
                    "detections": [
                        {
                            "object_id": "person_0",
                            "label": "person",
                            "bbox": [0.1, 0.1, 0.4, 0.9],
                            "description": "a woman holding a red umbrella",
                            "description_vi": "một phụ nữ cầm ô màu đỏ",
                            "attributes": ["woman", "red umbrella"],
                            "action": "holding an umbrella",
                        }
                    ],
                    "spatial_relations": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    return root


def test_bulk_actions_join_video_context_and_derive_frame_fields(tmp_path: Path) -> None:
    caption_dir = _write_caption_artifact(tmp_path / "caption")
    youtube_path = tmp_path / "youtube.jsonl"
    youtube_path.write_text(
        json.dumps(
            {
                "video_id": "L21_V001",
                "search_fields": {
                    "title": "60 Giay Sang",
                    "description": "Ban tin buoi sang",
                    "keywords_common": ["HTV News", "tin tuc"],
                    "series_name": "60 Giay",
                    "broadcast_slot": "Sang",
                    "channel_name": "HTV",
                    "source_network": "HTV",
                    "episode_date": "2024-08-01",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    actions = list(iter_bulk_actions(caption_dir, youtube_path, "semantic_frames_test"))

    assert len(actions) == 1
    assert actions[0]["_id"] == "L21_V001_f0001"
    assert actions[0]["_index"] == "semantic_frames_test"
    assert actions[0]["_source"] == {
        "frame_id": "L21_V001_f0001",
        "video_id": "L21_V001",
        "program_code": "L21",
        "frame_number": 1,
        "caption": "A woman holds a red umbrella.",
        "detailed_caption": "A woman is standing outside with a red umbrella.",
        "caption_vi": "Một phụ nữ cầm ô màu đỏ.",
        "detailed_caption_vi": "Một phụ nữ đứng ngoài trời và cầm một chiếc ô màu đỏ.",
        "ocr_text": "THOI SU",
        "news_ticker_text": "Tin moi nhat",
        "detections": [
            {
                "object_id": "person_0",
                "label": "person",
                "bbox": [0.1, 0.1, 0.4, 0.9],
                "description": "a woman holding a red umbrella",
                "description_vi": "một phụ nữ cầm ô màu đỏ",
                "attributes": ["woman", "red umbrella"],
                "action": "holding an umbrella",
            }
        ],
        "video_title": "60 Giay Sang",
        "video_description": "Ban tin buoi sang",
        "video_keywords": "HTV News tin tuc",
        "series_name": "60 Giay",
        "broadcast_slot": "Sang",
        "channel_name": "HTV",
        "source_network": "HTV",
        "episode_date": "2024-08-01",
    }


def test_lexical_query_prioritises_detailed_visual_evidence() -> None:
    query = build_lexical_query(
        ["red umbrella", "tin tuc", "RED UMBRELLA"],
        object_queries=[
            {
                "english_phrase": "woman holding a red umbrella",
                "vietnamese_phrase": "phụ nữ cầm ô màu đỏ",
            }
        ],
        ocr_queries=["THOI SU"],
        program_queries=["60 Giay Sang"],
    )

    assert query["function_score"]["score_mode"] == "sum"
    assert query["function_score"]["boost_mode"] == "sum"
    assert len(query["function_score"]["functions"]) == 1
    assert query["function_score"]["functions"][0]["weight"] == 2.0
    base_query = query["function_score"]["query"]
    clauses = base_query["bool"]["should"]
    assert base_query["bool"]["minimum_should_match"] == 1
    assert sum("combined_fields" in clause for clause in clauses) == 4
    assert sum("nested" in clause for clause in clauses) == 3
    assert any(
        clause.get("combined_fields", {}).get("fields")
        == ["detailed_caption^4", "caption"]
        for clause in clauses
    )
    assert any(
        clause.get("combined_fields", {}).get("fields")
        == ["detailed_caption_vi^4", "caption_vi"]
        for clause in clauses
    )
    assert any(
        clause.get("multi_match", {}).get("type") == "phrase"
        and "news_ticker_text^10" in clause["multi_match"]["fields"]
        for clause in clauses
    )
    nested = next(clause["nested"] for clause in clauses if "nested" in clause)
    assert nested["path"] == "detections"
    assert nested["score_mode"] == "max"
    object_fields = nested["query"]["bool"]["should"][0]["combined_fields"]["fields"]
    assert object_fields == [
        "detections.description",
        "detections.attributes",
        "detections.action",
        "detections.label",
    ]
    structured_object = [clause["nested"] for clause in clauses if "nested" in clause][-1]
    assert structured_object["query"]["bool"]["should"][0]["combined_fields"]["query"] == (
        "woman holding a red umbrella"
    )
    assert structured_object["query"]["bool"]["should"][1]["match"]["detections.description_vi"]["query"] == (
        "phụ nữ cầm ô màu đỏ"
    )
    assert any(
        clause.get("multi_match", {}).get("fields") == ["ocr_text", "news_ticker_text"]
        for clause in clauses
    )
    assert any(
        clause.get("multi_match", {}).get("fields", [None])[0] == "video_title"
        for clause in clauses
    )
    assert any(
        clause.get("multi_match", {}).get("type") == "phrase"
        and clause["multi_match"]["fields"] == ["detailed_caption^2", "caption"]
        for clause in clauses
    )


class _FakeClient:
    def __init__(self) -> None:
        self.request: dict | None = None

    def search(self, **kwargs):
        self.request = kwargs
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            "frame_id": "L21_V001_f0001",
                            "video_id": "L21_V001",
                            "frame_number": 1,
                            "caption": "A woman holds a red umbrella.",
                            "detailed_caption": "",
                            "caption_vi": "Một phụ nữ cầm ô màu đỏ.",
                            "detailed_caption_vi": "",
                            "ocr_text": "",
                            "news_ticker_text": "",
                        },
                        "highlight": {"caption": ["A woman holds a <em>red umbrella</em>."]},
                    }
                ]
            }
        }


def test_search_returns_existing_internal_api_shape() -> None:
    client = _FakeClient()
    backend = ElasticsearchTextSearch(client, index="semantic_frames")

    results = backend.search(["red umbrella"], top_k=5)

    assert client.request is not None
    assert client.request["index"] == "semantic_frames"
    assert results == [
        {
            "frame_id": "L21_V001_f0001",
            "score": 12.5,
            "video_name": "L21_V001",
            "frame_index": 1,
            "metadata": {
                "caption": "A woman holds a red umbrella.",
                "detailed_caption": "",
                "caption_vi": "Một phụ nữ cầm ô màu đỏ.",
                "detailed_caption_vi": "",
                "ocr_text": "",
                "news_ticker_text": "",
                "highlights": {"caption": ["A woman holds a <em>red umbrella</em>."]},
            },
        }
    ]


def test_lexical_query_adds_batch_and_video_scope_filters() -> None:
    query = build_lexical_query(
        ["person"],
        batch_ids=["L21"],
        video_ids=["V006"],
    )

    filters = query["bool"]["filter"]
    assert {"terms": {"program_code": ["l21"]}} in filters
    assert {"wildcard": {"video_id": {"value": "*_V006"}}} in filters
