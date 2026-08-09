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
                    "ocr_text": "THOI SU",
                    "news_ticker_text": "Tin moi nhat",
                    "detections": [],
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
        "ocr_text": "THOI SU",
        "news_ticker_text": "Tin moi nhat",
        "video_title": "60 Giay Sang",
        "video_description": "Ban tin buoi sang",
        "video_keywords": "HTV News tin tuc",
        "series_name": "60 Giay",
        "broadcast_slot": "Sang",
        "channel_name": "HTV",
        "source_network": "HTV",
        "episode_date": "2024-08-01",
    }


def test_lexical_query_keeps_visual_ocr_and_phrase_routes_separate() -> None:
    query = build_lexical_query(["red umbrella", "tin tuc", "RED UMBRELLA"])

    clauses = query["bool"]["should"]
    assert query["bool"]["minimum_should_match"] == 1
    assert sum("combined_fields" in clause for clause in clauses) == 2
    assert any(
        clause.get("multi_match", {}).get("type") == "phrase"
        and "news_ticker_text^10" in clause["multi_match"]["fields"]
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
                "ocr_text": "",
                "news_ticker_text": "",
                "highlights": {"caption": ["A woman holds a <em>red umbrella</em>."]},
            },
        }
    ]
