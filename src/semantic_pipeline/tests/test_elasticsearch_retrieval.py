import json
from pathlib import Path

from semantic_pipeline.core.compact_metadata import (
    CompactDetection,
    CompactSpatialRelation,
    CompactVisualRecord,
)
from semantic_pipeline.retrieval.elasticsearch_backend import (
    ElasticsearchTextSearch,
    build_frame_document,
    build_lexical_query,
    iter_bulk_actions,
)
from semantic_pipeline.retrieval.index_definition import index_definition


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
        "visual_source_frame_id": "L21_V001_f0001",
        "ocr_source_frame_id": "L21_V001_f0001",
        "quality_flags": [],
        "is_visual_representative": True,
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
        "video_title": "60 Giay Sang",
        "video_description": "Ban tin buoi sang",
        "video_keywords": "HTV News tin tuc",
        "series_name": "60 Giay",
        "broadcast_slot": "Sang",
        "channel_name": "HTV",
        "source_network": "HTV",
        "episode_date": "2024-08-01",
    }


def test_bulk_actions_can_limit_ingestion_to_one_batch(tmp_path: Path) -> None:
    caption_dir = _write_caption_artifact(tmp_path / "caption")
    l22_dir = caption_dir / "L22"
    l22_dir.mkdir()
    l22_payload = json.loads((caption_dir / "L21/L21_V001.json").read_text(encoding="utf-8"))
    l22_payload[0]["frame_id"] = "L22_V001_f0001"
    (l22_dir / "L22_V001.json").write_text(
        json.dumps(l22_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    youtube_path = tmp_path / "youtube.jsonl"
    youtube_path.write_text("", encoding="utf-8")

    actions = list(
        iter_bulk_actions(
            caption_dir,
            youtube_path,
            "semantic_frames_test",
            batch_ids=["l22"],
        )
    )

    assert [action["_id"] for action in actions] == ["L22_V001_f0001"]


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


def test_v4_mapping_covers_provenance_and_spatial_relations() -> None:
    properties = index_definition()["mappings"]["properties"]

    assert properties["visual_source_frame_id"] == {"type": "keyword"}
    assert properties["ocr_source_frame_id"] == {"type": "keyword"}
    assert properties["quality_flags"] == {"type": "keyword"}
    assert properties["is_visual_representative"] == {"type": "boolean"}
    assert properties["spatial_relations"]["type"] == "nested"
    assert "subject_label" in properties["spatial_relations"]["properties"]
    assert "object_description_vi" in properties["spatial_relations"]["properties"]


def test_spatial_query_keeps_subject_predicate_object_in_one_nested_relation() -> None:
    query = build_lexical_query(
        [],
        spatial_queries=[
            {
                "subject_english_phrase": "person",
                "subject_vietnamese_phrase": "người",
                "predicate": "left_of",
                "object_english_phrase": "car",
                "object_vietnamese_phrase": "ô tô",
            }
        ],
    )

    relation = query["bool"]["should"][0]["nested"]
    assert relation["path"] == "spatial_relations"
    must = relation["query"]["bool"]["must"]
    assert {"term": {"spatial_relations.predicate": "left_of"}} in must
    assert must[1]["bool"]["minimum_should_match"] == 1
    assert must[2]["bool"]["minimum_should_match"] == 1


def test_interaction_query_binds_subject_action_object_in_one_nested_relation() -> None:
    query = build_lexical_query(
        ["man riding blue motorcycle"],
        interaction_queries=[
            {
                "subject_english_phrase": "person wearing blue shirt",
                "subject_vietnamese_phrase": "người mặc áo xanh",
                "action_english_phrase": "riding",
                "action_vietnamese_phrase": "đang chạy xe",
                "object_english_phrase": "blue motorcycle",
                "object_vietnamese_phrase": "xe máy màu xanh",
            }
        ],
    )

    assert query["function_score"]["functions"][-1]["weight"] == 30.0
    clauses = query["function_score"]["query"]["bool"]["should"]
    interaction = next(
        clause["nested"]
        for clause in clauses
        if clause.get("nested", {}).get("path") == "spatial_relations"
        and clause["nested"].get("boost") == 20.0
    )
    must = interaction["query"]["bool"]["must"]
    assert must[0]["bool"]["minimum_should_match"] == 1
    assert must[1]["bool"]["minimum_should_match"] == 1
    assert must[1]["bool"]["should"][0]["combined_fields"]["query"] == "riding"
    assert must[2]["bool"]["minimum_should_match"] == 1


def test_document_builder_denormalizes_spatial_relation_entities() -> None:
    record = CompactVisualRecord(
        frame_id="L21_V001_f0001",
        visual_source_frame_id="L21_V001_f0001",
        ocr_source_frame_id="L21_V001_f0001",
        caption="A person stands beside a car.",
        detections=[
            CompactDetection(
                object_id="person_0",
                label="person",
                bbox=(0.1, 0.1, 0.3, 0.4),
                description="a person",
            ),
            CompactDetection(
                object_id="car_0",
                label="car",
                bbox=(0.5, 0.1, 0.8, 0.4),
                description="a red car",
            ),
        ],
        spatial_relations=[
            CompactSpatialRelation(
                subject_id="person_0",
                predicate="left_of",
                object_id="car_0",
            )
        ],
    )

    relation = build_frame_document(record)["spatial_relations"][0]
    assert relation["predicate"] == "left_of"
    assert relation["subject_label"] == "person"
    assert relation["object_description"] == "a red car"


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

    results = backend.search(
        ["red umbrella"], top_k=5, collapse_visual_duplicates=True
    )

    assert client.request is not None
    assert client.request["index"] == "semantic_frames"
    assert client.request["collapse"] == {"field": "visual_source_frame_id"}
    assert results == [
        {
            "frame_id": "L21_V001_f0001",
            "score": 12.5,
            "video_name": "L21_V001",
            "frame_index": 1,
                "metadata": {
                    "visual_source_frame_id": "L21_V001_f0001",
                    "ocr_source_frame_id": "L21_V001_f0001",
                    "quality_flags": [],
                    "is_visual_representative": True,
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
