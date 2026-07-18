"""Unit tests for Elasticsearch mapping, ingestion and search translation."""

import json
import os
from pathlib import Path

import pytest

import elasticsearch_backend as elasticsearch_backend_module
from elasticsearch_backend import (
    DEFAULT_DEFINITION_PATH,
    ElasticsearchBackend,
    activate_alias,
    create_client,
    build_search_query,
    bulk_ingest,
    collapse_spatial_relations_for_index,
    ensure_index,
    iter_bulk_actions,
    load_index_definition,
    build_parser,
    validate_cli_args,
)
from schemas import SpatialRelation

SEMANTIC_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = SEMANTIC_DIR / "sample_frames" / "metadata.json"


class FakeIndices:
    def __init__(self, exists=False, aliases=None):
        self._exists = exists
        self.aliases = aliases or {}
        self.created = []
        self.alias_actions = None
        self.refreshed = []

    def exists(self, index):
        return self._exists

    def create(self, **kwargs):
        self.created.append(kwargs)
        self._exists = True
        return {"acknowledged": True}

    def get_alias(self, name):
        return self.aliases

    def update_aliases(self, actions):
        self.alias_actions = actions
        return {"acknowledged": True}

    def refresh(self, index):
        self.refreshed.append(index)


class FakeClient:
    def __init__(self, exists=False, aliases=None):
        self.indices = FakeIndices(exists=exists, aliases=aliases)
        self.search_calls = []
        self.count_value = 24

    def ping(self):
        return True

    def count(self, index):
        return {"count": self.count_value}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            "frame_id": "vid01_f0001",
                            "video_name": "vid01.mp4",
                            "frame_index": 1,
                        },
                    }
                ]
            }
        }


class TestClientConfiguration:
    ENV_KEYS = (
        "ELASTICSEARCH_USERNAME",
        "ELASTICSEARCH_PASSWORD",
        "ELASTICSEARCH_CA_CERT",
    )

    @staticmethod
    def _clear_secure_env(monkeypatch):
        for key in TestClientConfiguration.ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_local_http_client_remains_backward_compatible(self, monkeypatch):
        self._clear_secure_env(monkeypatch)
        captured = {}

        def fake_elasticsearch(url, **kwargs):
            captured.update({"url": url, **kwargs})
            return "client"

        monkeypatch.setattr(
            elasticsearch_backend_module, "Elasticsearch", fake_elasticsearch
        )

        assert create_client("http://127.0.0.1:9200") == "client"
        assert captured == {
            "url": "http://127.0.0.1:9200",
            "request_timeout": 30,
        }

    def test_secure_client_uses_basic_auth_and_ca_from_env(
        self, monkeypatch, tmp_path
    ):
        ca_cert = tmp_path / "ca.crt"
        ca_cert.write_text("test CA", encoding="utf-8")
        monkeypatch.setenv("ELASTICSEARCH_USERNAME", "semantic_app")
        monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "dummy-test-secret")
        monkeypatch.setenv("ELASTICSEARCH_CA_CERT", str(ca_cert))
        captured = {}

        def fake_elasticsearch(url, **kwargs):
            captured.update({"url": url, **kwargs})
            return "secure-client"

        monkeypatch.setattr(
            elasticsearch_backend_module, "Elasticsearch", fake_elasticsearch
        )

        assert create_client("https://127.0.0.1:19200") == "secure-client"
        assert captured == {
            "url": "https://127.0.0.1:19200",
            "request_timeout": 30,
            "ca_certs": str(ca_cert.resolve()),
            "basic_auth": ("semantic_app", "dummy-test-secret"),
        }

    def test_rejects_credentials_over_plain_http(self, monkeypatch):
        self._clear_secure_env(monkeypatch)
        with pytest.raises(ValueError, match="only be sent over HTTPS"):
            create_client(
                "http://127.0.0.1:9200",
                username="elastic",
                password="dummy-test-secret",
            )

    def test_rejects_partial_basic_auth(self, monkeypatch):
        self._clear_secure_env(monkeypatch)
        with pytest.raises(ValueError, match="must be set together"):
            create_client("https://127.0.0.1:19200", username="elastic")

    def test_rejects_missing_ca_file(self, monkeypatch, tmp_path):
        self._clear_secure_env(monkeypatch)
        missing_ca = tmp_path / "missing-ca.crt"
        with pytest.raises(ValueError, match="does not exist"):
            create_client(
                "https://127.0.0.1:19200",
                ca_cert=missing_ca,
            )


class TestIndexDefinition:
    def test_mapping_is_explicit_and_strict(self):
        definition = load_index_definition()
        mappings = definition["mappings"]
        assert mappings["dynamic"] == "strict"
        assert mappings["properties"]["frame_id"]["type"] == "keyword"
        assert mappings["properties"]["detections"]["type"] == "nested"
        assert mappings["properties"]["spatial_relations"]["type"] == "nested"

    def test_mapping_json_is_valid_utf8(self):
        raw = json.loads(DEFAULT_DEFINITION_PATH.read_text(encoding="utf-8"))
        assert "kis_english" in raw["settings"]["analysis"]["analyzer"]

    def test_ensure_index_creates_once_without_deleting(self):
        client = FakeClient(exists=False)
        assert ensure_index(client, "semantic_frames_test") is True
        assert client.indices.created[0]["index"] == "semantic_frames_test"
        assert ensure_index(client, "semantic_frames_test") is False
        assert len(client.indices.created) == 1

    def test_alias_switch_removes_old_index_and_adds_new(self):
        client = FakeClient(
            exists=True,
            aliases={"semantic_frames_v0": {"aliases": {"semantic_frames": {}}}},
        )
        activate_alias(client, "semantic_frames_v1", "semantic_frames")
        assert client.indices.alias_actions == [
            {
                "remove": {
                    "index": "semantic_frames_v0",
                    "alias": "semantic_frames",
                }
            },
            {
                "add": {
                    "index": "semantic_frames_v1",
                    "alias": "semantic_frames",
                    "is_write_index": True,
                }
            },
        ]


class TestBulkIngest:
    def test_actions_use_frame_id_as_id_and_materialise_schema_v1(self):
        actions = list(iter_bulk_actions(METADATA_PATH, "semantic_frames_test"))
        assert len(actions) == 24
        first = actions[0]
        assert first["_id"] == first["_source"]["frame_id"]
        assert first["_index"] == "semantic_frames_test"
        assert first["_source"]["schema_version"] == "1.0"
        assert first["_source"]["entities"]["setting"] == "unknown"
        assert first["_source"]["processing"]["spatial_index_policy"] == (
            "label-triple-max-confidence-v1"
        )
        assert first["_source"]["processing"]["spatial_relations_raw_count"] == 0
        assert (
            first["_source"]["processing"]["spatial_relations_indexed_count"]
            == 0
        )

    def test_collapses_query_equivalent_instances_but_keeps_best_confidence(self):
        def relation(subject_id, predicate, object_id, confidence):
            return SpatialRelation.model_validate(
                {
                    "subject_id": subject_id,
                    "subject_label": "flower",
                    "predicate": predicate,
                    "object_id": object_id,
                    "object_label": "flower",
                    "confidence": confidence,
                }
            )

        collapsed = collapse_spatial_relations_for_index(
            [
                relation("flower_0", "left_of", "flower_1", 0.3),
                relation("flower_2", "left_of", "flower_3", 0.4),
                relation("flower_1", "right_of", "flower_0", 0.3),
            ]
        )

        assert len(collapsed) == 2
        by_predicate = {item.predicate.value: item for item in collapsed}
        assert by_predicate["left_of"].subject_id == "flower_2"
        assert by_predicate["right_of"].subject_id == "flower_1"

    def test_bulk_ingest_counts_and_refreshes(self):
        client = FakeClient(exists=True)

        def fake_streaming_bulk(client_arg, actions, **kwargs):
            assert client_arg is client
            assert kwargs["chunk_size"] == 7
            for action in actions:
                yield True, {"index": {"_id": action["_id"]}}

        result = bulk_ingest(
            client,
            METADATA_PATH,
            "semantic_frames_test",
            chunk_size=7,
            streaming_bulk_fn=fake_streaming_bulk,
        )
        assert result == {
            "indexed": 24,
            "documents_in_index": 24,
            "failures": 0,
        }
        assert client.indices.refreshed == ["semantic_frames_test"]

    def test_bulk_ingest_surfaces_failures(self):
        client = FakeClient(exists=True)

        def failing_bulk(client_arg, actions, **kwargs):
            first = next(iter(actions))
            yield False, {"index": {"_id": first["_id"], "error": "bad document"}}

        with pytest.raises(RuntimeError, match="bulk ingest failed"):
            bulk_ingest(
                client,
                METADATA_PATH,
                "semantic_frames_test",
                streaming_bulk_fn=failing_bulk,
            )


class TestSearch:
    def test_query_has_field_boosts_and_filters(self):
        query = build_search_query(
            ["red car", "person"],
            filters={"setting": "outdoor", "objects": ["person", "car"]},
        )
        multi_match = query["bool"]["must"][0]["multi_match"]
        assert multi_match["fields"] == [
            "ocr_text^2.0",
            "ocr_text.stemmed^1.5",
            "caption",
        ]
        assert query["bool"]["filter"] == [
            {"term": {"entities.setting": "outdoor"}},
            {"term": {"entities.objects": "person"}},
            {"term": {"entities.objects": "car"}},
        ]
        assert query["bool"]["should"] == [
            {
                "multi_match": {
                    "query": "red car",
                    "fields": ["ocr_text^3.0", "caption^2.0"],
                    "type": "phrase",
                    "boost": 2.0,
                }
            }
        ]

    def test_rejects_unknown_filter(self):
        with pytest.raises(ValueError, match="Unsupported filter"):
            build_search_query(["person"], {"made_up": "x"})

    def test_spatial_filter_uses_one_nested_query_for_the_whole_triple(self):
        query = build_search_query(
            ["person", "car"],
            filters={
                "spatial_relations": [
                    {"subject": "Person", "predicate": "left_of", "object": "Car"}
                ]
            },
        )

        nested = query["bool"]["filter"][0]["nested"]
        assert nested["path"] == "spatial_relations"
        assert nested["query"]["bool"]["filter"] == [
            {"term": {"spatial_relations.subject_label": "person"}},
            {"term": {"spatial_relations.predicate": "left_of"}},
            {"term": {"spatial_relations.object_label": "car"}},
        ]

    def test_rejects_malformed_spatial_filter(self):
        with pytest.raises(ValueError, match="exactly subject"):
            build_search_query(
                ["person"],
                {"spatial_relations": [{"subject": "person", "object": "car"}]},
            )

    def test_search_returns_existing_api_contract(self):
        client = FakeClient(exists=True)
        backend = ElasticsearchBackend(client=client, index="semantic_frames")
        results = backend.search(["booking", "entity"], top_k=5)
        assert results == [
            {
                "frame_id": "vid01_f0001",
                "score": 12.5,
                "video_name": "vid01.mp4",
                "frame_index": 1,
            }
        ]
        assert client.search_calls[0]["index"] == "semantic_frames"
        assert client.search_calls[0]["size"] == 5

    def test_empty_query_returns_without_calling_elasticsearch(self):
        client = FakeClient(exists=True)
        backend = ElasticsearchBackend(client=client)
        assert backend.search(["", "   "]) == []
        assert client.search_calls == []


class TestCliSafety:
    def test_mutating_command_requires_explicit_physical_index(self):
        args = build_parser().parse_args(["bootstrap"])
        with pytest.raises(ValueError, match="--index-name is required"):
            validate_cli_args(args)

    def test_explicit_versioned_index_is_accepted(self):
        args = build_parser().parse_args(
            ["--index-name", "semantic_frames_v4", "bootstrap"]
        )
        validate_cli_args(args)


@pytest.mark.skipif(
    os.getenv("RUN_ELASTICSEARCH_INTEGRATION") != "1",
    reason="set RUN_ELASTICSEARCH_INTEGRATION=1 after bootstrapping local ES",
)
def test_live_bootstrapped_elasticsearch():
    backend = ElasticsearchBackend()
    assert backend.ping()
    assert backend.document_count() == 24
    assert backend.search(["booking", "entity"], top_k=5)
