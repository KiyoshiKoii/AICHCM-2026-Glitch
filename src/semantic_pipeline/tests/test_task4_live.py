"""Opt-in live acceptance: FastAPI -> Elasticsearch v4 -> metadata filters."""

import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_TASK4_LIVE") != "1",
    reason="set RUN_TASK4_LIVE=1 with Elasticsearch v4 running",
)
def test_live_fastapi_entity_spatial_and_all_of_filters(monkeypatch):
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setenv("SEMANTIC_SEARCH_BACKEND", "elasticsearch")
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    monkeypatch.setenv("ELASTICSEARCH_INDEX", "semantic_frames")

    with TestClient(server.app) as client:
        text_only = client.post(
            "/internal/search/text",
            json={"keywords": ["booking", "entity"], "top_k": 5},
        )
        assert text_only.status_code == 200
        assert text_only.json()["data"][0]["frame_id"] == "vid01_f0001"

        entity_filtered = client.post(
            "/internal/search/text",
            json={
                "keywords": ["field", "flowers"],
                "top_k": 5,
                "filters": {"setting": "outdoor", "objects": ["flowers"]},
            },
        )
        assert entity_filtered.status_code == 200
        assert [item["frame_id"] for item in entity_filtered.json()["data"]] == [
            "vid03_f0009"
        ]

        spatial_filtered = client.post(
            "/internal/search/text",
            json={
                "keywords": ["domino", "pieces"],
                "top_k": 5,
                "filters": {
                    "spatial_relations": [
                        {
                            "subject": "domino",
                            "predicate": "left_of",
                            "object": "domino",
                        }
                    ]
                },
            },
        )
        assert spatial_filtered.status_code == 200
        assert [item["frame_id"] for item in spatial_filtered.json()["data"]] == [
            "vid01_f0007"
        ]

        # List filters are AND/all-of: adding an absent object must exclude it.
        all_of_negative = client.post(
            "/internal/search/text",
            json={
                "keywords": ["field", "flowers"],
                "top_k": 5,
                "filters": {"objects": ["flowers", "airplane"]},
            },
        )
        assert all_of_negative.status_code == 200
        assert all_of_negative.json()["data"] == []
