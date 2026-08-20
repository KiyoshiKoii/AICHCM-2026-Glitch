from __future__ import annotations

import pytest

from backend.config import Settings
from backend.services.search_orchestrator import SearchService


class _SemanticPipeline:
    def __init__(self) -> None:
        self.payload = None

    async def search_asr(self, payload: dict) -> dict:
        self.payload = payload
        return {
            "status": "success",
            "data": [
                {
                    "asr_id": "L26_V001_asr_000001",
                    "video_id": "L26_V001",
                    "start_ms": 12_345,
                    "end_ms": 18_000,
                    "text": "cho dáº§u vÃ o cháº£o",
                    "score": 0.91,
                    "start_segment_index": 3,
                    "end_segment_index": 5,
                }
            ],
        }


@pytest.mark.asyncio
async def test_search_asr_returns_playable_passage_with_exact_seek(monkeypatch) -> None:
    semantic = _SemanticPipeline()
    monkeypatch.setattr(
        "backend.utils.keyframe_mapper.get_nearest_keyframe_position",
        lambda video_id, timestamp_ms: (
            7,
            {"frame_index": 300, "timestamp_ms": 12_000, "fps": 25.0},
        ),
    )
    service = SearchService(
        settings=Settings(gemini_api_key=None),
        parser=object(),
        dev1=object(),
        dev2=semantic,
    )

    response = await service.search_asr(
        "cho dáº§u vÃ o cháº£o",
        20,
        batch_ids=["L26"],
        video_ids=[],
    )

    assert semantic.payload == {
        "query": "cho dáº§u vÃ o cháº£o",
        "batch_ids": ["L26"],
        "video_ids": [],
        "top_k": 20,
    }
    hit = response.data.results[0]
    assert hit.frame_id == "L26_V001_f0007"
    assert hit.frame_index == 300
    assert hit.metadata["transcript"] == "cho dáº§u vÃ o cháº£o"
    assert hit.metadata["seek_timestamp_ms"] == 12_345
    assert hit.metadata["asr_end_ms"] == 18_000
