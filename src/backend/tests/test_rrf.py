import pytest
from backend.schemas.search import UpstreamResult
from backend.utils.rrf import reciprocal_rank_fusion

def test_rrf_basic():
    # Dev 1 returns frame A and B
    dev1_results = [
        UpstreamResult(frame_id="A", score=0.9),
        UpstreamResult(frame_id="B", score=0.8),
    ]
    # Dev 2 returns frame B and C
    dev2_results = [
        UpstreamResult(frame_id="B", score=0.85),
        UpstreamResult(frame_id="C", score=0.7),
    ]
    
    rankings = {
        "dev1": dev1_results,
        "dev2": dev2_results
    }
    
    results = reciprocal_rank_fusion(
        rankings,
        k=60,
        limit=10,
        thumbnail_base_url="/media/thumbnails"
    )
    
    # B appears in both, should have highest score
    # C receives the configured semantic-source weight boost and outranks A.
    assert len(results) == 3
    assert results[0].frame_id == "B"
    assert results[1].frame_id == "C"
    assert results[2].frame_id == "A"

def test_rrf_limit():
    dev1_results = [
        UpstreamResult(frame_id=f"frame_{i}", score=0.9) for i in range(10)
    ]
    rankings = {"dev1": dev1_results}
    
    results = reciprocal_rank_fusion(
        rankings,
        k=60,
        limit=5,
        thumbnail_base_url="/media/thumbnails"
    )
    
    assert len(results) == 5

def test_rrf_duplicate_frame():
    # A source returning the same frame twice should only count it once
    dev1_results = [
        UpstreamResult(frame_id="A", score=0.9),
        UpstreamResult(frame_id="A", score=0.8), # duplicate
    ]
    rankings = {"dev1": dev1_results}
    
    results = reciprocal_rank_fusion(
        rankings,
        k=60,
        limit=10,
        thumbnail_base_url="/media/thumbnails"
    )
    
    assert len(results) == 1
    # RRF score should be 1 / (60 + 1)
    assert results[0].score == 1.0 / 61


def test_rrf_respects_explicit_visual_text_weights():
    rankings = {
        "dev1": [UpstreamResult(frame_id="visual", score=0.9)],
        "dev2": [UpstreamResult(frame_id="text", score=0.9)],
    }

    results = reciprocal_rank_fusion(
        rankings,
        limit=2,
        thumbnail_base_url="/media/thumbnails",
        source_weights={"dev1": 0.8, "dev2": 0.2},
    )

    assert [item.frame_id for item in results] == ["visual", "text"]


def test_rrf_zero_weight_excludes_a_source():
    rankings = {
        "dev1": [UpstreamResult(frame_id="visual", score=0.9)],
        "dev2": [UpstreamResult(frame_id="text", score=0.9)],
    }

    results = reciprocal_rank_fusion(
        rankings,
        limit=2,
        thumbnail_base_url="/media/thumbnails",
        source_weights={"dev1": 0.0, "dev2": 1.0},
    )

    assert [item.frame_id for item in results] == ["text"]
