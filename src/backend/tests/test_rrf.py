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
    # A appears in dev1 (rank 1), C in dev2 (rank 2)
    assert len(results) == 3
    assert results[0].frame_id == "B"
    assert results[1].frame_id == "A"
    assert results[2].frame_id == "C"

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
