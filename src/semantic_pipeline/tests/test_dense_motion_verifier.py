from __future__ import annotations

import pytest

from semantic_pipeline.retrieval.dense_motion_verifier import (
    DenseMotionTemporalVerifier,
    select_dense_boundary,
)


def test_first_visible_selects_start_of_high_semantic_region() -> None:
    index, confidence = select_dense_boundary(
        semantic_scores=[0.10, 0.20, 0.76, 0.82, 0.80],
        motion_scores=[0.01, 0.05, 0.30, 0.10, 0.08],
        required_anchor="first_visible",
    )

    assert index == 2
    assert 0.0 <= confidence <= 1.0


def test_action_start_uses_motion_only_inside_first_semantic_run() -> None:
    index, _ = select_dense_boundary(
        semantic_scores=[0.10, 0.74, 0.78, 0.81, 0.90],
        motion_scores=[0.99, 0.10, 0.80, 0.20, 1.00],
        required_anchor="action_start",
    )

    assert index == 2


def test_last_complete_selects_end_of_high_semantic_region() -> None:
    index, _ = select_dense_boundary(
        semantic_scores=[0.10, 0.75, 0.82, 0.79, 0.12],
        motion_scores=[0.01, 0.20, 0.30, 0.10, 0.02],
        required_anchor="last_complete",
    )

    assert index == 3


def test_dense_boundary_rejects_misaligned_scores() -> None:
    with pytest.raises(ValueError, match="aligned"):
        select_dense_boundary([0.2], [], "first_visible")


def test_visual_query_removes_temporal_boilerplate() -> None:
    query = DenseMotionTemporalVerifier._visual_query("first moment performers lift the dragon")

    assert query == "a video frame showing performers lift the dragon"


def test_visual_query_prefers_explicit_english_hint() -> None:
    query = DenseMotionTemporalVerifier._visual_query(
        "Khoảnh khắc ban giám khảo giơ bảng điểm (judges raise score cards)"
    )

    assert query == "a video frame showing judges raise score cards"
