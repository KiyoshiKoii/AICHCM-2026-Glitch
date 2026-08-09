from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_pipeline.quality.clip_caption_alignment import alignment_rows


def test_alignment_flags_a_caption_that_matches_another_frame_better():
    rows = alignment_rows(
        ["L21_V001_f0001", "L21_V001_f0002"],
        [[0.21, 0.34], [0.31, 0.22]],
        min_score=0.20,
        min_margin=0.02,
    )

    assert rows[0]["flag"] is True
    assert "alternate_image_higher" in rows[0]["reasons"]
    assert rows[0]["best_match_frame_id"] == "L21_V001_f0002"


def test_alignment_keeps_a_clear_diagonal_pairing_unflagged():
    rows = alignment_rows(
        ["L21_V001_f0001", "L21_V001_f0002"],
        [[0.42, 0.18], [0.19, 0.39]],
        min_score=0.20,
        min_margin=0.02,
    )

    assert [row["flag"] for row in rows] == [False, False]


def test_alignment_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        alignment_rows(["frame"], [[0.2, 0.3]], min_score=0.2, min_margin=0.02)
