"""BTC native identity tests."""

from pathlib import Path

import pytest

from src.common import frame_ref


@pytest.fixture
def mapping_root(tmp_path, monkeypatch):
    root = tmp_path / "map-keyframes"
    root.mkdir()
    monkeypatch.setattr(frame_ref.paths, "MAP_KEYFRAMES", root)
    frame_ref.clear_cache()
    yield root
    frame_ref.clear_cache()


def test_round_trip_uses_original_frame_index_and_pts_time(mapping_root):
    (mapping_root / "L21_V001.csv").write_text(
        "\ufeffn,pts_time,fps,frame_idx\n"
        "1,0.0,30.0,0\n"
        "2,8.7,30.0,261\n",
        encoding="utf-8",
    )

    ref = frame_ref.resolve("L21_V001", 2)
    assert frame_ref.frame_id(ref) == "L21_V001_f0261"
    assert frame_ref.parse_frame_id("L21_V001_f0261") == ("L21_V001", 261)
    assert frame_ref.resolve_by_frame_idx("L21_V001", 261) == ref
    assert ref.timestamp_ms == 8700
    assert frame_ref.submission_row(ref) == ("L21_V001", 261)


def test_missing_map_is_a_clear_error(mapping_root):
    with pytest.raises(FileNotFoundError, match="Missing BTC map-keyframes"):
        frame_ref.load_keyframe_map("L99_V999")


@pytest.mark.parametrize(
    "rows, message",
    [
        ("1,0,30,0\n3,3,30,90\n", "contiguous"),
        ("1,0,30,90\n2,3,30,80\n", "non-decreasing"),
    ],
)
def test_rejects_broken_mapping_invariants(mapping_root, rows, message):
    (mapping_root / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n" + rows,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        frame_ref.load_keyframe_map("L21_V001")


def test_duplicate_submission_frame_uses_first_keyframe(mapping_root):
    (mapping_root / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0,30,0\n2,0.033,30,0\n3,3,30,90\n",
        encoding="utf-8",
    )
    assert list(frame_ref.canonical_keyframe_map("L21_V001")) == [1, 3]
    assert frame_ref.resolve_by_frame_idx("L21_V001", 0).keyframe_n == 1
