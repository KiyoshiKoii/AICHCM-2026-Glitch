from pathlib import Path

import pytest

from semantic_pipeline.video_understanding.loaders import load_keyframe_map


def _write_map(path: Path, rows: list[str]) -> None:
    path.write_text(
        "n,pts_time,fps,frame_idx\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_load_keyframe_map_accepts_repeated_native_frame_index(tmp_path: Path) -> None:
    path = tmp_path / "L24_V003.csv"
    _write_map(
        path,
        [
            "1,556.52,25.0,13913",
            "2,556.56,25.0,13913",
            "3,556.64,25.0,13916",
        ],
    )

    rows = load_keyframe_map(path)

    assert [row.frame_idx for row in rows] == [13913, 13913, 13916]


def test_load_keyframe_map_rejects_descending_native_frame_index(tmp_path: Path) -> None:
    path = tmp_path / "L24_V003.csv"
    _write_map(path, ["1,1.0,25.0,25", "2,1.1,25.0,24"])

    with pytest.raises(ValueError, match="map must be chronological"):
        load_keyframe_map(path)
