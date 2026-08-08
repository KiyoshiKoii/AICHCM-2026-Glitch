from pathlib import Path

from extractor import parse_frame_info
from common import frame_ref


def test_parse_btc_keyframe_path_uses_map_identity(tmp_path, monkeypatch):
    maps = tmp_path / "map-keyframes"
    maps.mkdir()
    (maps / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0,30,0\n2,8.7,30,261\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_ref.paths, "MAP_KEYFRAMES", maps)
    frame_ref.clear_cache()
    try:
        info = parse_frame_info(Path("keyframes/L21_V001/002.jpg"))
    finally:
        frame_ref.clear_cache()
    assert info == {
        "schema_version": "1.1",
        "frame_id": "L21_V001_f0261",
        "video_name": "L21_V001",
        "frame_index": 261,
        "keyframe_n": 2,
        "timestamp_ms": 8700,
        "has_visual_text": True,
    }


def test_parse_legacy_fixture_name_remains_compatible():
    assert parse_frame_info("vid01_f0001.png") == {
        "frame_id": "vid01_f0001",
        "video_name": "vid01.mp4",
        "frame_index": 1,
    }
