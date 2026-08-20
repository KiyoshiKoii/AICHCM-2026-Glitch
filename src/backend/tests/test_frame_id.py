import pytest
from backend.utils.frame_id import build_frame_context
from backend.core.errors import FrameIdError

def test_build_frame_context_normal():
    result = build_frame_context("vid05_f0015", thumbnail_base_url="/media/thumbnails", radius=2)
    data = result.data
    assert data.center_frame.frame_id == "vid05_f0015"
    
    assert len(data.before_frames) == 2
    assert data.before_frames[0].frame_id == "vid05_f0013"
    assert data.before_frames[1].frame_id == "vid05_f0014"
    
    assert len(data.after_frames) == 2
    assert data.after_frames[0].frame_id == "vid05_f0016"
    assert data.after_frames[1].frame_id == "vid05_f0017"

def test_build_frame_context_boundary_early():
    # Frame index 1, radius 3
    result = build_frame_context("vid05_f0001", thumbnail_base_url="/media/thumbnails", radius=3)
    data = result.data
    
    # Should only get frames 0
    assert len(data.before_frames) == 1
    assert data.before_frames[0].frame_id == "vid05_f0000"
    
    assert len(data.after_frames) == 3
    assert data.after_frames[0].frame_id == "vid05_f0002"
    assert data.after_frames[2].frame_id == "vid05_f0004"

def test_build_frame_context_invalid_id():
    with pytest.raises(FrameIdError):
        build_frame_context("invalid_id", thumbnail_base_url="/media/thumbnails", radius=2)


def test_build_frame_context_includes_native_video_positions(monkeypatch):
    positions = {
        2: {"frame_index": 21, "timestamp_ms": 840, "fps": 25.0},
        4: {"frame_index": 83, "timestamp_ms": 3320, "fps": 25.0},
        7: {"frame_index": 165, "timestamp_ms": 6600, "fps": 25.0},
    }
    monkeypatch.setattr(
        "backend.utils.frame_id.get_keyframe_ordinals",
        lambda _video_name: tuple(positions),
    )
    monkeypatch.setattr(
        "backend.utils.frame_id.get_keyframe_position",
        lambda _video_name, ordinal: positions.get(ordinal),
    )

    result = build_frame_context(
        "L26_V001_f0004",
        thumbnail_base_url="/media/thumbnails",
        radius=1,
    )

    assert result.data.before_frames[0].frame_id == "L26_V001_f0002"
    assert result.data.center_frame.frame_index == 83
    assert result.data.center_frame.timestamp_ms == 3320
    assert result.data.center_frame.fps == 25.0
    assert result.data.after_frames[0].frame_id == "L26_V001_f0007"


def test_build_frame_context_can_return_the_full_timeline(monkeypatch):
    positions = {
        ordinal: {
            "frame_index": ordinal * 25,
            "timestamp_ms": ordinal * 1_000,
            "fps": 25.0,
        }
        for ordinal in range(1, 8)
    }
    monkeypatch.setattr(
        "backend.utils.frame_id.get_keyframe_ordinals",
        lambda _video_name: tuple(positions),
    )
    monkeypatch.setattr(
        "backend.utils.frame_id.get_keyframe_position",
        lambda _video_name, ordinal: positions.get(ordinal),
    )

    result = build_frame_context(
        "L26_V001_f0004",
        thumbnail_base_url="/media/thumbnails",
        radius=None,
    )

    assert [item.frame_id for item in result.data.before_frames] == [
        "L26_V001_f0001",
        "L26_V001_f0002",
        "L26_V001_f0003",
    ]
    assert [item.frame_id for item in result.data.after_frames] == [
        "L26_V001_f0005",
        "L26_V001_f0006",
        "L26_V001_f0007",
    ]
