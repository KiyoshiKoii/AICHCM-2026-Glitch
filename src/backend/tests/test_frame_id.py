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
