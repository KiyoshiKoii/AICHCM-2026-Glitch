"""Read-only checks against locally attached BTC data (skipped in clean CI)."""

from pathlib import Path

import pytest

from src.common import frame_ref, paths
from schemas import validate_metadata_file


HAS_BTC_MAPS = paths.MAP_KEYFRAMES.is_dir() and any(paths.MAP_KEYFRAMES.glob("*.csv"))


@pytest.mark.skipif(not HAS_BTC_MAPS, reason="local BTC map-keyframes data not attached")
def test_all_attached_btc_maps_round_trip_and_join_object_files():
    map_files = sorted(paths.MAP_KEYFRAMES.glob("*.csv"))
    assert map_files
    total_frames = 0
    for source in map_files:
        video_id = source.stem
        refs = frame_ref.load_keyframe_map(video_id)
        assert list(refs) == list(range(1, len(refs) + 1))
        assert all(
            frame_ref.resolve_by_frame_idx(video_id, ref.frame_idx) == ref
            and frame_ref.parse_frame_id(frame_ref.frame_id(ref))
            == (video_id, ref.frame_idx)
            for ref in frame_ref.canonical_keyframe_map(video_id).values()
        )
        object_dir = paths.objects_dir(video_id)
        if object_dir.is_dir():
            assert len(list(object_dir.glob("*.json"))) == len(refs)
        total_frames += len(refs)
    assert total_frames > 0


@pytest.mark.skipif(not HAS_BTC_MAPS, reason="local BTC data not attached")
def test_l21_l22_processed_indexes_cover_every_attached_keyframe():
    pyarrow = pytest.importorskip("pyarrow.parquet")
    videos = [
        source.stem
        for source in sorted(paths.MAP_KEYFRAMES.glob("L2[12]_V*.csv"))
    ]
    if not videos or any(not paths.objects_index(video).is_file() for video in videos):
        pytest.skip("run scripts/build_objects_index.py --videos L21,L22 first")
    indexed_frames = 0
    for video_id in videos:
        parquet_rows = pyarrow.read_metadata(paths.objects_index(video_id)).num_rows
        assert parquet_rows == len(frame_ref.load_keyframe_map(video_id))
        indexed_frames += parquet_rows
    assert indexed_frames > 0


@pytest.mark.skipif(not HAS_BTC_MAPS, reason="local BTC data not attached")
def test_generated_btc_metadata_uses_native_identity_and_pruned_objects():
    source = paths.processed_metadata_dir() / "L21_V001.json"
    if not source.is_file():
        pytest.skip("run metadata_builder for L21_V001 first")
    records = validate_metadata_file(source)
    assert len(records) == len(frame_ref.canonical_keyframe_map("L21_V001"))
    for record in records:
        ref = frame_ref.resolve("L21_V001", record.keyframe_n)
        assert record.frame_id == frame_ref.frame_id(ref)
        assert record.frame_index == ref.frame_idx
        assert record.timestamp_ms == ref.timestamp_ms
        assert len(record.detections) <= 15
        assert all(
            item.label_source in {"btc_detector", "context_grounding"}
            and item.mid
            and item.confidence >= 0.2
            for item in record.detections
        )
        assert sum(record.object_counts.values()) == len(record.detections)
