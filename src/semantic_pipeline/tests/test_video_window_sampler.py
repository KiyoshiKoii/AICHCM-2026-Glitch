from __future__ import annotations

import pytest

from semantic_pipeline.retrieval.video_window_sampler import bounded_window, sample_timestamps


def test_bounded_window_clamps_to_real_video_duration() -> None:
    assert bounded_window(center_ms=500, radius_ms=1_000, duration_ms=2_000) == (0, 1_500)
    assert bounded_window(center_ms=2_500, radius_ms=1_000, duration_ms=2_000) == (1_000, 2_000)


def test_sample_timestamps_keeps_window_ends_when_capped() -> None:
    timestamps = sample_timestamps(start_ms=100, end_ms=2_100, fps=10, max_frames=5)

    assert timestamps[0] == 100
    assert timestamps[-1] == 2_100
    assert len(timestamps) == 5
    assert timestamps == sorted(set(timestamps))


def test_sample_timestamps_rejects_invalid_sampling_settings() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_timestamps(start_ms=0, end_ms=10, fps=0, max_frames=1)
    with pytest.raises(ValueError, match="at least one"):
        sample_timestamps(start_ms=0, end_ms=10, fps=1, max_frames=0)
