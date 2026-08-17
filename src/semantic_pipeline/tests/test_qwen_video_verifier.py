from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from semantic_pipeline.retrieval.qwen_video_verifier import (
    QwenTemporalVerifier,
    QwenVerifierConfig,
    TemporalVerificationRequest,
    TemporalVerificationResult,
)
from semantic_pipeline.retrieval.video_window_sampler import SampledVideoFrame, VideoWindow


@pytest.fixture
def window() -> VideoWindow:
    return VideoWindow(
        video_path=Path("L23_V001.mp4"),
        start_ms=1_000,
        end_ms=2_000,
        duration_ms=4_000,
        frames=(
            SampledVideoFrame(timestamp_ms=1_000, image=object()),
            SampledVideoFrame(timestamp_ms=1_500, image=object()),
            SampledVideoFrame(timestamp_ms=2_000, image=object()),
        ),
    )


def test_qwen_response_maps_selected_frame_to_native_timestamp(window: VideoWindow) -> None:
    parsed = QwenTemporalVerifier._parse_response(
        '{"supported": true, "selected_frame_index": 1, "selected_state": "transition", '
        '"confidence": 0.83, "reason": "first visible contact"}',
        window,
    )

    assert parsed["supported"] is True
    assert parsed["timestamp_ms"] == 1_500
    assert parsed["confidence"] == 0.83


def test_qwen_response_falls_back_when_frame_index_is_invalid(window: VideoWindow) -> None:
    parsed = QwenTemporalVerifier._parse_response(
        '{"supported": true, "selected_frame_index": 8, "selected_state": "after", "confidence": 2}',
        window,
    )

    assert parsed["supported"] is False
    assert parsed["timestamp_ms"] is None
    assert parsed["confidence"] == 1.0


def test_qwen_result_cache_round_trip(tmp_path: Path) -> None:
    verifier = QwenTemporalVerifier(QwenVerifierConfig(cache_dir=tmp_path))
    request = TemporalVerificationRequest(
        video_id="L23_V001",
        event_id="L23_V001_event_1",
        event_text="first cyclists visible",
        required_anchor="first_visible",
        video_path=Path("data/videos/L23_V001.mp4"),
        start_ms=1_000,
        end_ms=3_000,
    )
    written = TemporalVerificationResult(
        supported=True,
        timestamp_ms=1_250,
        confidence=0.9,
        selected_state="transition",
        verifier="fake-qwen",
        coarse_window_ms=(0, 4_000),
        refined_window_ms=(1_000, 1_500),
        reason="first visible frame",
    )

    verifier._write_cache(request, written)

    assert verifier._read_cache(request) == written


def test_qwen_resizes_images_to_a_bounded_multiple_of_vision_patch_size() -> None:
    verifier = QwenTemporalVerifier(QwenVerifierConfig(image_max_side=280))

    resized = verifier._resize_image(Image.new("RGB", (1_920, 1_080)))

    assert max(resized.size) <= 280
    assert resized.size[0] % 28 == 0
    assert resized.size[1] % 28 == 0
