"""Dense, bounded frame sampling for temporal-event verification.

The normal retrieval corpus stays sparse and inexpensive.  This module opens a
raw video only after a candidate event has been selected, so a VLM can inspect
the short interval around that event without materializing dense frames for the
whole collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampledVideoFrame:
    """One RGB frame paired with its source-video timestamp."""

    timestamp_ms: int
    image: Any


@dataclass(frozen=True)
class VideoWindow:
    """A sampled interval and the effective bounds used to obtain it."""

    video_path: Path
    start_ms: int
    end_ms: int
    duration_ms: int
    frames: tuple[SampledVideoFrame, ...]


def bounded_window(
    *,
    center_ms: int,
    radius_ms: int,
    duration_ms: int,
) -> tuple[int, int]:
    """Return an inclusive interval clamped to a video's real duration."""

    end_limit = max(0, int(duration_ms))
    center = min(end_limit, max(0, int(center_ms)))
    radius = max(0, int(radius_ms))
    return max(0, center - radius), min(end_limit, center + radius)


def sample_timestamps(
    *,
    start_ms: int,
    end_ms: int,
    fps: float,
    max_frames: int,
) -> list[int]:
    """Return deterministic inclusive timestamps, capped without clustering."""

    start = max(0, int(start_ms))
    end = max(start, int(end_ms))
    if fps <= 0:
        raise ValueError("sampling fps must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be at least one")
    step_ms = max(1, round(1_000 / fps))
    timestamps = list(range(start, end + 1, step_ms))
    if not timestamps or timestamps[-1] != end:
        timestamps.append(end)
    if len(timestamps) <= max_frames:
        return timestamps
    if max_frames == 1:
        return [timestamps[0]]
    indices = [round(index * (len(timestamps) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [timestamps[index] for index in dict.fromkeys(indices)]


def sample_video_window(
    video_path: Path,
    *,
    start_ms: int,
    end_ms: int,
    fps: float,
    max_frames: int = 32,
) -> VideoWindow:
    """Decode a small RGB frame sequence from a raw video using OpenCV."""

    if not video_path.is_file():
        raise FileNotFoundError(f"raw video does not exist: {video_path}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("Qwen temporal verification requires opencv-python") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open raw video: {video_path}")
    try:
        raw_duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1.0)
        duration_ms = max(0, round(raw_duration * 1_000))
        bounded_start = min(duration_ms, max(0, int(start_ms)))
        bounded_end = min(duration_ms, max(bounded_start, int(end_ms)))
        timestamps = sample_timestamps(
            start_ms=bounded_start,
            end_ms=bounded_end,
            fps=fps,
            max_frames=max_frames,
        )
        source_fps = max(capture.get(cv2.CAP_PROP_FPS), 1.0)
        target_frames = [round(timestamp_ms * source_fps / 1_000) for timestamp_ms in timestamps]
        current_frame = target_frames[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        frames: list[SampledVideoFrame] = []
        for timestamp_ms, target_frame in zip(timestamps, target_frames):
            while current_frame < target_frame:
                if not capture.grab():
                    break
                current_frame += 1
            ok, bgr = capture.read()
            current_frame += 1
            if not ok or bgr is None:
                continue
            frames.append(
                SampledVideoFrame(
                    timestamp_ms=timestamp_ms,
                    image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                )
            )
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path} in {bounded_start}..{bounded_end} ms")
    return VideoWindow(
        video_path=video_path,
        start_ms=bounded_start,
        end_ms=bounded_end,
        duration_ms=duration_ms,
        frames=tuple(frames),
    )
