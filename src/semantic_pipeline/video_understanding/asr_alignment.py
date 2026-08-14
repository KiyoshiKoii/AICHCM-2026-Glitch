"""Temporal alignment between timestamped ASR and Gemini keyframes."""

from __future__ import annotations

import bisect
import re
from collections import defaultdict

from .models import ASRSegment, RuntimeFrame


UNK_RE = re.compile(r"\bunk\b", re.IGNORECASE)


def _nearest_index(timestamps: list[int], target: int) -> int:
    right = bisect.bisect_left(timestamps, target)
    if right == 0:
        return 0
    if right == len(timestamps):
        return len(timestamps) - 1
    left = right - 1
    return left if target - timestamps[left] <= timestamps[right] - target else right


def align_asr(
    frames: list[RuntimeFrame],
    segments: list[ASRSegment],
    context_radius_ms: int = 5000,
) -> dict[int, ASRSegment]:
    """Assign every ASR segment once and return it by source index.

    The primary assignment uses the segment midpoint. Nearby links are added
    only as retrieval context and never duplicate the segment text in output.
    """

    timestamps = [frame.timestamp_ms for frame in frames]
    by_index = {segment.index: segment for segment in segments}
    primary: dict[int, list[int]] = defaultdict(list)
    for segment in segments:
        position = _nearest_index(timestamps, round(segment.midpoint * 1000))
        primary[position].append(segment.index)
        for frame_position, frame in enumerate(frames):
            if abs(frame.timestamp_ms - round(segment.midpoint * 1000)) <= context_radius_ms:
                frame.nearby_asr_segment_indices.append(segment.index)
    for position, indices in primary.items():
        frames[position].asr_segment_indices.extend(sorted(indices))
    if sum(len(items) for items in primary.values()) != len(segments):
        raise AssertionError("ASR segments must be assigned exactly once")
    return by_index


def asr_text_for_frame(frame: RuntimeFrame, segments: dict[int, ASRSegment]) -> str:
    return " ".join(segments[index].text for index in frame.asr_segment_indices)


def quality_flags_for_segments(indices: list[int], segments: dict[int, ASRSegment]) -> list[str]:
    return ["contains_unk"] if any(UNK_RE.search(segments[index].text) for index in indices) else []
