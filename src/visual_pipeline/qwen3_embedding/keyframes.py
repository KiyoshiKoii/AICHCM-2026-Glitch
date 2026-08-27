"""Keyframe discovery and deterministic metadata validation for the pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Keyframe:
    path: Path
    filename: str
    frame_index: int


def _frame_index(path: Path, position: int) -> int:
    """Parse a numeric frame id, falling back to deterministic position."""

    try:
        return int(path.stem)
    except ValueError:
        # Some extracted datasets use names such as ``frame_001``.  Preserve
        # those files instead of silently dropping them; the manifest records
        # the fallback index so the mismatch is visible during validation.
        digits = "".join(char for char in path.stem if char.isdigit())
        if digits:
            return int(digits)
        return position


def discover_keyframes(video_dir: Path) -> list[Keyframe]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Keyframe directory does not exist: {video_dir}")

    paths = sorted(
        (path for path in video_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(f"No keyframes found in: {video_dir}")

    frames = [
        Keyframe(path=path, filename=path.name, frame_index=_frame_index(path, pos))
        for pos, path in enumerate(paths)
    ]
    duplicate_indices = sorted(
        {frame.frame_index for frame in frames if sum(item.frame_index == frame.frame_index for item in frames) > 1}
    )
    if duplicate_indices:
        raise ValueError(
            "Duplicate frame_index values in keyframes: "
            + ", ".join(map(str, duplicate_indices[:10]))
            + (" ..." if len(duplicate_indices) > 10 else "")
        )
    return frames
