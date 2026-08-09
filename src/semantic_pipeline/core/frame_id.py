"""Canonical parsing for keyframe identifiers.

Persisted visual metadata keeps only ``frame_id``.  Other identifiers are
derived at the system boundary so the corpus does not duplicate them per row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_FRAME_ID_PATTERN = re.compile(r"^(?P<video_name>.+)_f(?P<frame_index>\d+)$")


@dataclass(frozen=True)
class FrameReference:
    frame_id: str
    video_name: str
    frame_index: int


def parse_frame_id(frame_id: str) -> FrameReference:
    """Parse ``L21_V001_f0017`` into its derived video and keyframe number."""
    match = _FRAME_ID_PATTERN.fullmatch(frame_id.strip())
    if match is None:
        raise ValueError(
            "frame_id must use '<video_name>_f<frame_number>', "
            f"got {frame_id!r}"
        )
    return FrameReference(
        frame_id=match.group(0),
        video_name=match.group("video_name"),
        frame_index=int(match.group("frame_index")),
    )


def frame_id_from_path(path: str | Path) -> str:
    """Derive a frame ID from either ``L21_V001_f0017.jpg`` or ``001.jpg``."""
    image_path = Path(path)
    stem = image_path.stem
    if "_f" in stem:
        return parse_frame_id(stem).frame_id
    try:
        number = int(stem)
    except ValueError as exc:
        raise ValueError(f"Cannot derive frame_id from image path {image_path}") from exc
    return f"{image_path.parent.name}_f{number:04d}"
