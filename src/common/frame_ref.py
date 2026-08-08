"""Canonical BTC keyframe identities shared by every pipeline.

BTC uses two different frame numbers: ``n`` names keyframe/object files while
``frame_idx`` is the original video frame submitted to AIC.  This module keeps
that distinction explicit and makes ``map-keyframes`` the only source of truth.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache

from . import paths


_FRAME_ID_RE = re.compile(r"^(?P<video_id>.+)_f(?P<frame_idx>\d{4,})$")


@dataclass(frozen=True, slots=True)
class FrameRef:
    video_id: str
    keyframe_n: int
    frame_idx: int
    pts_time: float
    fps: float

    @property
    def timestamp_ms(self) -> int:
        return round(self.pts_time * 1000)


@lru_cache(maxsize=64)
def load_keyframe_map(video_id: str) -> dict[int, FrameRef]:
    """Load and validate one BTC ``map-keyframes`` CSV, keyed by ``n``."""
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id must not be blank")
    path = paths.map_keyframes(video_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BTC map-keyframes file for {video_id}: {path}"
        )

    refs: dict[int, FrameRef] = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"n", "pts_time", "fps", "frame_idx"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    ref = FrameRef(
                        video_id=video_id,
                        keyframe_n=int(row["n"]),
                        frame_idx=int(row["frame_idx"]),
                        pts_time=float(row["pts_time"]),
                        fps=float(row["fps"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid BTC keyframe mapping at {path}:{line_number}: {row}"
                    ) from exc
                if ref.keyframe_n in refs:
                    raise ValueError(
                        f"Duplicate keyframe n={ref.keyframe_n} in {path}"
                    )
                if ref.keyframe_n < 1 or ref.frame_idx < 0 or ref.pts_time < 0:
                    raise ValueError(f"Negative/zero mapping value at {path}:{line_number}")
                if ref.fps <= 0:
                    raise ValueError(f"fps must be positive at {path}:{line_number}")
                refs[ref.keyframe_n] = ref
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV in {path}: {exc}") from exc

    if not refs:
        raise ValueError(f"BTC keyframe map is empty: {path}")
    expected = list(range(1, len(refs) + 1))
    if sorted(refs) != expected:
        raise ValueError(f"BTC keyframe n must be contiguous 1..N in {path}")
    ordered = [refs[n] for n in expected]
    if any(a.frame_idx > b.frame_idx for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"BTC frame_idx must be non-decreasing in {path}")
    if any(a.pts_time > b.pts_time for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"BTC pts_time must be non-decreasing in {path}")
    return refs


def frame_id(ref: FrameRef) -> str:
    return f"{ref.video_id}_f{ref.frame_idx:04d}"


def parse_frame_id(value: str) -> tuple[str, int]:
    match = _FRAME_ID_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "frame_id must have form '<video_id>_f<frame_idx>' with at least "
            "four frame digits"
        )
    return match.group("video_id"), int(match.group("frame_idx"))


def resolve(video_id: str, keyframe_n: int) -> FrameRef:
    try:
        return load_keyframe_map(video_id)[keyframe_n]
    except KeyError as exc:
        raise KeyError(
            f"Unknown BTC keyframe n={keyframe_n} for video {video_id}"
        ) from exc


def resolve_by_frame_idx(video_id: str, frame_idx: int) -> FrameRef:
    """Resolve to the first keyframe when BTC repeats a rounded ``frame_idx``.

    Some official maps contain two keyframes for the same original frame (for
    example 0.000s and 0.033s both mapped to frame 0).  AIC submission identity
    cannot distinguish them, so the first ``n`` is the canonical representative.
    """
    try:
        return _load_frame_index_map(video_id)[frame_idx]
    except KeyError as exc:
        raise KeyError(
            f"Unknown BTC frame_idx={frame_idx} for video {video_id}"
        ) from exc


@lru_cache(maxsize=64)
def _load_frame_index_map(video_id: str) -> dict[int, FrameRef]:
    by_frame_idx: dict[int, FrameRef] = {}
    for ref in load_keyframe_map(video_id).values():
        by_frame_idx.setdefault(ref.frame_idx, ref)
    return by_frame_idx


def canonical_keyframe_map(video_id: str) -> dict[int, FrameRef]:
    """Return one deterministic keyframe per unique submission ``frame_idx``."""
    canonical: dict[int, FrameRef] = {}
    seen_frame_indices: set[int] = set()
    for keyframe_n, ref in load_keyframe_map(video_id).items():
        if ref.frame_idx in seen_frame_indices:
            continue
        seen_frame_indices.add(ref.frame_idx)
        canonical[keyframe_n] = ref
    return canonical


def submission_row(ref: FrameRef) -> tuple[str, int]:
    return ref.video_id, ref.frame_idx


def clear_cache() -> None:
    """Test/setup helper for callers that replace ``AIC_DATA_ROOT`` at runtime."""
    load_keyframe_map.cache_clear()
    _load_frame_index_map.cache_clear()
