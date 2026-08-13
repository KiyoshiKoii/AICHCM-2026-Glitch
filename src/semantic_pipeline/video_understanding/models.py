"""Small, JSON-friendly models for the video-understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KeyframeMapRow:
    n: int
    pts_time: float
    fps: float
    frame_idx: int

    @property
    def timestamp_ms(self) -> int:
        return round(self.pts_time * 1000)


@dataclass(frozen=True)
class ASRSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def start_ms(self) -> int:
        return round(self.start * 1000)

    @property
    def end_ms(self) -> int:
        return round(self.end * 1000)


@dataclass
class RuntimeFrame:
    """Canonical frame kept in memory; raw Gemini metadata is never persisted here."""

    video_id: str
    keyframe_n: int
    frame_id: str
    timestamp_ms: int
    fps: float
    native_frame_idx: int
    raw_metadata: dict[str, Any]
    asr_segment_indices: list[int] = field(default_factory=list)
    nearby_asr_segment_indices: list[int] = field(default_factory=list)


@dataclass
class MicroScene:
    scene_id: str
    scene_type: str
    frame_indices: list[int]
    start_ms: int
    end_ms: int
    representative_keyframe_n: int
    asr_segment_indices: list[int]
    boundary_reason: str = ""


@dataclass
class StoryCandidate:
    title: str
    summary: str
    topics: list[str]
    entities: list[str]
    locations: list[str]
    scene_ids: list[str]
    asr_segment_indices: list[int]
    source_window_id: str | None = None
    uncertain: bool = False
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    visual_states: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceInfo:
    path: str
    sha256: str


@dataclass
class BuildArtifacts:
    timeline: dict[str, Any]
    video_summary: dict[str, Any]
    validation_report: dict[str, Any]
