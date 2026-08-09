"""Video-level context used to specialize Gemini visual extraction prompts.

The context is deliberately ephemeral: it selects a prompt profile for a
batch, but is not copied into every compact frame record.  Dataset-level
facets can be added later at indexing time without re-running Gemini.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .frame_id import frame_id_from_path, parse_frame_id


DEFAULT_YOUTUBE_METADATA_PATH = Path("data/metadata/metadata_youtube.jsonl")
_COLLECTION_PATTERN = re.compile(r"^(?P<collection>L\d+)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class VisualProfile:
    profile_id: str
    collection_code: str
    content_domain: str
    program_name: str
    source: str
    guidance: str


@dataclass(frozen=True)
class VideoContext:
    video_id: str
    profile: VisualProfile
    title: str | None = None
    series_name: str | None = None
    channel_name: str | None = None
    source_network: str | None = None
    broadcast_slot: str | None = None
    episode_date: str | None = None

    def prompt_context(self) -> str:
        """Return a weak, explicitly non-authoritative hint for Gemini."""
        lines = [
            "Video-level context is only a weak format hint; trust visible pixels over it.",
            f"Collection: {self.profile.collection_code} ({self.profile.content_domain}).",
            f"Program/source: {self.profile.program_name} — {self.profile.source}.",
        ]
        if self.title:
            lines.append(f"Episode title hint: {self.title}")
        if self.broadcast_slot:
            lines.append(f"Broadcast slot hint: {self.broadcast_slot}")
        lines.extend(
            [
                "Do not claim a title, location, event, or object merely because it appears in this context.",
                f"Domain priorities: {self.profile.guidance}",
            ]
        )
        return "\n".join(lines)


UNKNOWN_PROFILE = VisualProfile(
    profile_id="generic",
    collection_code="unknown",
    content_domain="general visual content",
    program_name="unknown",
    source="unknown",
    guidance=(
        "describe the visible scene broadly and detect only clearly bounded physical objects "
        "or structures"
    ),
)


VISUAL_PROFILES: dict[str, VisualProfile] = {
    "L21": VisualProfile(
        "news",
        "L21",
        "news",
        "60 Giây Sáng",
        "60 Giây Official / HTV",
        "newsroom, field report, interview, vehicle, landmark, sign, map, chart, and on-screen text",
    ),
    "L22": VisualProfile(
        "news",
        "L22",
        "news",
        "60 Giây Chiều",
        "60 Giây Official / HTV",
        "newsroom, field report, interview, vehicle, landmark, sign, map, chart, and on-screen text",
    ),
    "L23": VisualProfile(
        "cycling",
        "L23",
        "cycling sport",
        "Cúp Truyền Hình TP.HCM 2024",
        "HTV Sports",
        "cyclist, bicycle, helmet, race number, team jersey, peloton, finish line, and race graphics",
    ),
    "L24": VisualProfile(
        "lion_dance",
        "L24",
        "lion and dragon dance",
        "Cúp Chợ Lớn HTV 2024",
        "HTV Sports",
        "lion or dragon costume, performer, drum, pole, prop, team uniform, and performance area",
    ),
    "L25": VisualProfile(
        "education",
        "L25",
        "education",
        "Bí quyết ôn thi THPT 2024",
        "Báo Thanh Niên",
        "teacher, student, board, slide, diagram, formula, document, screen, and clearly legible text",
    ),
    "L26": VisualProfile(
        "cooking",
        "L26",
        "food and cooking",
        "Món ngon mỗi ngày / Thế giới ẩm thực",
        "ViVU TV",
        "dish, ingredient, cookware, utensil, stove, food preparation, and finished meal",
    ),
    "L27": VisualProfile(
        "travel_culture",
        "L27",
        "travel, food, and local culture",
        "Việt Nam Đi Là Ghiền",
        "HTV Giải Trí",
        "landmark, landscape, local dish, transport, sign, craft, and cultural activity",
    ),
    "L28": VisualProfile(
        "mekong_documentary",
        "L28",
        "Mekong life documentary",
        "Tản Mạn Mê Kông / Đến Và Ở Lại",
        "HTV Entertainment",
        "river, canal, boat, waterfront home, fishing, agriculture, and natural environment",
    ),
    "L29": VisualProfile(
        "mekong_documentary",
        "L29",
        "Mekong livelihoods and environment",
        "Đôi Mắt Mê Kông",
        "HTV Entertainment",
        "river, canal, boat, livelihood, farming, local home, environmental condition, and community activity",
    ),
    "L30": VisualProfile(
        "community_story",
        "L30",
        "community story",
        "Lan tỏa năng lượng tích cực 2024",
        "Báo Tuổi Trẻ",
        "people, occupation, interaction, assistance, tools, setting, and visible community activity; do not infer identity or emotion",
    ),
}


def collection_code_from_video_id(video_id: str) -> str:
    match = _COLLECTION_PATTERN.match(video_id.strip())
    return match.group("collection").upper() if match else "unknown"


def profile_for_video_id(video_id: str) -> VisualProfile:
    return VISUAL_PROFILES.get(collection_code_from_video_id(video_id), UNKNOWN_PROFILE)


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def load_youtube_context(path: str | Path = DEFAULT_YOUTUBE_METADATA_PATH) -> dict[str, dict[str, object]]:
    """Load only compact video-level fields needed for prompt context.

    The source contains a larger duplicated search corpus; this adapter avoids
    putting that text into prompts or retaining it in the frame metadata.
    """
    metadata_path = Path(path)
    if not metadata_path.is_file():
        return {}
    contexts: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid YouTube metadata at line {line_number}") from exc
        video_id = _optional_text(raw.get("video_id"))
        if not video_id:
            raise ValueError(f"YouTube metadata line {line_number} has no video_id")
        search_fields = raw.get("search_fields") if isinstance(raw.get("search_fields"), dict) else {}
        filters = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        contexts[video_id] = {
            "title": _optional_text(search_fields.get("title")) or _optional_text(payload.get("title")),
            "series_name": _optional_text(search_fields.get("series_name")) or _optional_text(filters.get("series_name")),
            "channel_name": _optional_text(search_fields.get("channel_name")) or _optional_text(payload.get("channel_name")),
            "source_network": _optional_text(search_fields.get("source_network")) or _optional_text(filters.get("source_network")),
            "broadcast_slot": _optional_text(search_fields.get("broadcast_slot")) or _optional_text(filters.get("broadcast_slot")),
            "episode_date": _optional_text(search_fields.get("episode_date")) or _optional_text(filters.get("episode_date")),
        }
    return contexts


class VisualContextResolver:
    """Resolve one stable prompt context for each video directory."""

    def __init__(self, youtube_context: dict[str, dict[str, object]] | None = None):
        self.youtube_context = youtube_context or {}

    @classmethod
    def from_youtube_metadata(cls, path: str | Path = DEFAULT_YOUTUBE_METADATA_PATH) -> "VisualContextResolver":
        return cls(load_youtube_context(path))

    def for_frame_id(self, frame_id: str) -> VideoContext:
        video_id = parse_frame_id(frame_id).video_name
        profile = profile_for_video_id(video_id)
        source = self.youtube_context.get(video_id, {})
        return VideoContext(
            video_id=video_id,
            profile=profile,
            title=source.get("title") if isinstance(source.get("title"), str) else None,
            series_name=source.get("series_name") if isinstance(source.get("series_name"), str) else None,
            channel_name=source.get("channel_name") if isinstance(source.get("channel_name"), str) else None,
            source_network=source.get("source_network") if isinstance(source.get("source_network"), str) else None,
            broadcast_slot=source.get("broadcast_slot") if isinstance(source.get("broadcast_slot"), str) else None,
            episode_date=source.get("episode_date") if isinstance(source.get("episode_date"), str) else None,
        )

    def for_paths(self, paths: Iterable[Path]) -> VideoContext:
        contexts = [self.for_frame_id(frame_id_from_path(path)) for path in paths]
        if not contexts:
            raise ValueError("Cannot resolve context for an empty batch")
        first = contexts[0]
        mismatched = [context.video_id for context in contexts if context.video_id != first.video_id]
        if mismatched:
            raise ValueError(
                "A Gemini batch must contain one video only; "
                f"expected {first.video_id}, found {sorted(set(mismatched))}"
            )
        return first
