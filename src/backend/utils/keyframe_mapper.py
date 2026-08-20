import os
import csv
from functools import lru_cache
from typing import TypedDict

# Path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAP_DIR = os.path.join(BASE_DIR, "data", "map-keyframes")


class KeyframePosition(TypedDict):
    frame_index: int
    timestamp_ms: int
    fps: float


@lru_cache(maxsize=1024)
def _load_video_timeline(video_name: str) -> dict[int, KeyframePosition]:
    """Load keyframe ordinals and their positions in the source video."""

    csv_path = os.path.join(MAP_DIR, f"{video_name}.csv")
    timeline: dict[int, KeyframePosition] = {}
    if not os.path.exists(csv_path):
        return timeline

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                ordinal = int(row["n"])
                timeline[ordinal] = {
                    "frame_index": int(row["frame_idx"]),
                    "timestamp_ms": round(float(row["pts_time"]) * 1_000),
                    "fps": float(row["fps"]),
                }
    except Exception as exc:
        print(f"Error loading {csv_path}: {exc}")

    return timeline

@lru_cache(maxsize=1024)
def _load_video_mapping(video_name: str) -> dict[int, int]:
    """
    Loads the keyframe mapping for a specific video.
    Returns a dict mapping `n` to `frame_idx`.
    """
    return {
        ordinal: position["frame_index"]
        for ordinal, position in _load_video_timeline(video_name).items()
    }

def get_true_frame_idx(video_name: str, n: int) -> int | None:
    """
    Gets the true original frame index for a given video and keyframe `n`.
    Returns None if mapping doesn't exist.
    """
    mapping = _load_video_mapping(video_name)
    return mapping.get(n)


def get_keyframe_position(video_name: str, n: int) -> KeyframePosition | None:
    """Return native frame, timestamp and FPS for a keyframe ordinal."""

    return _load_video_timeline(video_name).get(n)


def get_keyframe_ordinals(video_name: str) -> tuple[int, ...]:
    """Return available keyframe ordinals in timeline order."""

    return tuple(sorted(_load_video_timeline(video_name)))
