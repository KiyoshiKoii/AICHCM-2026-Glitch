"""Load the global keyframe deduplication map used before Gemini extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .frame_id import frame_id_from_path


DEFAULT_GLOBAL_FILTER_RESULTS_PATH = Path("data/global_filter_results.csv")


@dataclass(frozen=True)
class GlobalFrameDeduplication:
    """Map each frame ID to the representative frame in the same video."""

    representative_by_frame: dict[str, str]

    def representative_for(self, frame_id: str) -> str:
        """Treat unlisted frames as unique so a partial CSV never drops data."""
        return self.representative_by_frame.get(frame_id, frame_id)


def _frame_id(video_name: str, frame_name: str) -> str:
    return frame_id_from_path(Path(video_name) / frame_name)


def load_global_frame_deduplication(path: str | Path) -> GlobalFrameDeduplication:
    """Read ``global_filter_results.csv`` and validate its representative links."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "video_name",
            "frame_name",
            "is_representative",
            "representative_frame",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} is missing required global-filter columns")

        representatives: dict[str, str] = {}
        for row in reader:
            video_name = (row["video_name"] or "").strip()
            frame_name = (row["frame_name"] or "").strip()
            representative_name = (row["representative_frame"] or "").strip()
            if not video_name or not frame_name or not representative_name:
                raise ValueError(f"{csv_path} contains an incomplete frame mapping")

            frame_id = _frame_id(video_name, frame_name)
            representative_id = _frame_id(video_name, representative_name)
            is_representative = (row["is_representative"] or "").strip().casefold()
            if is_representative not in {"true", "false"}:
                raise ValueError(f"{csv_path} has invalid is_representative for {frame_id}")
            if (is_representative == "true") != (frame_id == representative_id):
                raise ValueError(f"{csv_path} has inconsistent representative mapping for {frame_id}")
            if frame_id in representatives:
                raise ValueError(f"{csv_path} maps {frame_id} more than once")
            representatives[frame_id] = representative_id

    return GlobalFrameDeduplication(representative_by_frame=representatives)
