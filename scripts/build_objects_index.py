"""Build per-video Parquet indexes from BTC ``objects/`` JSON files.

Examples:
    python scripts/build_objects_index.py --videos L21_V001,L22_V001
    python scripts/build_objects_index.py --videos L21,L22 --resume
    python scripts/build_objects_index.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import paths  # noqa: E402
from semantic_pipeline.btc_objects import build_video_objects_index  # noqa: E402


def select_videos(value: str | None) -> list[str]:
    available = [
        item.name for item in paths.OBJECTS.iterdir() if item.is_dir()
    ] if paths.OBJECTS.is_dir() else []
    available.sort()
    if not value:
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    selected: list[str] = []
    for item in requested:
        if "_V" in item:
            selected.append(item)
        else:
            selected.extend(video for video in available if video.startswith(f"{item}_"))
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown/missing BTC object videos: {unknown}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        help="Comma-separated video IDs or groups (for example L21,L22)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        videos = select_videos(args.videos)
    except ValueError as exc:
        parser.error(str(exc))
    if not videos:
        parser.error(f"No BTC object directories found under {paths.OBJECTS}")

    reports = []
    skipped = 0
    for position, video_id in enumerate(videos, start=1):
        output = paths.objects_index(video_id)
        if args.resume and output.is_file():
            skipped += 1
            continue
        report = build_video_objects_index(video_id)
        reports.append(report)
        print(
            f"[{position}/{len(videos)}] {video_id}: {report['frames']} frames, "
            f"raw={report['raw_per_frame']:.1f}, "
            f">=threshold={report['above_threshold_per_frame']:.1f}, "
            f"after={report['filtered_per_frame']:.1f} detections/frame"
        )

    frames = sum(item["frames"] for item in reports)
    summary = {
        "selected_videos": len(videos),
        "built_videos": len(reports),
        "skipped_videos": skipped,
        "frames": frames,
        "raw_per_frame": (
            sum(item["raw_per_frame"] * item["frames"] for item in reports) / frames
            if frames else 0.0
        ),
        "above_threshold_per_frame": (
            sum(
                item["above_threshold_per_frame"] * item["frames"]
                for item in reports
            ) / frames
            if frames else 0.0
        ),
        "filtered_per_frame": (
            sum(item["filtered_per_frame"] * item["frames"] for item in reports)
            / frames
            if frames else 0.0
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
