"""CLI for the fixed L22_V001 video-understanding pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # Support ``python src/semantic_pipeline/.../cli.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from semantic_pipeline.video_understanding.pipeline import build_video
else:
    from .pipeline import build_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the fixed L22_V001 video-understanding pilot")
    parser.add_argument("--video-id", default="L22_V001")
    parser.add_argument("--caption-dir", type=Path, default=Path("data/metadata/caption"))
    parser.add_argument("--asr-dir", type=Path, default=Path("data/metadata/metadata_asr"))
    parser.add_argument("--map-dir", type=Path, default=Path("data/map-keyframes"))
    parser.add_argument("--keyframe-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/video_understanding"),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Gemini for evidence-grounded story and video summaries",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail instead of falling back if Gemini is unavailable",
    )
    parser.add_argument(
        "--force-publish",
        action="store_true",
        help="Publish a passing candidate even when its score is lower than the current pilot",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_video(
            video_id=args.video_id,
            caption_dir=args.caption_dir,
            asr_dir=args.asr_dir,
            map_dir=args.map_dir,
            keyframe_dir=args.keyframe_dir,
            output_root=args.output_root,
            use_llm=args.llm,
            require_llm=args.require_llm,
            force_publish=args.force_publish,
        )
    except Exception as exc:
        print(f"video-understanding failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
