"""Run reproducible local hierarchical retrieval for the fixed L22 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from semantic_pipeline.retrieval.hierarchical_video_search import (
        HierarchicalVideoSearch,
        build_hierarchical_documents,
    )
else:
    from .hierarchical_video_search import HierarchicalVideoSearch, build_hierarchical_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the fixed L22_V001 video-understanding pilot")
    parser.add_argument("query", help="Vietnamese or English query")
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/processed/video_understanding/L22/L22_V001/pilot"))
    parser.add_argument("--caption-path", type=Path, default=Path("data/metadata/caption/L22/L22_V001.json"))
    parser.add_argument("--asr-path", type=Path, default=Path("data/metadata/metadata_asr/L22_V001.json"))
    parser.add_argument("--map-path", type=Path, default=Path("data/map-keyframes/L22_V001.csv"))
    parser.add_argument("--keyframe-dir", type=Path, default=Path("data/keyframes/L22_V001"))
    parser.add_argument("--top-segments", type=int, default=3)
    parser.add_argument("--top-frames", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    documents = build_hierarchical_documents(
        pilot_dir=args.pilot_dir,
        caption_path=args.caption_path,
        asr_path=args.asr_path,
        map_path=args.map_path,
        keyframe_dir=args.keyframe_dir,
    )
    result = HierarchicalVideoSearch(documents).search(
        args.query,
        top_segments=args.top_segments,
        top_frames=args.top_frames,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
