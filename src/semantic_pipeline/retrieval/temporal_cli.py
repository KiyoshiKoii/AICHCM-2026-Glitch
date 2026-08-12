"""Run same-video temporal event retrieval from the local pilot corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_pipeline.retrieval.temporal_event_search import (  # noqa: E402
    TemporalEventSearch,
    discover_temporal_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search ordered BTC E1..En events in one video")
    parser.add_argument("query", help="Shared video description and E1..En event lines")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/video_understanding"))
    parser.add_argument("--caption-dir", type=Path, default=Path("data/metadata/caption"))
    parser.add_argument("--asr-dir", type=Path, default=Path("data/metadata/metadata_asr"))
    parser.add_argument("--map-dir", type=Path, default=Path("data/map-keyframes"))
    parser.add_argument("--keyframe-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument("--batch-id", action="append", default=[])
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--top-k-videos", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        corpora = discover_temporal_corpus(
            output_root=args.output_root,
            caption_dir=args.caption_dir,
            asr_dir=args.asr_dir,
            map_dir=args.map_dir,
            keyframe_dir=args.keyframe_dir,
            batch_ids=args.batch_id,
            video_ids=args.video_id,
        )
        result = TemporalEventSearch(corpora).search(args.query, top_k_videos=args.top_k_videos)
    except Exception as exc:
        print(f"temporal retrieval failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
