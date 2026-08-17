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
from semantic_pipeline.retrieval.temporal_query_expander import GeminiTemporalQueryParser  # noqa: E402
from semantic_pipeline.retrieval.qwen_video_verifier import QwenTemporalVerifier  # noqa: E402
from semantic_pipeline.retrieval.dense_motion_verifier import DenseMotionTemporalVerifier  # noqa: E402


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
    parser.add_argument(
        "--qwen-verify",
        action="store_true",
        help="Use local Qwen2.5-VL only to refine event candidates in the selected raw video",
    )
    parser.add_argument(
        "--dense-verify",
        action="store_true",
        help="Use batched CLIP plus frame motion to scan selected raw-video event windows",
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--qwen-candidate-events", type=int, default=3)
    parser.add_argument(
        "--without-gemini-query-parser",
        action="store_true",
        help="Use only explicit query terms; useful for deterministic retrieval debugging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.qwen_verify and args.dense_verify:
        print("temporal retrieval failed: choose only one of --qwen-verify or --dense-verify")
        return 2
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
        query_parser = None if args.without_gemini_query_parser else GeminiTemporalQueryParser()
        temporal_verifier = (
            QwenTemporalVerifier()
            if args.qwen_verify
            else DenseMotionTemporalVerifier()
            if args.dense_verify
            else None
        )
        result = TemporalEventSearch(
            corpora,
            query_parser=query_parser,
            temporal_verifier=temporal_verifier,
            video_dir=args.video_dir if temporal_verifier else None,
            verifier_candidates=args.qwen_candidate_events,
        ).search(
            args.query,
            top_k_videos=args.top_k_videos,
        )
    except Exception as exc:
        print(f"temporal retrieval failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
