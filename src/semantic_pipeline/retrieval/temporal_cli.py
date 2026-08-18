"""Run same-video temporal event retrieval from the local pilot corpus."""

from __future__ import annotations

import argparse
import json
import os
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
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query  # noqa: E402
from semantic_pipeline.retrieval.qwen_video_verifier import QwenTemporalVerifier  # noqa: E402
from semantic_pipeline.retrieval.dense_motion_verifier import DenseMotionTemporalVerifier  # noqa: E402
from semantic_pipeline.retrieval.elasticsearch_backend import (  # noqa: E402
    DEFAULT_ELASTICSEARCH_URL,
    create_client,
)
from semantic_pipeline.retrieval.hierarchical_index_definition import (  # noqa: E402
    SEGMENT_INDEX_NAME,
    VIDEO_INDEX_NAME,
)
from semantic_pipeline.retrieval.temporal_video_selector import ElasticsearchVideoSelector  # noqa: E402


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
        "--video-selector",
        choices=("elasticsearch", "exhaustive"),
        default="elasticsearch",
        help="Select a video in Elasticsearch first, or use the legacy all-video event scan",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL),
    )
    parser.add_argument("--video-index", default=VIDEO_INDEX_NAME)
    parser.add_argument("--segment-index", default=SEGMENT_INDEX_NAME)
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
        parsed = parse_temporal_query(args.query)
        query_parser = None if args.without_gemini_query_parser else GeminiTemporalQueryParser()
        if query_parser is not None:
            parsed = query_parser.parse(parsed)

        selected_video_ids = list(args.video_id)
        video_selection: dict | None = None
        if not selected_video_ids and args.video_selector == "elasticsearch":
            video_selection = ElasticsearchVideoSelector(
                create_client(args.elasticsearch_url),
                video_index=args.video_index,
                segment_index=args.segment_index,
            ).select(
                parsed,
                batch_ids=args.batch_id,
                top_k=args.top_k_videos,
            )
            selected_video_id = video_selection.get("selected_video_id")
            if not selected_video_id:
                print(
                    json.dumps(
                        {
                            "query": args.query,
                            "mode": "temporal_event_search",
                            "video_selection": video_selection,
                            "selected_video": None,
                            "videos": [],
                            "events": [],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            # Resolve E1..En only after the batch-wide selector has chosen one
            # video.  The remaining candidates stay in video_selection for
            # manual review without triggering frame/event scans.
            selected_video_ids = [str(selected_video_id)]
        elif selected_video_ids:
            video_selection = {
                "mode": "explicit_video_ids",
                "selected_video_id": selected_video_ids[0],
                "candidates": [{"video_id": item} for item in selected_video_ids],
            }

        corpora = discover_temporal_corpus(
            output_root=args.output_root,
            caption_dir=args.caption_dir,
            asr_dir=args.asr_dir,
            map_dir=args.map_dir,
            keyframe_dir=args.keyframe_dir,
            batch_ids=args.batch_id,
            video_ids=selected_video_ids,
        )
        temporal_verifier = (
            QwenTemporalVerifier()
            if args.qwen_verify
            else DenseMotionTemporalVerifier()
            if args.dense_verify
            else None
        )
        result = TemporalEventSearch(
            corpora,
            # Gemini already ran before video selection. Reusing the parsed
            # object prevents a second API request in the event phase.
            query_parser=None,
            temporal_verifier=temporal_verifier,
            video_dir=args.video_dir if temporal_verifier else None,
            verifier_candidates=args.qwen_candidate_events,
        ).search(parsed, top_k_videos=args.top_k_videos)
        result["query"] = args.query
        result["query_parsing"] = {
            "mode": getattr(query_parser, "mode", "deterministic"),
            "model": getattr(query_parser, "model", None),
        }
        if video_selection is not None:
            result["video_selection"] = video_selection
    except Exception as exc:
        print(f"temporal retrieval failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
