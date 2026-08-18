"""Run the existing dense boundary verifier on narrow InternVideo2 regions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_pipeline.retrieval.dense_motion_verifier import (
    DenseMotionConfig,
    DenseMotionTemporalVerifier,
)
from semantic_pipeline.retrieval.qwen_video_verifier import TemporalVerificationRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--pre-seconds", type=float, default=1.0)
    parser.add_argument("--post-seconds", type=float, default=8.0)
    args = parser.parse_args()

    region_results = json.loads(args.regions.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    specs = {event["event_id"]: event for event in spec["events"]}
    verifier = DenseMotionTemporalVerifier(
        DenseMotionConfig(
            sample_fps=args.fps,
            max_frames=96,
            padding_ms=0,
            batch_size=64,
            semantic_weight=0.72,
            motion_weight=0.28,
        )
    )

    started = time.perf_counter()
    outputs = []
    for event in region_results["results"]:
        region = event["regions_by_score"][0]
        event_spec = specs[event["event_id"]]
        boundary = float(region.get("boundary_seconds", region["leading_edge_seconds"]))
        selection_rule = event_spec.get("selection_rule", "earliest_true")
        if selection_rule == "latest_true":
            start_seconds = max(0.0, boundary - args.post_seconds)
            end_seconds = boundary + args.pre_seconds
        else:
            start_seconds = max(0.0, boundary - args.pre_seconds)
            end_seconds = boundary + args.post_seconds
        target_predicates = event_spec.get("target_predicates") or event_spec.get("after_prompts")
        if not target_predicates:
            raise ValueError(f"{event['event_id']} has no target predicate")
        request = TemporalVerificationRequest(
            video_id=args.video.stem,
            event_id=event["event_id"],
            event_text=target_predicates[0],
            required_anchor=event_spec.get("required_anchor", "action_start"),
            video_path=args.video,
            start_ms=round(start_seconds * 1_000),
            end_ms=round(end_seconds * 1_000),
        )
        event_started = time.perf_counter()
        result = verifier.verify(request)
        event_type = event_spec.get("event_type", "action_event")
        verifier_required_types = {
            "first_contact",
            "state_attainment",
            "action_completion",
            "compound_event",
        }
        needs_vlm_verification = (
            event_type in verifier_required_types or result.confidence < 0.70
        )
        outputs.append(
            {
                "event_id": event["event_id"],
                "event_type": event_type,
                "selection_rule": selection_rule,
                "region_boundary_seconds": boundary,
                "region_peak_seconds": region["peak_seconds"],
                "dense_window_seconds": [start_seconds, end_seconds],
                "refined_timestamp_seconds": (
                    round(result.timestamp_ms / 1_000, 3)
                    if result.timestamp_ms is not None
                    else None
                ),
                "confidence": round(result.confidence, 6),
                "needs_vlm_verification": needs_vlm_verification,
                "reason": result.reason,
                "elapsed_seconds": round(time.perf_counter() - event_started, 4),
            }
        )

    output = {
        "video": str(args.video),
        "fps": args.fps,
        "pre_seconds": args.pre_seconds,
        "post_seconds": args.post_seconds,
        "total_seconds": round(time.perf_counter() - started, 4),
        "results": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
