"""CLI for building video-understanding artifacts for any BTC video."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # Support ``python src/semantic_pipeline/.../cli.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from semantic_pipeline.video_understanding.pipeline import build_video
else:
    from .pipeline import build_video


BATCH_ID_RE = re.compile(r"^L\d{2}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build video-understanding artifacts by BTC video or batch"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--video-id",
        help="BTC video id in Lxx_Vyyy format, for example L22_V001",
    )
    selection.add_argument(
        "--batch-id",
        help="Build every video discovered under a batch such as L22",
    )
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


def discover_batch_video_ids(caption_dir: Path, batch_id: str) -> list[str]:
    if BATCH_ID_RE.fullmatch(batch_id) is None:
        raise ValueError("batch_id must use the BTC format Lxx, for example L22")
    batch_dir = caption_dir / batch_id
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Caption batch directory does not exist: {batch_dir}")
    video_ids = sorted(
        path.stem
        for path in batch_dir.glob(f"{batch_id}_V*.json")
        if re.fullmatch(rf"{re.escape(batch_id)}_V\d{{3}}", path.stem)
    )
    if not video_ids:
        raise FileNotFoundError(f"No BTC caption artifacts found in: {batch_dir}")
    return video_ids


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        video_ids = (
            [args.video_id]
            if args.video_id
            else discover_batch_video_ids(args.caption_dir, args.batch_id)
        )
        results = []
        for video_id in video_ids:
            if len(video_ids) > 1:
                print(f"Building {video_id}...", file=sys.stderr)
            results.append(
                build_video(
                    video_id=video_id,
                    caption_dir=args.caption_dir,
                    asr_dir=args.asr_dir,
                    map_dir=args.map_dir,
                    keyframe_dir=args.keyframe_dir,
                    output_root=args.output_root,
                    use_llm=args.llm,
                    require_llm=args.require_llm,
                    force_publish=args.force_publish,
                )
            )
    except Exception as exc:
        print(f"video-understanding failed: {exc}")
        return 2
    payload = (
        results[0]
        if len(results) == 1
        else {"batch_id": args.batch_id, "results": results}
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
