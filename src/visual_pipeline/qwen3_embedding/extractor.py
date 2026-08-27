"""CLI for extracting Qwen3-VL image embeddings for one video.

Example (do not run until the optional environment is installed):
    PYTHONPATH=src python -m visual_pipeline.qwen3_embedding.extractor \
      --video-id L21_V001
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .artifacts import save_image_embeddings
from .config import QwenEmbeddingConfig
from .encoder import Qwen3VLEmbeddingEncoder
from .keyframes import discover_keyframes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Video ID to process. Repeat this option for multiple videos.",
    )
    selection.add_argument(
        "--all-videos",
        action="store_true",
        help="Process every L##_V### keyframe directory under --keyframe-dir.",
    )
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--force", action="store_true", help="Recompute existing pilot artifacts")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed video and continue processing the remaining videos.",
    )
    return parser


def _discover_video_ids(keyframe_dir: Path) -> list[str]:
    pattern = re.compile(r"L\d+_V\d+")
    return sorted(
        path.name
        for path in keyframe_dir.iterdir()
        if path.is_dir() and pattern.fullmatch(path.name) and any(path.glob("*.jpg"))
    )


def _config_for_video(base: QwenEmbeddingConfig, video_id: str, args: argparse.Namespace) -> QwenEmbeddingConfig:
    return QwenEmbeddingConfig(
        model_id=base.model_id,
        model_revision=base.model_revision,
        video_id=video_id,
        keyframe_dir=base.keyframe_dir,
        feature_dir=base.feature_dir,
        dimension=args.dimension or base.dimension,
        batch_size=args.batch_size or base.batch_size,
        device=args.device or base.device,
        dtype=args.dtype or base.dtype,
        query_instruction=base.query_instruction,
        collection_name=base.collection_name,
    )


def main() -> None:
    args = build_parser().parse_args()
    base = QwenEmbeddingConfig.from_env()
    video_ids = _discover_video_ids(base.keyframe_dir) if args.all_videos else (args.video_id or ["L21_V001"])
    if not video_ids:
        raise RuntimeError(f"No keyframe video directories found under {base.keyframe_dir}")

    configs = [_config_for_video(base, video_id, args) for video_id in video_ids]
    pending: list[QwenEmbeddingConfig] = []
    for config in configs:
        artifact_paths = (config.feature_path(), config.filenames_path(), config.manifest_path())
        if not args.force and all(path.is_file() for path in artifact_paths):
            print(f"Artifacts already exist for {config.video_id}; skipping.")
        else:
            pending.append(config)
    if not pending:
        print("No videos need extraction.")
        return

    print(f"Loading Qwen once for {len(pending)} video(s), batch_size={pending[0].batch_size}.")
    encoder = Qwen3VLEmbeddingEncoder(pending[0])
    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for position, config in enumerate(pending, 1):
        try:
            keyframes = discover_keyframes(config.keyframe_video_dir())
            print(f"[{position}/{len(pending)}] Preparing {len(keyframes)} keyframes for {config.video_id}.")
            started = time.perf_counter()
            embeddings = encoder.encode_images([frame.path for frame in keyframes])
            manifest = save_image_embeddings(config, keyframes, embeddings)
            elapsed = time.perf_counter() - started
            successes.append(
                {
                    "video_id": config.video_id,
                    "keyframe_count": manifest["keyframe_count"],
                    "seconds": round(elapsed, 3),
                }
            )
            print(
                f"Saved {manifest['keyframe_count']} vectors with shape {embeddings.shape} "
                f"to {config.feature_path()} in {elapsed:.2f}s."
            )
        except Exception as exc:
            failures.append({"video_id": config.video_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {config.video_id}: {failures[-1]['error']}")
            if not args.continue_on_error:
                raise

    report_path = pending[0].feature_dir / "qwen3_extraction_report.json"
    report_path.write_text(
        json.dumps(
            {
                "requested_video_ids": video_ids,
                "skipped_video_ids": [config.video_id for config in configs if config not in pending],
                "succeeded": successes,
                "failed": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Finished: succeeded={len(successes)}, failed={len(failures)}. Report: {report_path}")


if __name__ == "__main__":
    main()
