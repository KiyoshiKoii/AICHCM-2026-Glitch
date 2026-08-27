"""Explicit opt-in Qdrant indexer for Qwen pilot artifacts."""

from __future__ import annotations

import argparse
import os
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .artifacts import load_image_embeddings
from .config import QwenEmbeddingConfig


def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, timeout=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")))
    # Keep the pilot's local database separate from the legacy visual pipeline.
    return QdrantClient(path=str(QwenEmbeddingConfig.from_env().feature_dir.parent / "qdrant_qwen3_vl_2b"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Video ID to index. Repeat this option for multiple videos.",
    )
    selection.add_argument(
        "--all-videos",
        action="store_true",
        help="Index every complete Qwen artifact under --feature-dir.",
    )
    parser.add_argument("--collection", default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--recreate", action="store_true", help="Explicitly replace only the pilot collection")
    return parser


def _discover_video_ids(feature_dir: Path) -> list[str]:
    pattern = re.compile(r"L\d+_V\d+")
    video_ids: list[str] = []
    for manifest_path in feature_dir.glob("*_manifest.json"):
        video_id = manifest_path.name.removesuffix("_manifest.json")
        if not pattern.fullmatch(video_id):
            continue
        if (feature_dir / f"{video_id}.npy").is_file() and (
            feature_dir / f"{video_id}_filenames.json"
        ).is_file():
            video_ids.append(video_id)
    return sorted(video_ids)


def _config_for_video(base: QwenEmbeddingConfig, video_id: str, args: argparse.Namespace) -> QwenEmbeddingConfig:
    return QwenEmbeddingConfig(
        model_id=base.model_id,
        model_revision=base.model_revision,
        video_id=video_id,
        keyframe_dir=base.keyframe_dir,
        feature_dir=base.feature_dir,
        dimension=args.dimension or base.dimension,
        batch_size=base.batch_size,
        device=base.device,
        dtype=base.dtype,
        query_instruction=base.query_instruction,
        collection_name=args.collection or base.collection_name,
    )


def main() -> None:
    args = build_parser().parse_args()
    base = QwenEmbeddingConfig.from_env()
    video_ids = _discover_video_ids(base.feature_dir) if args.all_videos else (args.video_id or ["L21_V001"])
    if not video_ids:
        raise RuntimeError(f"No complete Qwen artifacts found under {base.feature_dir}")
    configs = [_config_for_video(base, video_id, args) for video_id in video_ids]

    first_features, _, _ = load_image_embeddings(configs[0])
    client = _client()
    collection = configs[0].collection_name
    if client.collection_exists(collection_name=collection):
        if not args.recreate:
            raise RuntimeError(
                f"Collection {collection!r} already exists. Pass --recreate explicitly "
                "to replace the Qwen pilot collection."
            )
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=first_features.shape[1], distance=Distance.COSINE),
    )
    indexed_points = 0
    for config in configs:
        features, filenames, manifest = load_image_embeddings(config)
        if features.shape[1] != first_features.shape[1]:
            raise ValueError(
                f"Dimension mismatch for {config.video_id}: {features.shape[1]} vs {first_features.shape[1]}"
            )
        points = []
        for row, metadata in zip(features, filenames, strict=True):
            frame_name = metadata["filename"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"aic2026-qwen3/{config.video_id}/{frame_name}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=row.tolist(),
                    payload={
                        "video_id": config.video_id,
                        "frame_id": frame_name,
                        "frame_index": metadata["frame_index"],
                        "embedding_model": manifest["model_id"],
                    },
                )
            )
        client.upload_points(collection_name=collection, points=points)
        indexed_points += len(points)
        print(f"Indexed {len(points)} points for {config.video_id}.")
    info = client.get_collection(collection_name=collection)
    print(f"Indexed {indexed_points} points in {collection}; collection has {info.points_count} points.")
    client.close()


if __name__ == "__main__":
    main()
