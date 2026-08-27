"""Persistence and validation for Qwen pilot artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import QwenEmbeddingConfig
from .keyframes import Keyframe


def save_image_embeddings(
    config: QwenEmbeddingConfig,
    keyframes: list[Keyframe],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Embeddings must be rank-2, got shape {values.shape}")
    if values.shape[0] != len(keyframes):
        raise ValueError(
            f"Embedding/keyframe mismatch: {values.shape[0]} vs {len(keyframes)}"
        )
    if values.shape[1] != config.dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {config.dimension}, got {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain NaN or infinity")

    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings must be L2-normalized before persistence")

    config.feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.feature_path(), values)
    config.filenames_path().write_text(
        json.dumps(
            [
                {
                    "filename": frame.filename,
                    "frame_index": frame.frame_index,
                }
                for frame in keyframes
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "video_id": config.video_id,
        "dimension": config.dimension,
        "normalized": True,
        "query_instruction": config.query_instruction,
        "batch_size": config.batch_size,
        "device": config.device,
        "dtype": config.dtype,
        "keyframe_count": len(keyframes),
        "feature_file": config.feature_path().name,
        "filenames_file": config.filenames_path().name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config.manifest_path().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_image_embeddings(
    config: QwenEmbeddingConfig,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Load and validate an artifact before it is sent to Qdrant."""

    features = np.load(config.feature_path())
    filenames = json.loads(config.filenames_path().read_text(encoding="utf-8"))
    manifest = json.loads(config.manifest_path().read_text(encoding="utf-8"))
    if features.shape != (len(filenames), config.dimension):
        raise ValueError(
            f"Artifact shape mismatch: features={features.shape}, "
            f"filenames={len(filenames)}, dimension={config.dimension}"
        )
    if manifest.get("video_id") != config.video_id:
        raise ValueError("Artifact video_id does not match requested video")
    if not np.isfinite(features).all():
        raise ValueError("Artifact contains non-finite values")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-3):
        raise ValueError("Artifact vectors are not L2-normalized")
    return features.astype(np.float32), filenames, manifest
