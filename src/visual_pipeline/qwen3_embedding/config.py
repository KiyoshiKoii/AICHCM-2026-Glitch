"""Configuration for the Qwen3-VL visual retrieval pipeline.

All model/runtime knobs are environment-overridable. Qwen embeddings and their
collection are separate from retained legacy CLIP artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYFRAME_DIR = REPO_ROOT / "data" / "keyframes"
DEFAULT_FEATURE_DIR = REPO_ROOT / "data" / "npy_features_qwen3_vl_2b"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_COLLECTION = "kis_images_qwen3_vl_2b_test"
DEFAULT_QUERY_INSTRUCTION = "Retrieve relevant video keyframes for the user's query."


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class QwenEmbeddingConfig:
    """Runtime settings for one Qwen image/text embedding workload."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    video_id: str = "L21_V001"
    keyframe_dir: Path = DEFAULT_KEYFRAME_DIR
    feature_dir: Path = DEFAULT_FEATURE_DIR
    dimension: int = 2048
    batch_size: int = 4
    device: str = "auto"
    dtype: str = "auto"
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    collection_name: str = DEFAULT_COLLECTION

    @classmethod
    def from_env(cls, *, video_id: str | None = None) -> "QwenEmbeddingConfig":
        """Build pilot settings from environment variables.

        ``video_id`` is intentionally an explicit CLI argument in normal use;
        the environment fallback only makes notebook experiments convenient.
        """

        keyframe_dir = Path(
            _env("QWEN3_VL_EMBED_KEYFRAME_DIR", str(DEFAULT_KEYFRAME_DIR))
        )
        feature_dir = Path(
            _env("QWEN3_VL_EMBED_FEATURE_DIR", str(DEFAULT_FEATURE_DIR))
        )
        return cls(
            model_id=_env("QWEN3_VL_EMBED_MODEL", DEFAULT_MODEL_ID),
            model_revision=os.getenv("QWEN3_VL_EMBED_REVISION") or None,
            video_id=video_id or _env("QWEN3_VL_EMBED_VIDEO_ID", "L21_V001"),
            keyframe_dir=keyframe_dir,
            feature_dir=feature_dir,
            dimension=_env_int("QWEN3_VL_EMBED_DIMENSION", 2048),
            batch_size=_env_int("QWEN3_VL_EMBED_BATCH_SIZE", 4),
            device=_env("QWEN3_VL_EMBED_DEVICE", "auto"),
            dtype=_env("QWEN3_VL_EMBED_DTYPE", "auto"),
            query_instruction=_env(
                "QWEN3_VL_EMBED_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION
            ),
            collection_name=_env("QWEN3_VL_EMBED_COLLECTION", DEFAULT_COLLECTION),
        )

    def keyframe_video_dir(self) -> Path:
        return self.keyframe_dir / self.video_id

    def feature_path(self) -> Path:
        return self.feature_dir / f"{self.video_id}.npy"

    def filenames_path(self) -> Path:
        return self.feature_dir / f"{self.video_id}_filenames.json"

    def manifest_path(self) -> Path:
        return self.feature_dir / f"{self.video_id}_manifest.json"
