"""Lazy Qwen3-VL-Embedding adapter.

Imports for the optional Qwen stack happen only when the encoder is created,
so importing this package never changes the existing CLIP runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import QwenEmbeddingConfig


class QwenDependencyError(RuntimeError):
    """Raised when the optional Qwen environment is not installed correctly."""


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("QWEN3_VL_EMBED_DEVICE=cuda but CUDA is unavailable")
    return requested


class Qwen3VLEmbeddingEncoder:
    """Encode Qwen3-VL text and images with one consistent configuration."""

    def __init__(self, config: QwenEmbeddingConfig):
        try:
            import torch
            import qwen_vl_utils  # noqa: F401 - required by Qwen's official wrapper
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise QwenDependencyError(
                "Qwen pilot dependencies are missing. Install "
                "requirements-qwen3-vl-embedding.txt in the experimental env."
            ) from exc

        self.config = config
        self.torch = torch
        self.device = _resolve_device(config.device, torch)
        model_kwargs: dict[str, Any] = {}
        if config.dtype != "auto":
            try:
                model_kwargs["dtype"] = getattr(torch, config.dtype)
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported QWEN3_VL_EMBED_DTYPE={config.dtype!r}; "
                    "use auto, float16, bfloat16, or float32"
                ) from exc

        model_dir = Path(config.model_id)
        wrapper_path = model_dir / "scripts" / "qwen3_vl_embedding.py"
        if not wrapper_path.is_file():
            raise QwenDependencyError(
                "The official Qwen embedding wrapper is missing at "
                f"{wrapper_path}. Download scripts/qwen3_vl_embedding.py "
                "alongside the local model files."
            )
        module_spec = importlib.util.spec_from_file_location(
            "qwen3_vl_embedding_official", wrapper_path
        )
        if module_spec is None or module_spec.loader is None:
            raise QwenDependencyError(f"Cannot load Qwen wrapper: {wrapper_path}")
        module = importlib.util.module_from_spec(module_spec)
        # Transformers introspects the defining module of custom model classes
        # while constructing them. Register the dynamically loaded wrapper just
        # like Python's normal import machinery does before executing it.
        sys.modules[module_spec.name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_spec.name, None)
            raise

        # Load the repository's Qwen3VLForEmbedding directly instead of its
        # helper constructor: that constructor always selects CUDA, while a
        # text-only local query must be able to use CPU safely on an 8 GB GPU.
        # This class preserves the checkpoint's ``model.*`` key prefix; the
        # generic SentenceTransformer/AutoModel path does not.
        self.model = module.Qwen3VLForEmbedding.from_pretrained(
            config.model_id, trust_remote_code=True, **model_kwargs
        ).to(self.device)
        self.model.eval()
        self.embedder = object.__new__(module.Qwen3VLEmbedder)
        self.embedder.model = self.model
        self.embedder.processor = module.Qwen3VLProcessor.from_pretrained(
            config.model_id, padding_side="right"
        )
        self.embedder.max_length = module.MAX_LENGTH
        self.embedder.min_pixels = module.MIN_PIXELS
        self.embedder.max_pixels = module.MAX_PIXELS
        self.embedder.total_pixels = module.MAX_TOTAL_PIXELS
        self.embedder.fps = module.FPS
        self.embedder.num_frames = module.MAX_FRAMES
        self.embedder.max_frames = module.MAX_FRAMES
        self.embedder.default_instruction = "Represent the user's input."

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("Qwen returned a zero or non-finite embedding")
        return values / norms

    def _encode(self, inputs: Sequence[dict[str, Any]]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(inputs), self.config.batch_size):
            batch = list(inputs[start : start + self.config.batch_size])
            embeddings = self.embedder.process(batch)
            chunks.append(embeddings.detach().float().cpu().numpy())
        normalized = self._normalize(np.concatenate(chunks, axis=0))
        if self.config.dimension != 2048:
            # Qwen's embedding head is trained with Matryoshka-compatible
            # dimensions. Truncate first, then normalize again for cosine.
            normalized = self._normalize(normalized[:, : self.config.dimension])
        if normalized.shape[1] != self.config.dimension:
            raise ValueError(
                f"Expected Qwen dimension {self.config.dimension}, "
                f"got {normalized.shape[1]}. Set QWEN3_VL_EMBED_DIMENSION "
                "to the model's configured output dimension."
            )
        return normalized

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        # Keep only one inference batch of decoded images in host memory. This
        # matters for long videos in the full Kaggle run: reading every frame
        # before the first forward pass can consume several gigabytes of RAM.
        chunks: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.config.batch_size):
            images: list[Image.Image] = []
            try:
                for path in image_paths[start : start + self.config.batch_size]:
                    images.append(Image.open(path).convert("RGB"))
                chunks.append(self._encode([{"image": image} for image in images]))
            finally:
                for image in images:
                    image.close()
        if not chunks:
            raise ValueError("Cannot encode an empty image sequence")
        return np.concatenate(chunks, axis=0)

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> np.ndarray:
        prompt = instruction or self.config.query_instruction
        return self._encode([{"text": text, "instruction": prompt} for text in texts])
