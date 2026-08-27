"""Isolated Qwen3-VL-Embedding pilot utilities.

This package deliberately does not import or modify the production CLIP
pipeline.  It is intended for the L21_V001 pilot first and can be extended to
other videos after the embedding quality/latency benchmark is accepted.
"""

from .config import QwenEmbeddingConfig

__all__ = ["QwenEmbeddingConfig"]
