"""Utilities for preparing text inputs for CLIP retrieval."""

from __future__ import annotations

from typing import Any


DEFAULT_CLIP_CONTEXT_LENGTH = 77
DEFAULT_CLIP_TOKEN_OVERLAP = 12


def clip_context_length(model: Any, processor: Any) -> int:
    """Return the effective token limit advertised by the CLIP model."""

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    model_limit = getattr(text_config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        return model_limit

    tokenizer_limit = getattr(getattr(processor, "tokenizer", None), "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return tokenizer_limit

    return DEFAULT_CLIP_CONTEXT_LENGTH


def prepare_clip_text_inputs(
    processor: Any,
    model: Any,
    prompts: list[str],
    device: str,
) -> Any:
    """Tokenize every part of long prompts into overlapping CLIP windows."""

    context_length = clip_context_length(model, processor)
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        stride=min(DEFAULT_CLIP_TOKEN_OVERLAP, max(0, context_length - 3)),
    )
    # This tokenizer bookkeeping tensor is not accepted by CLIPModel.  The
    # remaining input rows are all overlapping windows and are intentionally
    # encoded/averaged by the visual search endpoint.
    inputs.pop("overflow_to_sample_mapping", None)
    return inputs.to(device)
