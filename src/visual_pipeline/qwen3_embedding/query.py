"""Text-to-keyframe query CLI for the isolated Qwen pilot."""

from __future__ import annotations

import argparse
import asyncio
import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import QwenEmbeddingConfig
from .encoder import Qwen3VLEmbeddingEncoder


def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, timeout=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")))
    return QdrantClient(path=str(QwenEmbeddingConfig.from_env().feature_dir.parent / "qdrant_qwen3_vl_2b"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--gemini-parse",
        action="store_true",
        help="Use the existing Gemini query parser and fuse its visual constraints.",
    )
    parser.add_argument(
        "--show-query-plan",
        action="store_true",
        help="Print the parser plan when --gemini-parse is enabled.",
    )
    return parser


def _interaction_prompt(interaction: object) -> str:
    return " ".join(
        (
            str(getattr(interaction, "subject_english_phrase")),
            str(getattr(interaction, "action_english_phrase")),
            str(getattr(interaction, "object_english_phrase")),
        )
    )


def _parsed_query_vector(encoder: Qwen3VLEmbeddingEncoder, query: str, *, show_plan: bool) -> list[float]:
    """Embed a Gemini query plan without allowing generic keywords to dilute it.

    The parser's compact ``visual_prompt`` carries most visual evidence.  Its
    subject-action-object and object rows add small, bounded boosts for details
    that distinguish visually similar scenes.  Semantic keywords are designed
    for caption/BM25 recall, so they are intentionally not added to the dense
    Qwen vector fusion.
    """

    try:
        from backend.config import Settings
        from backend.services.query_analyzer import GeminiQueryParser
    except ImportError as exc:  # pragma: no cover - optional backend dependency
        raise RuntimeError(
            "--gemini-parse requires the backend parser dependencies and configured Gemini API key"
        ) from exc

    settings = Settings()
    parsed = asyncio.run(
        GeminiQueryParser(settings.gemini_api_key, settings.gemini_query_model).parse(query)
    )
    if show_plan:
        print("Gemini visual prompt:", parsed.visual_prompt)
        print("Gemini interactions:", [_interaction_prompt(item) for item in parsed.interaction_queries])
        print("Gemini objects:", [item.english_phrase for item in parsed.object_queries])

    weighted_prompts: list[tuple[str, float]] = [(parsed.visual_prompt, 0.70)]
    interactions = [_interaction_prompt(item) for item in parsed.interaction_queries]
    if interactions:
        weighted_prompts.extend((prompt, 0.20 / len(interactions)) for prompt in interactions)
    objects = [item.english_phrase for item in parsed.object_queries]
    if objects:
        weighted_prompts.extend((prompt, 0.10 / len(objects)) for prompt in objects)

    prompts = [prompt for prompt, _ in weighted_prompts]
    weights = np.asarray([weight for _, weight in weighted_prompts], dtype=np.float32)
    vectors = encoder.encode_texts(prompts)
    fused = np.sum(vectors * weights[:, None], axis=0)
    norm = float(np.linalg.norm(fused))
    if norm <= 0 or not np.isfinite(norm):
        raise RuntimeError("Gemini parser fusion produced an invalid Qwen query vector")
    return (fused / norm).tolist()


def main() -> None:
    args = build_parser().parse_args()
    base = QwenEmbeddingConfig.from_env(video_id=args.video_id or "L21_V001")
    config = QwenEmbeddingConfig(
        model_id=base.model_id,
        model_revision=base.model_revision,
        video_id=base.video_id,
        keyframe_dir=base.keyframe_dir,
        feature_dir=base.feature_dir,
        dimension=args.dimension or base.dimension,
        batch_size=args.batch_size or base.batch_size,
        device=args.device or base.device,
        dtype=args.dtype or base.dtype,
        query_instruction=base.query_instruction,
        collection_name=base.collection_name,
    )
    encoder = Qwen3VLEmbeddingEncoder(config)
    if args.gemini_parse:
        query_vector = _parsed_query_vector(
            encoder,
            args.query,
            show_plan=args.show_query_plan,
        )
    else:
        query_vector = encoder.encode_texts([args.query], instruction=args.instruction)[0].tolist()

    query_filter = None
    if args.video_id:
        query_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=args.video_id))]
        )
    client = _client()
    results = client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=args.limit,
        with_payload=True,
    ).points
    for rank, result in enumerate(results, 1):
        print(
            f"{rank:02d} score={result.score:.6f} "
            f"{result.payload.get('video_id')} {result.payload.get('frame_id')}"
        )
    client.close()


if __name__ == "__main__":
    main()
