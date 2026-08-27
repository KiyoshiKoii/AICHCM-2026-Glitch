"""Qwen3-VL text-to-keyframe retrieval service.

CLIP artifacts remain on disk for historical comparison, but the serving path
uses only the separately indexed Qwen3-VL embeddings.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from threading import Lock

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchAny

try:
    from config import KEYFRAME_DIR, create_qdrant_client, qdrant_target_description
    from qwen3_embedding.config import QwenEmbeddingConfig
    from qwen3_embedding.encoder import Qwen3VLEmbeddingEncoder
except ImportError:
    from .config import KEYFRAME_DIR, create_qdrant_client, qdrant_target_description
    from .qwen3_embedding.config import QwenEmbeddingConfig
    from .qwen3_embedding.encoder import Qwen3VLEmbeddingEncoder


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


app = FastAPI(
    title="Qwen3-VL Visual Retrieval Internal API",
    description="Raw-query text-to-keyframe search over the Qwen3-VL Qdrant index.",
)

print(f"Connecting Qdrant at {qdrant_target_description()}")
client = create_qdrant_client()
print("Qdrant client is ready.")

_qwen_encoder: Qwen3VLEmbeddingEncoder | None = None
_qwen_encoder_lock = Lock()


def _qwen_runtime_config() -> QwenEmbeddingConfig:
    """Build Qwen serving config, preferring fp16 when CUDA is available."""

    base = QwenEmbeddingConfig.from_env()
    dtype = base.dtype
    if dtype == "auto" and torch.cuda.is_available() and base.device != "cpu":
        dtype = os.getenv("QWEN3_VL_SERVE_DTYPE", "float16")
    return QwenEmbeddingConfig(
        model_id=base.model_id,
        model_revision=base.model_revision,
        video_id=base.video_id,
        keyframe_dir=base.keyframe_dir,
        feature_dir=base.feature_dir,
        dimension=base.dimension,
        batch_size=base.batch_size,
        device=base.device,
        dtype=dtype,
        query_instruction=base.query_instruction,
        collection_name=base.collection_name,
    )


def _get_qwen_encoder() -> Qwen3VLEmbeddingEncoder:
    """Load Qwen exactly once, on the first retrieval request."""

    global _qwen_encoder
    if _qwen_encoder is not None:
        return _qwen_encoder
    with _qwen_encoder_lock:
        if _qwen_encoder is not None:
            return _qwen_encoder
        config = _qwen_runtime_config()
        print(
            "Loading Qwen3-VL embedding model "
            f"({config.model_id}) on {config.device}, dtype={config.dtype}."
        )
        _qwen_encoder = Qwen3VLEmbeddingEncoder(config)
        print("Qwen3-VL embedding model is ready.")
        return _qwen_encoder


class SearchRequest(BaseModel):
    visual_prompt: str
    prompt_variants: list[str] = Field(default_factory=list)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=10, ge=1)
    candidate_k: int = Field(default=50, ge=1)
    temporal_window: int = Field(default=0, ge=0)


def get_frame_index(payload: dict) -> int:
    frame_index = payload.get("frame_index")
    if isinstance(frame_index, int):
        return frame_index
    try:
        return int(os.path.splitext(str(payload.get("frame_id", "")))[0])
    except ValueError:
        return 0


def matches_scope(video_name: str, batch_ids: list[str], video_ids: list[str]) -> bool:
    video_name = str(video_name).strip().upper()
    batches = {item.strip().upper() for item in batch_ids if item.strip()}
    videos = {
        (f"V{int(item.strip()):03d}" if re.fullmatch(r"\d{1,3}", item.strip()) else item.strip().upper())
        for item in video_ids
        if item.strip()
    }
    if batches and video_name.split("_", 1)[0] not in batches:
        return False
    return not videos or any(
        video_name == video or video_name.endswith(f"_{video}") for video in videos
    )


def scoped_video_ids(batch_ids: list[str], video_ids: list[str]) -> list[str]:
    requested_batches = {item.strip().upper() for item in batch_ids if item.strip()}
    requested_videos = {
        (f"V{int(item.strip()):03d}" if re.fullmatch(r"\d{1,3}", item.strip()) else item.strip().upper())
        for item in video_ids
        if item.strip()
    }
    available = [
        entry.name.upper()
        for entry in os.scandir(KEYFRAME_DIR)
        if entry.is_dir() and re.fullmatch(r"L\d+_V\d+", entry.name.upper())
    ]
    return [
        video
        for video in available
        if (not requested_batches or video.split("_", 1)[0] in requested_batches)
        and (
            not requested_videos
            or video in requested_videos
            or any(video.endswith(f"_{suffix}") for suffix in requested_videos if "_" not in suffix)
        )
    ]


def temporal_deduplicate(points: list, *, top_k: int, temporal_window: int) -> list:
    """Keep the highest-ranked points separated by a frame-number window."""

    selected: list = []
    selected_frames: dict[str, list[int]] = defaultdict(list)
    for hit in points:
        payload = hit.payload or {}
        video_name = str(payload.get("video_id", "unknown"))
        frame_index = get_frame_index(payload)
        if any(
            abs(frame_index - selected_index) <= temporal_window
            for selected_index in selected_frames[video_name]
        ):
            continue
        selected.append(hit)
        selected_frames[video_name].append(frame_index)
        if len(selected) == top_k:
            break
    return selected


@app.post("/internal/search/visual")
async def search_visual(req: SearchRequest):
    try:
        prompts = list(
            dict.fromkeys(
                prompt.strip()
                for prompt in [req.visual_prompt, *req.prompt_variants]
                if prompt.strip()
            )
        )
        if not prompts:
            raise ValueError("visual_prompt must not be blank")

        config = _qwen_runtime_config()
        if not client.collection_exists(collection_name=config.collection_name):
            raise RuntimeError(
                f"Qwen collection {config.collection_name!r} does not exist. "
                "Run visual_pipeline.qwen3_embedding.database first."
            )

        vectors = _get_qwen_encoder().encode_texts(prompts)
        query_vector = vectors.mean(axis=0)
        norm = float(np.linalg.norm(query_vector))
        if norm <= 0 or not np.isfinite(norm):
            raise RuntimeError("Qwen returned an invalid query embedding")
        query_vector = (query_vector / norm).tolist()

        scope_requested = bool(req.batch_ids or req.video_ids)
        candidate_limit = min(max(req.top_k * 4, req.candidate_k), 10_000)
        exact_scope = scoped_video_ids(req.batch_ids, req.video_ids) if scope_requested else []
        if scope_requested and not exact_scope:
            candidate_limit = min(candidate_limit * 20, 10_000)
        scope_filter = (
            Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=exact_scope))])
            if exact_scope
            else None
        )
        searched = client.query_points(
            collection_name=config.collection_name,
            query=query_vector,
            limit=candidate_limit,
            query_filter=scope_filter,
        )
        candidates = searched.points if hasattr(searched, "points") else searched
        if scope_requested:
            candidates = [
                point
                for point in candidates
                if matches_scope(
                    (point.payload or {}).get("video_id", ""),
                    req.batch_ids,
                    req.video_ids,
                )
            ]
        points = temporal_deduplicate(
            candidates,
            top_k=req.top_k,
            temporal_window=req.temporal_window,
        )

        results = []
        for hit in points:
            payload = hit.payload or {}
            video_name = str(payload.get("video_id", "unknown"))
            frame_index = get_frame_index(payload)
            frame_name = str(payload.get("frame_id", "unknown"))
            frame_id = f"{video_name}_f{frame_index:04d}" if video_name != "unknown" else frame_name
            raw_score = float(hit.score)
            results.append(
                {
                    "frame_id": frame_id,
                    "score": raw_score,
                    "normalized_score": (raw_score + 1.0) / 2.0,
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "visual_model": "qwen3",
                }
            )
        return {"status": "success", "data": results}
    except Exception as exc:
        print(f"Error during Qwen visual search: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("Starting Qwen3-VL Visual API on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
