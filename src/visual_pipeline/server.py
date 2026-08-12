import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise RuntimeError(
        "Qwen visual embedding cần sentence-transformers bản mới, "
        "transformers>=4.57.0 và qwen-vl-utils>=0.0.14."
    ) from error

try:
    from config import (
        COLLECTION_NAME,
        INDEX_METADATA_PATH,
        KEYFRAME_DIR,
        QDRANT_DB_PATH,
        QUERY_INSTRUCTION,
        VECTOR_SIZE,
        VISUAL_ARTIFACT_ID,
        VISUAL_ATTN_IMPLEMENTATION,
        VISUAL_DTYPE,
        VISUAL_MODEL_ID,
        VISUAL_MODEL_REVISION,
    )
except ImportError:
    from .config import (
        COLLECTION_NAME,
        INDEX_METADATA_PATH,
        KEYFRAME_DIR,
        QDRANT_DB_PATH,
        QUERY_INSTRUCTION,
        VECTOR_SIZE,
        VISUAL_ARTIFACT_ID,
        VISUAL_ATTN_IMPLEMENTATION,
        VISUAL_DTYPE,
        VISUAL_MODEL_ID,
        VISUAL_MODEL_REVISION,
    )

# Fix encoding issue for Vietnamese characters in Windows Terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(title="CV Internal API", description="API for Text-to-Video Search")


def _expected_metadata() -> dict[str, object]:
    return {
        "model_id": VISUAL_MODEL_ID,
        "model_revision": VISUAL_MODEL_REVISION,
        "artifact_id": VISUAL_ARTIFACT_ID,
        "vector_size": VECTOR_SIZE,
        "query_instruction": QUERY_INSTRUCTION,
    }


def _resolve_torch_dtype(device: str) -> torch.dtype:
    """Resolve the same precision policy used by extractor.py."""
    if VISUAL_DTYPE == "auto":
        if device == "cuda":
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if supports_bf16 else torch.float16
        return torch.float32

    requested = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[VISUAL_DTYPE]
    if device == "cpu" and requested != torch.float32:
        raise RuntimeError(
            "VISUAL_DTYPE phải là float32 khi chạy CPU; "
            "hãy dùng GPU CUDA cho bfloat16/float16."
        )
    if requested == torch.bfloat16 and device == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supports_bf16:
            raise RuntimeError("GPU hiện tại không hỗ trợ bfloat16; hãy dùng VISUAL_DTYPE=float16.")
    return requested


def _resolved_model_revision(embedder: SentenceTransformer) -> str:
    """Return the resolved HF commit when it is exposed by Sentence Transformers."""
    try:
        first_module = embedder._first_module()
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        return getattr(config, "_commit_hash", None) or VISUAL_MODEL_REVISION
    except (AttributeError, IndexError, TypeError):
        return VISUAL_MODEL_REVISION


def _read_index_metadata() -> dict[str, object]:
    try:
        with open(INDEX_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
            value = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Không đọc được metadata index Qwen. Hãy chạy database.py hoàn tất trước khi chạy server."
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("Metadata index Qwen không hợp lệ.")
    return value


def _collection_vector_size(collection_info) -> int | None:
    vectors_config = collection_info.config.params.vectors
    size = getattr(vectors_config, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors_config, dict):
        # Pipeline này dùng unnamed dense vector. Nhánh này chỉ giúp báo lỗi rõ
        # hơn nếu Qdrant/client trả config dưới dạng mapping.
        default_vector = vectors_config.get("")
        return getattr(default_vector, "size", None)
    return None


def _validate_qwen_index(qdrant_client: QdrantClient) -> dict[str, object]:
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise RuntimeError(
            f"Không tìm thấy Qwen collection '{COLLECTION_NAME}'. Hãy chạy database.py trước."
        )

    metadata = _read_index_metadata()
    if metadata.get("status") != "complete":
        raise RuntimeError("Qwen index chưa hoàn tất build; server không thể query collection này.")
    if metadata.get("collection_name") != COLLECTION_NAME:
        raise RuntimeError("Metadata index không thuộc collection Qwen hiện tại.")

    mismatched_keys = [
        key
        for key, expected_value in _expected_metadata().items()
        if metadata.get(key) != expected_value
    ]
    if mismatched_keys:
        raise RuntimeError(
            "Metadata index không khớp config hiện tại ở: " + ", ".join(mismatched_keys)
        )

    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    actual_size = _collection_vector_size(collection_info)
    if actual_size != VECTOR_SIZE:
        raise RuntimeError(
            f"Collection Qdrant có dimension {actual_size}, cần {VECTOR_SIZE}. "
            "Hãy rebuild bằng database.py."
        )
    return metadata


def _encode_text_queries(prompts: list[str]) -> np.ndarray:
    """Encode text queries in the same Qwen vector space as extractor.py images."""
    with torch.inference_mode():
        embeddings = model.encode(
            prompts,
            batch_size=len(prompts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            prompt=QUERY_INSTRUCTION,
            truncate_dim=VECTOR_SIZE,
        )

    text_features = np.asarray(embeddings, dtype=np.float32)
    if text_features.ndim == 1:
        text_features = text_features.reshape(1, -1)
    expected_shape = (len(prompts), VECTOR_SIZE)
    if tuple(text_features.shape) != expected_shape:
        raise RuntimeError(
            f"Qwen trả text vector shape {tuple(text_features.shape)}, cần {expected_shape}. "
            "Kiểm tra sentence-transformers có hỗ trợ truncate_dim/MRL hay không."
        )
    if not np.isfinite(text_features).all():
        raise RuntimeError("Qwen trả text vector chứa NaN hoặc Inf.")

    norms = np.linalg.norm(text_features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Qwen trả text vector có norm bằng 0.")
    return text_features / norms


print(f"Đang kết nối Qdrant DB tại: {QDRANT_DB_PATH}")
client = QdrantClient(path=QDRANT_DB_PATH)
index_metadata = _validate_qwen_index(client)
print("Qdrant Client đã sẵn sàng.")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = _resolve_torch_dtype(device)
model_kwargs: dict[str, object] = {"torch_dtype": MODEL_DTYPE}
if VISUAL_ATTN_IMPLEMENTATION:
    model_kwargs["attn_implementation"] = VISUAL_ATTN_IMPLEMENTATION

print(
    f"Đang khởi tạo Qwen visual embedding ({VISUAL_MODEL_ID}, "
    f"revision={VISUAL_MODEL_REVISION}) trên {device.upper()} "
    f"với {str(MODEL_DTYPE).replace('torch.', '')}..."
)
model = SentenceTransformer(
    VISUAL_MODEL_ID,
    revision=VISUAL_MODEL_REVISION,
    device=device,
    model_kwargs=model_kwargs,
)
model.eval()

indexed_revision = index_metadata.get("resolved_model_revision")
loaded_revision = _resolved_model_revision(model)
if indexed_revision and indexed_revision != loaded_revision:
    raise RuntimeError(
        "Checkpoint Qwen của server không khớp checkpoint lúc extraction/index "
        f"({loaded_revision} != {indexed_revision}). Hãy pin VISUAL_MODEL_REVISION hoặc re-index."
    )
print("Qwen visual embedding đã sẵn sàng.")


class SearchRequest(BaseModel):
    visual_prompt: str
    prompt_variants: list[str] = Field(default_factory=list)
    batch_ids: list[str] = Field(default_factory=list, max_length=10)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=10, ge=1)
    candidate_k: int = Field(default=50, ge=1)
    # Keyframes are already sampled sparsely.  Comparing their ordinal file
    # numbers with a +/-5 window removes distinct moments, especially when the
    # user scopes search to one video.  Zero still removes exact duplicate
    # points without suppressing neighbouring keyframes.
    temporal_window: int = Field(default=0, ge=0)


def get_frame_index(payload: dict) -> int:
    """Lấy frame index từ payload, fallback sang tên file."""
    frame_index = payload.get("frame_index")
    if isinstance(frame_index, int):
        return frame_index

    frame_name = payload.get("frame_id", "unknown")
    try:
        return int(os.path.splitext(frame_name)[0])
    except (TypeError, ValueError):
        return 0


def matches_scope(video_name: str, batch_ids: list[str], video_ids: list[str]) -> bool:
    """Apply the UI's batch/video scope to a Qdrant payload."""
    video_name = str(video_name).strip().upper()
    batches = {item.strip().upper() for item in batch_ids if item.strip()}
    videos = {
        (f"V{int(item.strip()):03d}" if re.fullmatch(r"\d{1,3}", item.strip()) else item.strip().upper())
        for item in video_ids
        if item.strip()
    }
    batch = video_name.split("_", 1)[0]
    if batches and batch not in batches:
        return False
    if videos and not any(video_name == video or video_name.endswith(f"_{video}") for video in videos):
        return False
    return True


def scoped_video_ids(batch_ids: list[str], video_ids: list[str]) -> list[str]:
    """Resolve the UI scope to exact Qdrant ``video_id`` values."""
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


def temporal_deduplicate(points, top_k: int, temporal_window: int):
    """
    Greedy temporal NMS trên danh sách đã sắp theo score giảm dần.

    Với mỗi video, chỉ giữ hit tốt nhất trong cửa sổ ±temporal_window.
    """
    selected = []
    selected_frames = defaultdict(list)

    for hit in points:
        payload = hit.payload
        video_name = payload.get("video_id", "unknown")
        frame_index = get_frame_index(payload)

        is_near_selected_frame = any(
            abs(frame_index - selected_index) <= temporal_window
            for selected_index in selected_frames[video_name]
        )
        if is_near_selected_frame:
            continue

        selected.append(hit)
        selected_frames[video_name].append(frame_index)
        if len(selected) == top_k:
            break

    return selected


@app.post("/internal/search/visual")
async def search_visual(req: SearchRequest):
    try:
        # 1. Encode prompt gốc và các biến thể tùy chọn trong cùng một batch.
        # Loại prompt rỗng/trùng để không query Qdrant dư thừa.
        prompts = list(dict.fromkeys(
            prompt.strip()
            for prompt in [req.visual_prompt, *req.prompt_variants]
            if prompt.strip()
        ))
        if not prompts:
            raise ValueError("visual_prompt không được để trống")

        # Qwen thay CLIP ở đúng bước embedding; normalize từng prompt, average
        # prompt variants rồi normalize lại vẫn giữ nguyên logic cũ.
        text_features = _encode_text_queries(prompts)
        ensemble_features = text_features.mean(axis=0, keepdims=True)
        ensemble_norm = np.linalg.norm(ensemble_features, axis=1, keepdims=True)
        if np.any(ensemble_norm == 0):
            raise RuntimeError("Ensemble text embedding có norm bằng 0.")
        ensemble_features = ensemble_features / ensemble_norm
        query_vector = ensemble_features[0].astype(np.float32).tolist()

        # 2. Lấy ít nhất 50 candidate rồi loại frame gần nhau trong cùng video.
        scope_requested = bool(req.batch_ids or req.video_ids)
        # Temporal NMS (and any legacy duplicate points) may discard a large
        # share of the nearest neighbours.  Pull a deeper candidate pool first
        # so the requested top_k can still be filled after deduplication.
        candidate_limit = min(max(req.top_k * 4, req.candidate_k), 10000)
        exact_scope = scoped_video_ids(req.batch_ids, req.video_ids) if scope_requested else []
        if scope_requested and not exact_scope:
            # Fall back to broad over-fetch only when the requested scope cannot
            # be resolved to exact video_id payloads.
            candidate_limit = min(candidate_limit * 20, 10000)
        scope_filter = (
            Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=exact_scope))])
            if exact_scope
            else None
        )
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=candidate_limit,
            query_filter=scope_filter,
        )
        candidate_points = (
            search_results.points
            if hasattr(search_results, "points")
            else search_results
        )
        if scope_requested:
            candidate_points = [
                point
                for point in candidate_points
                if matches_scope(
                    (point.payload or {}).get("video_id", ""),
                    req.batch_ids,
                    req.video_ids,
                )
            ]
        points = temporal_deduplicate(
            candidate_points,
            top_k=req.top_k,
            temporal_window=req.temporal_window,
        )

        # 3. Định dạng kết quả.
        # `score` là cosine similarity gốc mà Qdrant dùng để xếp hạng.
        # `normalized_score` chỉ là phiên bản đổi thang sang [0, 1] cho UI,
        # không phải xác suất hay confidence.
        results = []
        for hit in points:
            raw_score = float(hit.score)
            normalized_score = (raw_score + 1.0) / 2.0

            payload = hit.payload
            video_name = payload.get("video_id", "unknown")
            frame_name = payload.get("frame_id", "unknown")

            frame_index = get_frame_index(payload)

            formatted_frame_id = f"{video_name}_f{frame_index:04d}" if video_name != "unknown" else frame_name

            results.append({
                "frame_id": formatted_frame_id,
                "score": float(raw_score),
                "normalized_score": float(normalized_score),
                "video_name": video_name,
                "frame_index": frame_index,
            })

        return {
            "status": "success",
            "data": results,
        }
    except Exception as error:
        print(f"Error during search: {error}")
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    print("Đang khởi động Server API Nội bộ trên port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
