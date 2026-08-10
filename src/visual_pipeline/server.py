import os
import re
import sys
from collections import defaultdict

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

try:
    from config import CLIP_MODEL_ID, QDRANT_DB_PATH, COLLECTION_NAME, KEYFRAME_DIR
except ImportError:
    from .config import CLIP_MODEL_ID, QDRANT_DB_PATH, COLLECTION_NAME, KEYFRAME_DIR

# Fix encoding issue for Vietnamese characters in Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="CV Internal API", description="API for Text-to-Video Search")

print(f"Đang khởi tạo CLIP Model ({CLIP_MODEL_ID})...")
model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model CLIP đã sẵn sàng trên {device.upper()}.")

print(f"Đang kết nối Qdrant DB tại: {QDRANT_DB_PATH}")
client = QdrantClient(path=QDRANT_DB_PATH)
print("Qdrant Client đã sẵn sàng.")

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

        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # get_text_features() symmetric với get_image_features() trong extractor.py
            text_features = model.get_text_features(**inputs)

        # Một số phiên bản transformers trả về BaseModelOutputWithPooling
        # thay vì tensor trực tiếp — cần extract đúng trường
        if hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        elif hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        # Normalize từng embedding, lấy trung bình, rồi normalize ensemble.
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        ensemble_features = text_features.mean(dim=0, keepdim=True)
        ensemble_features = ensemble_features / ensemble_features.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )
        query_vector = [
            float(x)
            for x in ensemble_features.cpu().numpy()[0]
        ]

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
                "frame_index": frame_index
            })

        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Đang khởi động Server API Nội bộ trên port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
