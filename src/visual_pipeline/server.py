import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

from config import CLIP_MODEL_ID, QDRANT_DB_PATH, COLLECTION_NAME

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
    top_k: int = 10

@app.post("/internal/search/visual")
async def search_visual(req: SearchRequest):
    try:
        # 1. Encode câu visual_prompt bằng CLIP thành vector text
        inputs = processor(text=[req.visual_prompt], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # get_text_features() symmetric với get_image_features() trong extractor.py
            text_features = model.get_text_features(**inputs)

        # Một số phiên bản transformers trả về BaseModelOutputWithPooling
        # thay vì tensor trực tiếp — cần extract đúng trường
        if hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        elif hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        # L2 normalize — mirror y hệt bước cuối của extractor.py
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = [float(x) for x in text_features.cpu().numpy()[0]]

        # 2. Truy vấn Qdrant DB
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=req.top_k
        )

        points = search_results.points if hasattr(search_results, 'points') else search_results

        # 3. Định dạng kết quả và chuẩn hóa điểm
        results = []
        for hit in points:
            # Chuẩn hóa Cosine Score từ [-1, 1] sang [0, 1]
            normalized_score = (hit.score + 1) / 2
            
            payload = hit.payload
            video_name = payload.get("video_id", "unknown")
            frame_name = payload.get("frame_id", "unknown")
            
            # Trích xuất frame index (ví dụ: '0001.jpg' -> 1)
            frame_index = 0
            try:
                frame_index = int(os.path.splitext(frame_name)[0])
            except ValueError:
                pass
                
            results.append({
                "frame_id": frame_name,
                "score": float(normalized_score),
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
