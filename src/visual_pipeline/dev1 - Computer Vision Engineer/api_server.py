import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# Fix encoding issue for Vietnamese characters in Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="CV Internal API", description="API for Text-to-Video Search")

print("Đang khởi tạo CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model CLIP đã sẵn sàng trên {device.upper()}.")

print("Đang kết nối Qdrant DB...")
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "local_qdrant_db")
client = QdrantClient(path=db_path)
collection_name = "kis_images"
print("Qdrant Client đã sẵn sàng.")

class SearchRequest(BaseModel):
    visual_prompt: str
    top_k: int = 10

@app.post("/internal/search/visual")
async def search_visual(req: SearchRequest):
    try:
        # 1. Encode câu visual_prompt bằng CLIP thành vector
        inputs = processor(text=[req.visual_prompt], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            
            # QUAN TRỌNG: Phải mirror đúng pipeline của embed_folder.py
            # embed_folder.py lưu: L2_norm(vision_pooler_output) -- KHÔNG qua visual_projection
            # vì vision_pooler_output là 512-dim, visual_projection.in_features=768 => dim check thất bại
            # => text phải dùng: L2_norm(text_pooler_output) -- KHÔNG qua text_projection
            if isinstance(outputs, torch.Tensor):
                # Transformers cũ: trả thẳng tensor đã projected
                # Trong trường hợp này image vectors cũng sẽ đã projected, nên dùng thẳng
                text_features = outputs
            elif hasattr(outputs, "pooler_output"):
                # Transformers v5: trả BaseModelOutputWithPooling, lấy pooler_output thô
                text_features = outputs.pooler_output
            else:
                text_features = outputs[1]
                
        # L2 normalize — mirror y chang bước cuoi cua embed_folder.py
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()
        query_vector = [float(x) for x in query_vector]

        # 2. Truy vấn Qdrant DB
        search_results = client.query_points(
            collection_name=collection_name,
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

        return results
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Đang khởi động Server API Nội bộ trên port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
