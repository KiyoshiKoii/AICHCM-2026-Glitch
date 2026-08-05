import os
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from backend.schemas.search import SearchHit

logger = logging.getLogger("uvicorn.error")


class FrameScore(BaseModel):
    frame_id: str
    score: float = Field(ge=0.0, le=1.0)


class RerankResponse(BaseModel):
    scores: list[FrameScore]


class GeminiReRanker:
    def __init__(self, api_key: str | None, model_name: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key
        # Use gemini-3.1-flash-lite which has generous free tier in 2026
        self.model_name = model_name
        if api_key and genai:
            self.client = genai.Client(api_key=api_key)
            logger.info("[GeminiReRanker] Initialized successfully with API Key")
        else:
            self.client = None
            logger.warning("[GeminiReRanker] Not initialized. Missing API Key or google-genai library.")

    async def rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        if not self.client or not hits:
            return hits

        # Prepare images
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        contents = [
            f"Here are {len(hits)} images retrieved for the query: '{query}'. "
            "Please act as an expert judge. Evaluate how well each image matches the query. "
            "Return the results as a JSON object mapping each frame_id to a relevance score (0.0 to 1.0)."
        ]
        
        valid_hits = []
        for hit in hits:
            # Reconstruct the physical path from frame_id
            frame_id = hit.frame_id
            if "_f" not in frame_id:
                continue
                
            video_name, frame_part = frame_id.rsplit("_f", 1)
            try:
                frame_int = int(frame_part)
            except ValueError:
                continue
                
            img_path = os.path.join(base_dir, "data", "keyframes", video_name, f"{frame_int:03d}.jpg")
            
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    
                    contents.append(f"Image for frame_id: {hit.frame_id}")
                    contents.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                    valid_hits.append(hit)
                except Exception as e:
                    logger.error(f"[GeminiReRanker] Error reading image {img_path}: {e}")
            else:
                logger.warning(f"[GeminiReRanker] Image not found on disk: {img_path}")
                
        if not valid_hits:
            logger.warning(f"[GeminiReRanker] No valid images found on disk out of {len(hits)} hits.")
            return hits

        try:
            # Use structured outputs with async client
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RerankResponse,
                    temperature=0.1,
                )
            )
            
            result_json = response.text
            data = json.loads(result_json)
            
            # Map new scores
            score_map = {item["frame_id"]: item["score"] for item in data.get("scores", [])}
            
            reranked_hits = []
            for hit in hits:
                new_hit = hit.model_copy()
                if new_hit.frame_id in score_map:
                    new_hit.score = score_map[new_hit.frame_id]
                reranked_hits.append(new_hit)
                
            # Sort descending by new score
            reranked_hits.sort(key=lambda x: x.score, reverse=True)
            logger.info(f"[GeminiReRanker] Đã chấm điểm và xếp hạng lại thành công {len(reranked_hits)} hình ảnh!")
            return reranked_hits
            
        except Exception as e:
            logger.error(f"[GeminiReRanker] Error calling Gemini API: {e}")
            return hits
