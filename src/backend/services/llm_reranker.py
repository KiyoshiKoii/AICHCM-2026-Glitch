import asyncio
import os
import json
import logging
from typing import List
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
    """List-wise Gemini reranker with a 50-image attention budget per call."""

    FIRST_PASS_BATCH_SIZE = 50
    FINALISTS_PER_BATCH = 25

    def __init__(self, api_key: str | None, model_name: str):
        self.api_key = api_key
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

        if len(hits) <= self.FIRST_PASS_BATCH_SIZE:
            return await self._rerank_once(query, hits)

        # Gemini scores from separate calls are not calibrated against each
        # other. Re-rank each 50-image group first, then compare the strongest
        # candidates from every group together in one final call.
        batches = [
            hits[offset : offset + self.FIRST_PASS_BATCH_SIZE]
            for offset in range(0, len(hits), self.FIRST_PASS_BATCH_SIZE)
        ]
        first_pass_batches = await asyncio.gather(
            *(self._rerank_once(query, batch) for batch in batches)
        )
        finalists = [
            hit
            for batch in first_pass_batches
            for hit in batch[: self.FINALISTS_PER_BATCH]
        ]

        # The public text-search endpoint returns at most 100 hits, so this
        # final comparison receives at most 50 images (two batches x 25).
        # Keep this safeguard if the reranker is reused by another caller.
        if len(finalists) > self.FIRST_PASS_BATCH_SIZE:
            logger.warning(
                "[GeminiReRanker] %s finalists exceed the final 50-image budget; "
                "only the first %s will receive cross-batch reranking.",
                len(finalists),
                self.FIRST_PASS_BATCH_SIZE,
            )
            finalists = finalists[: self.FIRST_PASS_BATCH_SIZE]

        final_reranked = await self._rerank_once(query, finalists)
        finalist_ids = {hit.frame_id for hit in finalists}

        # Scores from independent first-pass batches cannot safely order the
        # non-finalists globally. Preserve their original RRF order after the
        # consistently reranked final candidates.
        remaining_hits = [hit for hit in hits if hit.frame_id not in finalist_ids]
        return final_reranked + remaining_hits

    async def _rerank_once(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """Send one list-wise image batch to Gemini (maximum 50 images)."""
        # Prepare images
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        contents = []
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

        contents.insert(
            0,
            f"Here are {len(valid_hits)} images retrieved for the query: '{query}'. "
            "Please act as an expert judge. Evaluate how well each image matches the query. "
            "Return the results as a JSON object mapping each frame_id to a relevance score (0.0 to 1.0).",
        )

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
