import asyncio
from typing import Any, Protocol

from backend.config import Settings
from backend.core.errors import UpstreamError
from backend.schemas.search import ParsedQuery, TextSearchResponse, SearchHit, SearchData
from backend.clients.visual_client import InternalPipelineClient, normalize_upstream_results
from backend.utils.rrf import reciprocal_rank_fusion


from backend.services.llm_reranker import GeminiReRanker


class QueryParser(Protocol):
    async def parse(self, query: str) -> ParsedQuery: ...


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        parser: QueryParser,
        dev1: InternalPipelineClient,
        dev2: InternalPipelineClient,
    ) -> None:
        self.settings = settings
        self.parser = parser
        self.dev1 = dev1
        self.dev2 = dev2
        self.reranker = GeminiReRanker(api_key=settings.gemini_api_key)

    async def search_text(self, query: str, top_k: int) -> TextSearchResponse:
        try:
            parsed = await self.parser.parse(query)
            visual_prompt = parsed.visual_prompt
            semantic_keywords = parsed.semantic_keywords + [query]
        except Exception:
            visual_prompt = query
            semantic_keywords = [query]

        try:
            dev1_task = self.dev1.search_text({"visual_prompt": visual_prompt, "top_k": top_k * 2})
            dev2_task = self.dev2.search_text({"keywords": semantic_keywords, "top_k": top_k * 2})
            
            dev1_res, dev2_res = await asyncio.gather(dev1_task, dev2_task, return_exceptions=True)
            
            rankings = {}
            if not isinstance(dev1_res, Exception):
                rankings["dev1"] = normalize_upstream_results(dev1_res, source="dev1")
            
            if not isinstance(dev2_res, Exception):
                rankings["dev2"] = normalize_upstream_results(dev2_res, source="dev2")
                
            merged_hits = reciprocal_rank_fusion(
                rankings,
                limit=top_k,
                thumbnail_base_url=self.settings.thumbnail_base_url,
            )
        except Exception as e:
            raise UpstreamError(f"Failed to fetch from upstream pipelines: {e}")

        from backend.utils.keyframe_mapper import get_true_frame_idx
        
        for hit in merged_hits:
            if "_f" in hit.frame_id:
                video_name, frame_part = hit.frame_id.rsplit("_f", 1)
                try:
                    frame_index = int(frame_part)
                except ValueError:
                    frame_index = 0
            else:
                video_name = "unknown"
                frame_index = 0
                
            true_frame_idx = get_true_frame_idx(video_name, frame_index)
            
            hit.video_name = video_name
            hit.frame_index = true_frame_idx if true_frame_idx is not None else frame_index
            
        # Execute LLM Reranking on Top 100
        llm_reranked_results = None
        if self.reranker.client:
            top_100 = merged_hits[:100]
            reranked_top_100 = await self.reranker.rerank(query, top_100)
            
            # Combine the newly reranked top 100 with the rest (if any)
            llm_reranked_results = reranked_top_100 + merged_hits[100:]
            
        return TextSearchResponse(
            status="success",
            message="Retrieved successfully from Visual & Semantic Pipelines",
            data=SearchData(
                total_results=len(merged_hits),
                results=merged_hits,
                llm_reranked_results=llm_reranked_results,
            )
        )

    async def search_image(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        top_k: int,
    ) -> TextSearchResponse:
        results = []
        from backend.utils.thumbnail import build_thumbnail_url
        for i in range(1, min(top_k + 1, 51)):
            frame_id = f"L21_V001_f{i:04d}"
            results.append(
                SearchHit(
                    frame_id=frame_id,
                    score=0.99 - (i * 0.01),
                    thumbnail_url=build_thumbnail_url(frame_id, self.settings.thumbnail_base_url),
                    metadata={"timestamp": f"00:00:{i:02d}"}
                )
            )

        return TextSearchResponse(
            status="success",
            message="Image retrieved successfully (MOCK)",
            data=SearchData(
                total_results=len(results),
                results=results,
            )
        )
