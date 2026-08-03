import asyncio
from typing import Any, Protocol

from backend.config import Settings
from backend.core.errors import UpstreamError
from backend.schemas.search import ParsedQuery, TextSearchResponse, SearchHit, SearchData
from backend.clients.visual_client import InternalPipelineClient, normalize_upstream_results
from backend.utils.rrf import reciprocal_rank_fusion


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

    async def search_text(self, query: str, top_k: int) -> TextSearchResponse:
        # Fallback to raw query if parser is not available or LLM is offline
        try:
            parsed = await self.parser.parse(query)
            visual_prompt = parsed.visual_prompt
        except Exception:
            visual_prompt = query

        try:
            dev1_response = await self.dev1.search_text({"visual_prompt": visual_prompt, "top_k": top_k})
            dev1_results = normalize_upstream_results(dev1_response, source="dev1")
        except Exception as e:
            raise UpstreamError(f"Failed to fetch from visual pipeline: {e}")

        from backend.utils.thumbnail import build_thumbnail_url
        from backend.utils.keyframe_mapper import get_true_frame_idx
        results = []
        for r in dev1_results:
            video_name = r.metadata.get("video_name", "unknown")
            frame_index = r.metadata.get("frame_index", 0)
            # Create a standard frame_id format: L21_V022_f087
            formatted_frame_id = f"{video_name}_f{frame_index:04d}" if video_name != "unknown" else r.frame_id
            
            true_frame_idx = get_true_frame_idx(video_name, frame_index)
            if true_frame_idx is None:
                true_frame_idx = frame_index
            
            results.append(
                SearchHit(
                    frame_id=formatted_frame_id,
                    video_name=video_name,
                    frame_index=true_frame_idx,
                    score=r.score if r.score is not None else 0.0,
                    thumbnail_url=build_thumbnail_url(formatted_frame_id, self.settings.thumbnail_base_url),
                    metadata=r.metadata
                )
            )
            
        return TextSearchResponse(
            status="success",
            message="Retrieved successfully from Visual Pipeline",
            data=SearchData(
                total_results=len(results),
                results=results,
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
