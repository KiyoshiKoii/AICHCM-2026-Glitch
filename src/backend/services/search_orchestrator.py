import asyncio
from typing import Any, Protocol

from backend.config import Settings
from backend.core.errors import UpstreamError
from backend.schemas.search import ParsedQuery, TextSearchResponse, SearchHit, SearchData
from backend.clients.visual_client import InternalPipelineClient, normalize_upstream_results
from backend.utils.rrf import reciprocal_rank_fusion


from backend.services.llm_reranker import GeminiReRanker
from backend.services.query_analyzer import enrich_explicit_interactions


def _filter_upstream_results(results: list[Any], batch_ids: list[str], video_ids: list[str]) -> list[Any]:
    """Enforce scope locally even if an upstream service ignores the filter."""
    if not batch_ids and not video_ids:
        return results
    batches = {item.upper() for item in batch_ids}
    videos = {item.upper() for item in video_ids}

    def matches(item: Any) -> bool:
        frame_id = getattr(item, "frame_id", "").upper()
        video_name = frame_id.rsplit("_F", 1)[0] if "_F" in frame_id else ""
        batch = video_name.split("_", 1)[0]
        if batches and batch not in batches:
            return False
        if videos and not any(
            video_name == video or video_name.endswith(f"_{video}")
            for video in videos
        ):
            return False
        return True

    return [item for item in results if matches(item)]


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
        self.reranker = GeminiReRanker(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_rerank_model,
        )

    async def search_text(
        self,
        query: str,
        top_k: int,
        *,
        use_rerank: bool = True,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        batch_ids: list[str] | None = None,
        video_ids: list[str] | None = None,
    ) -> TextSearchResponse:
        batch_ids = batch_ids or []
        video_ids = video_ids or []
        try:
            # Keep the public orchestration boundary defensive: parser
            # implementations can be swapped (Gemini/Ollama), and an older
            # worker or a provider response may still omit the explicit
            # subject-action-object plan.  Repair it here before constructing
            # the Dev2 payload so semantic retrieval never silently falls
            # back to broad keyword-only ranking for an explicit interaction.
            parsed = enrich_explicit_interactions(query, await self.parser.parse(query))
            visual_prompt = parsed.visual_prompt
            semantic_keywords = parsed.semantic_keywords + [query]
            object_queries = [item.model_dump(mode="json") for item in parsed.object_queries]
            spatial_queries = [item.model_dump(mode="json") for item in parsed.spatial_queries]
            interaction_queries = [item.model_dump(mode="json") for item in parsed.interaction_queries]
            ocr_queries = parsed.ocr_queries
            program_queries = parsed.program_queries
        except Exception:
            visual_prompt = query
            semantic_keywords = [query]
            object_queries = []
            spatial_queries = []
            interaction_queries = []
            ocr_queries = []
            program_queries = []

        try:
            dev1_task = self.dev1.search_text(
                {
                    "visual_prompt": visual_prompt,
                    "batch_ids": batch_ids,
                    "video_ids": video_ids,
                    "top_k": top_k * 2,
                }
            )
            dev2_task = self.dev2.search_text(
                {
                    "keywords": semantic_keywords,
                    "object_queries": object_queries,
                    "spatial_queries": spatial_queries,
                    "interaction_queries": interaction_queries,
                    "ocr_queries": ocr_queries,
                    "program_queries": program_queries,
                    "batch_ids": batch_ids,
                    "video_ids": video_ids,
                    "top_k": top_k * 2,
                }
            )
            
            dev1_res, dev2_res = await asyncio.gather(dev1_task, dev2_task, return_exceptions=True)
            
            rankings = {}
            if not isinstance(dev1_res, Exception):
                rankings["dev1"] = _filter_upstream_results(
                    normalize_upstream_results(dev1_res, source="dev1"), batch_ids, video_ids
                )
            
            if not isinstance(dev2_res, Exception):
                rankings["dev2"] = _filter_upstream_results(
                    normalize_upstream_results(dev2_res, source="dev2"), batch_ids, video_ids
                )
                
            merged_hits = reciprocal_rank_fusion(
                rankings,
                limit=top_k,
                thumbnail_base_url=self.settings.thumbnail_base_url,
                source_weights={
                    "dev1": visual_weight,
                    "dev2": text_weight,
                },
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
        if use_rerank and self.reranker.client:
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
        for i in range(1, min(top_k + 1, 101)):
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

    async def search_temporal_events(
        self,
        query: str,
        *,
        batch_ids: list[str] | None = None,
        video_ids: list[str] | None = None,
        top_k_videos: int = 20,
    ) -> dict[str, Any]:
        """Run the same-video ordered event mode through Dev2."""

        try:
            payload = await self.dev2.search_temporal_events(
                {
                    "query": query,
                    "batch_ids": batch_ids or [],
                    "video_ids": video_ids or [],
                    "top_k_videos": top_k_videos,
                }
            )
        except Exception as exc:
            raise UpstreamError(f"Failed to fetch temporal events from Dev2: {exc}") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("Dev2 temporal response must be an object")
        data = payload.get("data")
        if isinstance(data, dict):
            from backend.utils.thumbnail import build_thumbnail_url

            selected_video = data.get("selected_video") or {}
            for event in data.get("events", []):
                if not isinstance(event, dict) or not event.get("frame_id"):
                    continue
                event.setdefault("video_name", selected_video.get("video_id"))
                event.setdefault("frame_index", event.get("native_frame_idx"))
                event["thumbnail_url"] = build_thumbnail_url(
                    str(event["frame_id"]), self.settings.thumbnail_base_url
                )
            for candidate in data.get("candidates", []):
                if not isinstance(candidate, dict) or not candidate.get("frame_id"):
                    continue
                candidate.setdefault("video_name", candidate.get("video_id"))
                candidate.setdefault("frame_index", candidate.get("native_frame_idx"))
                candidate["thumbnail_url"] = build_thumbnail_url(
                    str(candidate["frame_id"]), self.settings.thumbnail_base_url
                )
        return payload
