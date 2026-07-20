import asyncio
from typing import Any, Protocol

from backend.core.config import Settings
from backend.core.errors import UpstreamError
from backend.models.schemas import ParsedQuery, TextSearchResponse
from backend.services.pipeline_clients import InternalPipelineClient, normalize_upstream_results
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
        parsed = await self.parser.parse(query)
        dev1_payload = {
            "query": parsed.visual_prompt,
            "top_k": self.settings.upstream_top_k,
        }
        dev2_payload = {
            "query": " ".join(parsed.semantic_keywords),
            "keywords": parsed.semantic_keywords,
            "top_k": self.settings.upstream_top_k,
        }

        responses = await asyncio.gather(
            self.dev1.search_text(dev1_payload),
            self.dev2.search_text(dev2_payload),
            return_exceptions=True,
        )

        rankings: dict[str, list[Any]] = {}
        warnings: list[str] = []
        for source, response in zip(("dev1", "dev2"), responses, strict=True):
            if isinstance(response, BaseException):
                warnings.append(f"{source} unavailable: {response}")
                continue
            rankings[source] = normalize_upstream_results(response, source)

        if not rankings:
            raise UpstreamError("Both internal text-search APIs failed")
        if warnings and not self.settings.allow_partial_results:
            raise UpstreamError(
                "An internal text-search API failed and partial results are disabled"
            )

        results = reciprocal_rank_fusion(
            rankings,
            k=self.settings.rrf_k,
            limit=min(top_k, self.settings.output_top_k),
            thumbnail_base_url=self.settings.thumbnail_base_url,
        )
        return TextSearchResponse(
            query=query,
            parsed_query=parsed,
            count=len(results),
            results=results,
            warnings=warnings,
        )

    async def search_image(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> Any:
        return await self.dev1.search_image(
            filename=filename,
            content=content,
            content_type=content_type,
        )
