from backend.core.errors import VQAUnavailableError
from backend.schemas.vqa import VQACandidate, VQAData, VQARequest, VQAResponse
from backend.services.vqa_answerer import VQAAnswerer


class VQAService:
    """Compose existing retrieval/reranking with an image-only VLM answerer."""

    def __init__(self, *, search_service, answerer: VQAAnswerer) -> None:
        self.search_service = search_service
        self.answerer = answerer

    async def answer(self, request: VQARequest) -> VQAResponse:
        if not self.answerer.available:
            raise VQAUnavailableError("Gemini VQA is unavailable. Configure GEMINI_API_KEY first.")

        search_response = await self.search_service.search_text(
            request.query,
            request.retrieval_top_k,
            use_rerank=request.use_rerank,
        )
        rrf_hits = search_response.data.results
        reranked_hits = (
            search_response.data.llm_reranked_results
            if request.use_rerank
            else None
        )
        hits = (reranked_hits or rrf_hits)[: request.answer_top_k]

        answers = await self.answerer.answer_batch(request.question, hits)
        candidates: list[VQACandidate] = []
        for hit in hits:
            answer = answers.get(hit.frame_id)
            if answer is None:
                continue
            candidates.append(VQACandidate(
                **hit.model_dump(),
                answer=answer.answer,
                confidence=answer.confidence,
            ))

        return VQAResponse(
            data=VQAData(
                total_candidates=len(candidates),
                candidates=candidates,
                results=rrf_hits,
                llm_reranked_results=reranked_hits,
                use_rerank=request.use_rerank,
            )
        )
