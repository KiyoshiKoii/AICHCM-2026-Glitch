import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol

from backend.config import Settings
from backend.core.errors import UpstreamError
from backend.schemas.search import (
    ParsedQuery,
    TextSearchResponse,
    SearchHit,
    SearchData,
    UpstreamResult,
)
from backend.clients.visual_client import InternalPipelineClient, normalize_upstream_results
from backend.utils.rrf import reciprocal_rank_fusion


from backend.services.llm_reranker import GeminiReRanker
from backend.services.camera_motion_verifier import CameraMotionVerifier
from backend.services.query_analyzer import enrich_explicit_interactions
from backend.services.temporal_video_fusion import (
    KISQuery,
    aggregate_kis_rankings,
    build_kis_queries,
    fuse_summary_and_kis,
)


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


_KEYFRAME_ID_PATTERN = re.compile(r"^(?P<video_id>L\d+_V\d+)_f(?P<ordinal>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class _ASRWindow:
    """One retrieved transcript passage, anchored to its first keyframe."""

    rank: int
    anchor_frame_id: str
    video_id: str
    start_ms: int
    end_ms: int
    score: float
    metadata: dict[str, Any]


def _asr_windows_from_payload(payload: Any) -> tuple[list[UpstreamResult], list[_ASRWindow]]:
    """Map ASR passages to first keyframes while retaining their time ranges."""

    raw_results = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        return [], []

    from backend.utils.keyframe_mapper import get_nearest_keyframe_position

    anchors: list[UpstreamResult] = []
    windows: list[_ASRWindow] = []
    for rank, raw in enumerate(raw_results, start=1):
        if not isinstance(raw, dict):
            continue
        video_id = str(raw.get("video_id", "")).strip().upper()
        transcript = str(raw.get("text", "")).strip()
        if not video_id or not transcript:
            continue
        try:
            start_ms = max(0, int(raw.get("start_ms", 0)))
            end_ms = max(start_ms, int(raw.get("end_ms", start_ms)))
            score = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            continue

        nearest = get_nearest_keyframe_position(video_id, start_ms)
        if nearest is None:
            continue
        ordinal, position = nearest
        frame_id = f"{video_id}_f{ordinal:04d}"
        metadata = {
            "asr_id": raw.get("asr_id"),
            "transcript": transcript,
            "timestamp_ms": start_ms,
            "seek_timestamp_ms": start_ms,
            "asr_start_ms": start_ms,
            "asr_end_ms": end_ms,
            "start_segment_index": raw.get("start_segment_index"),
            "end_segment_index": raw.get("end_segment_index"),
            "fps": position.get("fps"),
        }
        anchors.append(UpstreamResult(frame_id=frame_id, score=score, metadata=metadata))
        windows.append(
            _ASRWindow(
                rank=rank,
                anchor_frame_id=frame_id,
                video_id=video_id,
                start_ms=start_ms,
                end_ms=end_ms,
                score=score,
                metadata=metadata,
            )
        )
    return anchors, windows


def _apply_asr_window_evidence(
    hits: list[SearchHit],
    windows: list[_ASRWindow],
    *,
    asr_weight: float,
    rrf_k: int = 60,
    limit: int,
) -> list[SearchHit]:
    """Boost visual/caption frames whose timestamps fall inside an ASR passage.

    ASR is timestamped at passage level, while Qwen and caption search are
    keyframe-level.  Matching only equal frame IDs would discard valid
    evidence (for example, speech may start several seconds before the action
    becomes visible), so a matching transcript passage supports all retrieved
    keyframes within its interval.
    """

    if asr_weight <= 0 or not windows:
        return hits[:limit]

    from backend.utils.keyframe_mapper import get_keyframe_position

    windows_by_video: dict[str, list[_ASRWindow]] = {}
    for window in windows:
        windows_by_video.setdefault(window.video_id, []).append(window)

    for hit in hits:
        parsed = _KEYFRAME_ID_PATTERN.fullmatch(hit.frame_id)
        if parsed is None:
            continue
        video_id = parsed.group("video_id").upper()
        position = get_keyframe_position(video_id, int(parsed.group("ordinal")))
        if position is None:
            continue
        timestamp_ms = int(position["timestamp_ms"])
        matches = [
            window
            for window in windows_by_video.get(video_id, [])
            if window.start_ms <= timestamp_ms <= window.end_ms
        ]
        if not matches:
            continue

        # A passage can support more than one keyframe. Keep the strongest
        # passage and attach its transcript so the result remains inspectable.
        evidence = min(matches, key=lambda item: (item.rank, -item.score))
        metadata = dict(hit.metadata)
        source_ranks = dict(metadata.get("source_ranks", {}))
        source_scores = dict(metadata.get("source_scores", {}))
        source_ranks.setdefault("asr", evidence.rank)
        source_scores.setdefault("asr", evidence.score)
        metadata.update(evidence.metadata)
        metadata["source_ranks"] = source_ranks
        metadata["source_scores"] = source_scores
        metadata["asr_temporal_match"] = True
        hit.metadata = metadata

        # The anchor frame already received its ASR RRF contribution. Nearby
        # visual/caption frames need one equivalent contribution from the
        # same spoken passage.
        if hit.frame_id != evidence.anchor_frame_id:
            hit.score += asr_weight / (rrf_k + evidence.rank)

    return sorted(
        hits,
        key=lambda item: (-item.score, item.frame_id),
    )[:limit]


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
        self.camera_motion_verifier = CameraMotionVerifier()

    async def search_text(
        self,
        query: str,
        top_k: int,
        *,
        use_rerank: bool = True,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        asr_weight: float = 0.0,
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
            # The visual service is Qwen-only and intentionally receives the
            # complete raw user wording. Gemini parsing above remains for the
            # semantic/Elasticsearch route and optional Gemini reranking.
            tasks: dict[str, Any] = {
                "dev1": self.dev1.search_text(
                    {
                        "visual_prompt": query,
                        "batch_ids": batch_ids,
                        "video_ids": video_ids,
                        "top_k": top_k * 2,
                    }
                ),
                "dev2": self.dev2.search_text(
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
                ),
            }
            if asr_weight > 0:
                tasks["asr"] = self.dev2.search_asr(
                    {
                        "query": query,
                        "batch_ids": batch_ids,
                        "video_ids": video_ids,
                        "top_k": top_k * 2,
                    }
                )

            task_names = list(tasks)
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            responses = dict(zip(task_names, task_results, strict=True))

            rankings: dict[str, list[UpstreamResult]] = {}
            dev1_res = responses["dev1"]
            if not isinstance(dev1_res, Exception):
                rankings["dev1"] = _filter_upstream_results(
                    normalize_upstream_results(dev1_res, source="dev1"), batch_ids, video_ids
                )

            dev2_res = responses["dev2"]
            if not isinstance(dev2_res, Exception):
                rankings["dev2"] = _filter_upstream_results(
                    normalize_upstream_results(dev2_res, source="dev2"), batch_ids, video_ids
                )

            asr_windows: list[_ASRWindow] = []
            asr_res = responses.get("asr")
            if asr_weight > 0 and asr_res is not None and not isinstance(asr_res, Exception):
                asr_anchors, asr_windows = _asr_windows_from_payload(asr_res)
                if asr_anchors:
                    rankings["asr"] = _filter_upstream_results(asr_anchors, batch_ids, video_ids)

            candidate_limit = max(top_k * 4, top_k)
            merged_hits = reciprocal_rank_fusion(
                rankings,
                limit=candidate_limit,
                thumbnail_base_url=self.settings.thumbnail_base_url,
                source_weights={
                    "dev1": visual_weight,
                    "dev2": text_weight,
                    "asr": asr_weight,
                },
            )
            merged_hits = _apply_asr_window_evidence(
                merged_hits,
                asr_windows,
                asr_weight=asr_weight,
                limit=top_k,
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
            message=(
                "Retrieved successfully from visual, caption and ASR pipelines"
                if asr_weight > 0
                else "Retrieved successfully from Visual & Semantic Pipelines"
            ),
            data=SearchData(
                total_results=len(merged_hits),
                results=merged_hits,
                llm_reranked_results=llm_reranked_results,
            )
        )

    async def search_asr(
        self,
        query: str,
        top_k: int,
        *,
        batch_ids: list[str] | None = None,
        video_ids: list[str] | None = None,
    ) -> TextSearchResponse:
        """Find timestamped transcript passages and attach a playable keyframe."""

        try:
            payload = await self.dev2.search_asr(
                {
                    "query": query,
                    "batch_ids": batch_ids or [],
                    "video_ids": video_ids or [],
                    "top_k": top_k,
                }
            )
        except Exception as exc:
            raise UpstreamError(f"Failed to fetch ASR passages from Dev2: {exc}") from exc

        raw_results = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise UpstreamError("Dev2 ASR response must contain a data array")

        from backend.utils.keyframe_mapper import get_nearest_keyframe_position
        from backend.utils.thumbnail import build_thumbnail_url

        results: list[SearchHit] = []
        for index, raw in enumerate(raw_results):
            if not isinstance(raw, dict):
                raise UpstreamError(f"Dev2 ASR result at index {index} is not an object")
            video_id = str(raw.get("video_id", "")).strip().upper()
            transcript = str(raw.get("text", "")).strip()
            if not video_id or not transcript:
                raise UpstreamError(f"Dev2 ASR result at index {index} is missing video_id/text")
            try:
                start_ms = max(0, int(raw.get("start_ms", 0)))
                end_ms = max(start_ms, int(raw.get("end_ms", start_ms)))
                score = float(raw.get("score", 0.0))
            except (TypeError, ValueError) as exc:
                raise UpstreamError(f"Dev2 ASR result at index {index} has invalid numeric data") from exc

            nearest = get_nearest_keyframe_position(video_id, start_ms)
            if nearest:
                keyframe_n, position = nearest
                native_frame_idx = position["frame_index"]
                fps = position["fps"]
            else:
                keyframe_n, native_frame_idx, fps = 1, 0, None
            frame_id = f"{video_id}_f{keyframe_n:04d}"
            results.append(
                SearchHit(
                    frame_id=frame_id,
                    video_name=video_id,
                    frame_index=native_frame_idx,
                    score=score,
                    thumbnail_url=build_thumbnail_url(frame_id, self.settings.thumbnail_base_url),
                    metadata={
                        "asr_id": raw.get("asr_id"),
                        "transcript": transcript,
                        "timestamp_ms": start_ms,
                        "seek_timestamp_ms": start_ms,
                        "asr_start_ms": start_ms,
                        "asr_end_ms": end_ms,
                        "start_segment_index": raw.get("start_segment_index"),
                        "end_segment_index": raw.get("end_segment_index"),
                        "fps": fps,
                    },
                )
            )

        return TextSearchResponse(
            status="success",
            message="Retrieved timestamped ASR passages",
            data=SearchData(
                total_results=len(results),
                results=results,
                llm_reranked_results=None,
            ),
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
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        event_options: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Locate event evidence, using scoped KIS after a video is chosen.

        The selected-video phase is an inspection workflow: users need several
        strong visual/caption candidates for each event, not a premature claim
        of the exact transition frame.  Qwen/dense verification remains an
        optional later refinement inside the Semantic Pipeline.
        """
        selected_video_ids = [item.upper() for item in (video_ids or []) if item.strip()]
        if selected_video_ids:
            return await self._search_selected_video_events_with_kis(
                query,
                batch_ids=batch_ids or [],
                video_ids=selected_video_ids,
                text_weight=text_weight,
                visual_weight=visual_weight,
                event_options=event_options or [],
            )

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

    async def _search_selected_video_events_with_kis(
        self,
        query: str,
        *,
        batch_ids: list[str],
        video_ids: list[str],
        candidates_per_event: int = 10,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        event_options: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Return KIS candidates for every explicitly selected TRAKE event."""
        specs = build_kis_queries(
            query,
            None,
            event_options=event_options or [],
            default_text_weight=text_weight,
            default_visual_weight=visual_weight,
        )
        rankings = await asyncio.gather(
            *(
                self._search_event_with_standard_kis(
                    spec,
                    top_k=candidates_per_event,
                    batch_ids=batch_ids,
                    video_ids=video_ids,
                    use_rerank=spec.use_rerank,
                    text_weight=spec.text_weight,
                    visual_weight=spec.visual_weight,
                )
                for spec in specs
            )
        )
        rankings = await asyncio.gather(
            *(
                self._apply_camera_motion_verification(spec, hits)
                for spec, hits in zip(specs, rankings)
            )
        )
        rankings = self._apply_selected_event_order(specs, rankings)

        events: list[dict[str, Any]] = []
        for spec, hits in zip(specs, rankings):
            for rank, hit in enumerate(hits, start=1):
                metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
                events.append(
                    {
                        "event_id": spec.event_id,
                        "description": spec.description,
                        "rank": rank,
                        "frame_id": hit.frame_id,
                        "video_id": video_ids[0],
                        "native_frame_idx": (
                            hit.frame_index
                            if isinstance(hit.frame_index, int)
                            else self._kis_frame_index(hit.frame_id, metadata)
                        ),
                        "score": float(hit.score),
                        "confidence": float(hit.score),
                        "anchor_type": "kis_candidate",
                        "reason_vi": "KIS visual + caption candidate within selected video",
                        "matched_context_entities": [],
                        "thumbnail_url": hit.thumbnail_url,
                        "source_ranks": metadata.get("source_ranks", {}),
                        "caption": self._kis_caption(metadata),
                        "camera_motion": metadata.get("camera_motion"),
                        "event_order": metadata.get("event_order"),
                    }
                )

        return {
            "status": "success",
            "data": {
                "mode": "temporal_events_kis_selected_video",
                "query": query,
                "selected_video": {"video_id": video_ids[0]},
                "events": events,
                "kis": {
                    "queries": [
                        {
                            "event_id": spec.event_id,
                            "description": spec.description,
                            "visual_prompt": spec.visual_prompt,
                            "weights": {
                                "text": spec.text_weight,
                                "visual": spec.visual_weight,
                            },
                            "requires_after_previous": spec.requires_after_previous,
                            "verify_camera_motion": spec.verify_camera_motion,
                            "motion_weight": spec.motion_weight,
                        }
                        for spec in specs
                    ],
                    "candidates_per_event": candidates_per_event,
                    "weights": {
                        "text": text_weight,
                        "visual": visual_weight,
                    },
                },
            },
        }

    async def _search_event_with_standard_kis(
        self,
        spec: KISQuery,
        *,
        top_k: int,
        batch_ids: list[str],
        video_ids: list[str],
        use_rerank: bool = False,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
    ) -> list[SearchHit]:
        """Reuse the public KIS path, including Gemini interaction parsing."""
        response = await self.search_text(
            spec.visual_prompt,
            top_k,
            use_rerank=use_rerank,
            text_weight=text_weight,
            visual_weight=visual_weight,
            batch_ids=batch_ids,
            video_ids=video_ids,
        )
        # The regular KIS endpoint keeps the RRF baseline and Gemini order in
        # separate response fields for UI comparison.  TRAKE needs the latter
        # to influence the video score, but only when the user explicitly
        # enables it.
        return response.data.llm_reranked_results or response.data.results

    async def _apply_camera_motion_verification(
        self,
        spec: KISQuery,
        hits: list[SearchHit],
        *,
        max_videos: int = 10,
    ) -> list[SearchHit]:
        """Re-rank only a small KIS finalist pool for one configured event."""

        if not spec.verify_camera_motion or not hits:
            return hits
        if not self.camera_motion_verifier.has_explicit_camera_constraint(spec.description):
            return hits

        by_video: dict[str, list[SearchHit]] = {}
        for hit in hits:
            video_id = (hit.video_name or hit.frame_id.rsplit("_f", 1)[0]).upper()
            if video_id:
                by_video.setdefault(video_id, []).append(hit)
        finalists = sorted(
            by_video,
            key=lambda video_id: max(float(hit.score) for hit in by_video[video_id]),
            reverse=True,
        )[:max_videos]
        if not finalists:
            return hits

        async def verify(video_id: str):
            candidates = sorted(by_video[video_id], key=lambda hit: -float(hit.score))[:3]
            frame_indices = [
                hit.frame_index
                if isinstance(hit.frame_index, int)
                else self._kis_frame_index(hit.frame_id, hit.metadata)
                for hit in candidates
            ]
            result = await asyncio.to_thread(
                self.camera_motion_verifier.verify,
                video_id=video_id,
                description=spec.description,
                frame_indices=frame_indices,
            )
            return video_id, result

        verified = dict(await asyncio.gather(*(verify(video_id) for video_id in finalists)))
        max_score = max(float(hit.score) for hit in hits)
        motion_weight = min(1.0, max(0.0, spec.motion_weight))
        for video_id, video_hits in by_video.items():
            result = verified.get(video_id)
            if result is None or not bool(result.details.get("available")):
                continue
            for hit in video_hits:
                relative_kis = float(hit.score) / max_score if max_score > 0 else 0.0
                hit.score = max_score * (
                    (1.0 - motion_weight) * relative_kis
                    + motion_weight * result.score
                )
                hit.metadata["camera_motion"] = {
                    "score": round(result.score, 8),
                    "weight": motion_weight,
                    "reason": result.reason,
                    **result.details,
                }
        return sorted(hits, key=lambda hit: -float(hit.score))

    def _apply_selected_event_order(
        self,
        specs: list[KISQuery],
        rankings: list[list[SearchHit]],
    ) -> list[list[SearchHit]]:
        """Promote selected-video candidates that satisfy an enabled E(i-1) -> E(i) relation."""

        ordered_rankings = [list(hits) for hits in rankings]
        for index, spec in enumerate(specs):
            if index == 0 or not spec.requires_after_previous or not ordered_rankings[index]:
                continue
            previous_hits = ordered_rankings[index - 1]
            if not previous_hits:
                continue
            previous_max = max(float(hit.score) for hit in previous_hits)
            previous_candidates = [
                (
                    self._kis_frame_index(hit.frame_id, hit.metadata),
                    float(hit.score) / previous_max if previous_max > 0 else 0.0,
                )
                for hit in previous_hits[:10]
            ]
            current_hits = ordered_rankings[index]
            current_max = max(float(hit.score) for hit in current_hits)
            for hit in current_hits:
                position = self._kis_frame_index(hit.frame_id, hit.metadata)
                relation_score = 0.0
                matched_previous_position: int | None = None
                for previous_position, previous_score in previous_candidates:
                    if position <= previous_position:
                        continue
                    compactness = 1.0 / (1.0 + (position - previous_position) / 1000.0)
                    candidate_score = previous_score * compactness
                    if candidate_score > relation_score:
                        relation_score = candidate_score
                        matched_previous_position = previous_position
                relative_kis = float(hit.score) / current_max if current_max > 0 else 0.0
                hit.score = current_max * (0.55 * relative_kis + 0.45 * relation_score)
                hit.metadata["event_order"] = {
                    "requires_after_previous": True,
                    "satisfies_order": matched_previous_position is not None,
                    "previous_event_id": specs[index - 1].event_id,
                    "previous_frame_index": matched_previous_position,
                    "score": round(relation_score, 8),
                }
            ordered_rankings[index] = sorted(
                current_hits,
                key=lambda hit: -float(hit.score),
            )
        return ordered_rankings

    @staticmethod
    def _kis_frame_index(frame_id: str, metadata: dict[str, Any]) -> int:
        """Use native frame metadata when available, then fall back to ID."""
        raw_index = metadata.get("frame_index")
        if isinstance(raw_index, int):
            return raw_index
        if isinstance(raw_index, str) and raw_index.isdigit():
            return int(raw_index)
        match = re.search(r"_f(\d+)", frame_id, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _kis_caption(metadata: dict[str, Any]) -> str:
        for field in (
            "detailed_caption_vi",
            "caption_vi",
            "detailed_caption",
            "caption",
        ):
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    async def search_temporal_videos(
        self,
        query: str,
        *,
        batch_ids: list[str] | None = None,
        video_ids: list[str] | None = None,
        top_k_videos: int = 100,
        summary_weight: float = 0.75,
        kis_weight: float = 0.25,
        use_rerank: bool = False,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        event_options: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Rank videos by summary evidence and the existing frame-level KIS."""

        try:
            payload = await self.dev2.search_temporal_videos(
                {
                    "query": query,
                    "batch_ids": batch_ids or [],
                    "video_ids": video_ids or [],
                    # Pull a deeper summary pool before combining it with KIS.
                    "top_k_videos": min(100, max(50, top_k_videos * 5)),
                    # Dev2 owns summary retrieval and the one-shot temporal
                    # parse.  Frame KIS fusion happens here, where both Dev1
                    # visual embeddings and Dev2 captions are available.
                    "summary_weight": 1.0,
                    "event_weight": 0.0,
                }
            )
        except Exception as exc:
            raise UpstreamError(f"Failed to select temporal videos from Dev2: {exc}") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("Dev2 temporal video response must be an object")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise UpstreamError("Dev2 temporal video response has no data object")
        summary_candidates = data.get("candidates")
        if not isinstance(summary_candidates, list):
            summary_candidates = []

        # Keep frame-level video selection on the exact E1...En wording,
        # matching the regular KIS search and the later selected-video search.
        # The temporal query plan remains useful for summary retrieval above,
        # but rewriting an event here made a card's evidence diverge from the
        # same event queried after the user selects that video.
        kis_queries = (
            build_kis_queries(
                query,
                None,
                event_options=event_options or [],
                default_text_weight=text_weight,
                default_visual_weight=visual_weight,
                default_use_rerank=use_rerank,
            )
            if kis_weight > 0
            else []
        )
        rerank_kis_events = bool(
            kis_queries and (use_rerank or any(spec.use_rerank for spec in kis_queries))
        )
        ordered_sequence = any(spec.requires_after_previous for spec in kis_queries)
        sequence_scope: list[str] = list(video_ids or [])
        broad_candidate_count = 0
        if ordered_sequence and not sequence_scope:
            # A batch-wide top-N list is too shallow to contain every stage
            # for the same video. First retrieve by the complete wording,
            # then evaluate each stage only inside a small candidate pool.
            broad_spec = KISQuery(
                event_id="C0",
                description=query,
                visual_prompt=query,
                prompt_variants=(),
                semantic_keywords=(query,),
            )
            broad_hits = await self._search_event_with_standard_kis(
                broad_spec,
                top_k=min(500, max(300, top_k_videos * 5)),
                batch_ids=batch_ids or [],
                video_ids=[],
                use_rerank=False,
                text_weight=text_weight,
                visual_weight=visual_weight,
            )
            broad_scores, _ = aggregate_kis_rankings([(broad_spec, broad_hits)])
            broad_ids = sorted(
                broad_scores,
                key=lambda video_id: (-broad_scores[video_id], video_id),
            )[:30]
            summary_ids = [
                str(item.get("video_id", "")).upper()
                for item in summary_candidates[:10]
                if isinstance(item, dict) and str(item.get("video_id", "")).strip()
            ]
            sequence_scope = list(dict.fromkeys([*broad_ids, *summary_ids]))[:35]
            broad_candidate_count = len(broad_ids)

        frame_pool = min(
            500,
            max(200, (len(sequence_scope) if ordered_sequence else top_k_videos) * 12),
        )
        kis_rankings: list[tuple[KISQuery, list[SearchHit]]] = []
        if kis_queries:
            ranking_results = await asyncio.gather(
                *(
                    self._search_event_with_standard_kis(
                        spec,
                        top_k=frame_pool,
                        batch_ids=batch_ids or [],
                        video_ids=sequence_scope if ordered_sequence else (video_ids or []),
                        use_rerank=use_rerank or spec.use_rerank,
                        text_weight=spec.text_weight,
                        visual_weight=spec.visual_weight,
                    )
                    for spec in kis_queries
                )
            )
            kis_rankings = list(zip(kis_queries, ranking_results))

        if kis_rankings:
            verified_rankings = await asyncio.gather(
                *(
                    self._apply_camera_motion_verification(spec, hits)
                    for spec, hits in kis_rankings
                )
            )
            kis_rankings = list(zip(kis_queries, verified_rankings))

        kis_scores, kis_evidence = aggregate_kis_rankings(kis_rankings)
        # Summary search returns only documents matching the query. A KIS hit
        # can therefore point to a valid video whose stored summary is outside
        # that result pool. Hydrate its card by exact ID without changing its
        # summary score (zero) or its KIS score.
        enriched_summaries = [dict(item) for item in summary_candidates if isinstance(item, dict)]
        by_video_id = {
            str(item.get("video_id", "")).upper(): item
            for item in enriched_summaries
            if str(item.get("video_id", "")).strip()
        }
        summary_lookup_ids = sorted(
            video_id
            for video_id in set(by_video_id) | set(kis_scores)
            if not str(by_video_id.get(video_id, {}).get("summary_vi", "")).strip()
        )
        if summary_lookup_ids:
            try:
                summary_payload = await self.dev2.get_video_summaries(summary_lookup_ids)
                summary_data = summary_payload.get("data", {}) if isinstance(summary_payload, dict) else {}
                looked_up = summary_data.get("summaries", {}) if isinstance(summary_data, dict) else {}
                if isinstance(looked_up, dict):
                    for video_id, source in looked_up.items():
                        if not isinstance(source, dict):
                            continue
                        normalized_id = str(video_id).upper()
                        candidate = by_video_id.get(normalized_id)
                        if candidate is None:
                            candidate = {"video_id": normalized_id, "summary_score": 0.0}
                            by_video_id[normalized_id] = candidate
                            enriched_summaries.append(candidate)
                        candidate.update(
                            {
                                field: source[field]
                                for field in ("summary_vi", "summary_en", "content_profile")
                                if source.get(field)
                            }
                        )
            except Exception:
                # Keep retrieval available if an older semantic service does
                # not expose summary hydration yet; the card will say so.
                pass
        candidates = fuse_summary_and_kis(
            enriched_summaries,
            kis_scores,
            kis_evidence,
            summary_weight=summary_weight,
            kis_weight=kis_weight,
            top_k=top_k_videos,
        )
        weight_total = summary_weight + kis_weight
        weights = {
            "summary": round(summary_weight / weight_total, 8),
            "kis": round(kis_weight / weight_total, 8),
        }
        data.update(
            {
                "mode": "temporal_video_selection_summary_kis",
                "selected_video_id": candidates[0]["video_id"] if candidates else None,
                "candidates": candidates,
                "weights": weights,
                "kis": {
                    "query_count": len(kis_queries),
                    "frame_pool": frame_pool,
                    "video_hits": len(kis_scores),
                    "ordered_sequence": ordered_sequence,
                    "candidate_scope_size": len(sequence_scope) if ordered_sequence else 0,
                    "broad_candidate_count": broad_candidate_count,
                    "gemini_rerank": rerank_kis_events,
                    "weights": {
                        "text": text_weight,
                        "visual": visual_weight,
                    },
                    "queries": [
                        {
                            "event_id": spec.event_id,
                            "description": spec.description,
                            "visual_prompt": spec.visual_prompt,
                            "weights": {
                                "text": spec.text_weight,
                                "visual": spec.visual_weight,
                            },
                            "requires_after_previous": spec.requires_after_previous,
                            "verify_camera_motion": spec.verify_camera_motion,
                            "motion_weight": spec.motion_weight,
                        }
                        for spec in kis_queries
                    ],
                },
                "video_selection": {
                    "mode": "summary_kis_fusion",
                    "selected_video_id": candidates[0]["video_id"] if candidates else None,
                    "candidates": candidates,
                    "weights": weights,
                },
            }
        )
        return payload

    async def get_video_summaries(self, video_ids: list[str]) -> dict[str, Any]:
        """Return exact indexed summaries for UI display, never for ranking."""
        try:
            payload = await self.dev2.get_video_summaries(video_ids)
        except Exception as exc:
            raise UpstreamError(f"Failed to fetch video summaries from Dev2: {exc}") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        summaries = data.get("summaries") if isinstance(data, dict) else None
        if not isinstance(summaries, dict):
            raise UpstreamError("Dev2 video summaries response has no summaries object")
        return {"status": "success", "data": {"summaries": summaries}}
