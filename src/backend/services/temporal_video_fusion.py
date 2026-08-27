"""Helpers for fusing video summaries with frame-level KIS retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from backend.schemas.search import SearchHit


EVENT_LINE_RE = re.compile(r"(?im)^\s*(E\d+)\s*[:：]\s*(.+?)\s*$")


@dataclass(frozen=True)
class KISQuery:
    event_id: str
    description: str
    visual_prompt: str
    prompt_variants: tuple[str, ...]
    semantic_keywords: tuple[str, ...]


def _clean_strings(values: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            normalized
            for value in values
            if isinstance(value, str) and (normalized := " ".join(value.split()))
        )
    )[:limit]


def _fallback_plan(query: str) -> dict[str, Any]:
    matches = list(EVENT_LINE_RE.finditer(query))
    if not matches:
        normalized = " ".join(query.split())
        return {
            "video_context": normalized,
            "events": [
                {
                    "event_id": "E1",
                    "description": normalized,
                    "retrieval_prompts": [normalized],
                }
            ],
        }
    return {
        "video_context": " ".join(query[: matches[0].start()].split()).strip(" :;,-"),
        "events": [
            {
                "event_id": match.group(1).upper(),
                "description": " ".join(match.group(2).split()),
                "retrieval_prompts": [" ".join(match.group(2).split())],
            }
            for match in matches
        ],
    }


def build_kis_queries(query: str, query_plan: Any, *, limit: int = 8) -> list[KISQuery]:
    """Build one KIS request per TRAKE event from the single temporal parse."""

    plan = query_plan if isinstance(query_plan, dict) else _fallback_plan(query)
    shared_context = " ".join(str(plan.get("video_context", "")).split())
    raw_events = plan.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raw_events = _fallback_plan(query)["events"]

    result: list[KISQuery] = []
    for index, raw_event in enumerate(raw_events[:limit], start=1):
        if not isinstance(raw_event, dict):
            continue
        description = " ".join(str(raw_event.get("description", "")).split())
        prompts = _clean_strings(raw_event.get("retrieval_prompts"), limit=3)
        target_predicates = _clean_strings(raw_event.get("target_predicates"), limit=3)
        context_predicates = _clean_strings(raw_event.get("context_predicates"), limit=4)
        if not description and not prompts:
            continue
        if not prompts:
            prompts = [description]
        keywords = list(
            dict.fromkeys(
                item
                for item in [
                    shared_context,
                    description,
                    *prompts,
                    *target_predicates,
                    *context_predicates,
                ]
                if item
            )
        )[:20]
        result.append(
            KISQuery(
                event_id=str(raw_event.get("event_id") or f"E{index}").upper(),
                description=description or prompts[0],
                visual_prompt=prompts[0],
                prompt_variants=tuple(prompts[1:]),
                semantic_keywords=tuple(keywords),
            )
        )
    return result or build_kis_queries(query, _fallback_plan(query), limit=limit)


def video_id_from_frame_id(frame_id: str) -> str:
    marker = frame_id.upper().rfind("_F")
    return frame_id[:marker].upper() if marker > 0 else ""


def aggregate_kis_rankings(
    rankings: Sequence[tuple[KISQuery, Sequence[SearchHit]]],
    *,
    evidence_per_event: int = 3,
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Aggregate frame rankings into normalized video scores with event coverage."""

    if not rankings:
        return {}, {}

    per_event_scores: list[dict[str, float]] = []
    evidence: dict[str, list[dict[str, Any]]] = {}
    for spec, hits in rankings:
        max_hit_score = max((float(hit.score) for hit in hits), default=0.0)
        raw_video_scores: dict[str, list[float]] = {}
        event_evidence_counts: dict[str, int] = {}
        for rank, hit in enumerate(hits, start=1):
            video_id = (hit.video_name or video_id_from_frame_id(hit.frame_id)).upper()
            if not video_id:
                continue
            relative_score = float(hit.score) / max_hit_score if max_hit_score > 0 else 0.0
            raw_video_scores.setdefault(video_id, []).append(relative_score)
            count = event_evidence_counts.get(video_id, 0)
            if count >= evidence_per_event:
                continue
            metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
            caption = (
                metadata.get("detailed_caption_vi")
                or metadata.get("caption_vi")
                or metadata.get("detailed_caption")
                or metadata.get("caption")
                or ""
            )
            evidence.setdefault(video_id, []).append(
                {
                    "event_id": spec.event_id,
                    "frame_id": hit.frame_id,
                    "rank": rank,
                    "score": round(relative_score, 8),
                    "thumbnail_url": hit.thumbnail_url,
                    "caption": str(caption)[:500],
                    "source_ranks": metadata.get("source_ranks", {}),
                }
            )
            event_evidence_counts[video_id] = count + 1

        event_scores: dict[str, float] = {}
        for video_id, values in raw_video_scores.items():
            top = sorted(values, reverse=True)[:3]
            weights = (1.0, 0.25, 0.1)
            event_scores[video_id] = sum(
                score * weights[index] for index, score in enumerate(top)
            ) / sum(weights[: len(top)])
        max_video_score = max(event_scores.values(), default=0.0)
        if max_video_score > 0:
            event_scores = {
                video_id: score / max_video_score for video_id, score in event_scores.items()
            }
        per_event_scores.append(event_scores)

    candidate_ids = set().union(*(scores for scores in per_event_scores))
    query_count = len(per_event_scores)
    scores: dict[str, float] = {}
    for video_id in candidate_ids:
        values = [event_scores.get(video_id, 0.0) for event_scores in per_event_scores]
        mean_score = sum(values) / query_count
        if query_count == 1:
            scores[video_id] = mean_score
            continue
        coverage = sum(value > 0 for value in values) / query_count
        scores[video_id] = 0.85 * mean_score + 0.15 * coverage
    return scores, evidence


def fuse_summary_and_kis(
    summary_candidates: Sequence[dict[str, Any]],
    kis_scores: dict[str, float],
    kis_evidence: dict[str, list[dict[str, Any]]],
    *,
    summary_weight: float,
    kis_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if summary_weight < 0 or kis_weight < 0:
        raise ValueError("video selection weights must be non-negative")
    total_weight = summary_weight + kis_weight
    if total_weight <= 0:
        raise ValueError("summary_weight and kis_weight cannot both be zero")
    normalized_summary_weight = summary_weight / total_weight
    normalized_kis_weight = kis_weight / total_weight

    summaries = {
        str(candidate.get("video_id", "")).upper(): dict(candidate)
        for candidate in summary_candidates
        if str(candidate.get("video_id", "")).strip()
    }
    # KIS and summary rank over different evidence sources. Keep their union
    # so the user-controlled fusion weight can promote a strong KIS match,
    # even when that video did not make the lexical summary result pool.
    candidate_ids = set(summaries) | set(kis_scores)
    fused: list[dict[str, Any]] = []
    for video_id in candidate_ids:
        summary = summaries.get(video_id, {})
        summary_score = float(summary.get("summary_score") or 0.0)
        kis_score = float(kis_scores.get(video_id, 0.0))
        candidate = {
            key: value
            for key, value in summary.items()
            if key not in {"score", "event_score", "segment_evidence", "weights"}
        }
        candidate.update(
            {
                "video_id": video_id,
                "score": round(
                    normalized_summary_weight * summary_score
                    + normalized_kis_weight * kis_score,
                    8,
                ),
                "summary_score": round(summary_score, 8),
                "kis_score": round(kis_score, 8),
                "kis_evidence": kis_evidence.get(video_id, []),
            }
        )
        fused.append(candidate)
    fused.sort(key=lambda item: (-item["score"], item["video_id"]))
    return fused[:top_k]
