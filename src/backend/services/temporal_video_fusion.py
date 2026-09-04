"""Helpers for fusing video summaries with frame-level KIS retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from backend.schemas.search import SearchHit


EVENT_LINE_RE = re.compile(r"(?im)^\s*(E\d+)\s*[:：]\s*(.+?)\s*$")
SEQUENCE_CUE_RE = re.compile(
    r"\b(?:lần\s+lượt|chuyển\s+cảnh|sau\s+đó|tiếp\s+theo|cuối\s+cùng|"
    r"đầu\s+tiên|thứ\s+hai|thứ\s+ba|then|next|finally|sequence|shot)\b",
    flags=re.IGNORECASE,
)
INNER_SEQUENCE_BREAK_RE = re.compile(
    r"\s+(?:rồi|sau\s+đó|tiếp\s+theo|then)\s+"
    r"(?=(?:chuyển|xuất\s+hiện|cho\s+thấy|cảnh|máy\s+quay|quay|cú\s+máy))",
    flags=re.IGNORECASE,
)
SEAFOOD_RE = re.compile(r"\b(?:hải\s+sản|seafood)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class KISQuery:
    event_id: str
    description: str
    visual_prompt: str
    prompt_variants: tuple[str, ...]
    semantic_keywords: tuple[str, ...]
    sequence_position: int | None = None
    text_weight: float = 0.5
    visual_weight: float = 0.5
    use_rerank: bool = False
    requires_after_previous: bool = False
    verify_camera_motion: bool = False
    motion_weight: float = 0.7


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


def _observable_sequence_prompt(context: str, clause: str) -> str:
    """Add observable aliases without assuming the exact ingredient identity."""

    parts = [item for item in (context, clause) if item]
    prompt = ". ".join(parts)
    if SEAFOOD_RE.search(clause):
        prompt += (
            ". Nguyên liệu hải sản có thể còn nguyên hình hoặc đã được chế biến, "
            "ví dụ tôm, cá, thịt cá xay, chả cá hay cá thác lác."
        )
    return prompt


def _sequence_plan(query: str) -> dict[str, Any] | None:
    """Split only explicitly sequential prose into observable anchors.

    Ordinary KIS requests stay on their original one-frame path. This router
    activates only when the request contains ordering/camera cues and at least
    two semicolon-delimited (or connector-delimited) stages.
    """

    if EVENT_LINE_RE.search(query) or not SEQUENCE_CUE_RE.search(query):
        return None

    normalized = " ".join(query.split())
    head, separator, tail = normalized.partition(":")
    context = head.strip(" :;,-") if separator else ""
    stage_text = tail if separator else normalized
    clauses: list[str] = []
    for outer in re.split(r"\s*;\s*", stage_text):
        clauses.extend(INNER_SEQUENCE_BREAK_RE.split(outer))
    clauses = [item.strip(" .:;,-") for item in clauses if item.strip(" .:;,-")]
    if len(clauses) < 2:
        return None

    # A long sentence can contain many incidental connectors. Sequence
    # ranking is intended for a small shot/action chain, not arbitrary prose.
    clauses = clauses[:6]
    return {
        "video_context": context,
        "requires_order": True,
        "events": [
            {
                "event_id": f"S{index}",
                "description": clause,
                "retrieval_prompts": [_observable_sequence_prompt(context, clause)],
                "sequence_position": index,
            }
            for index, clause in enumerate(clauses, start=1)
        ],
    }


def _event_options_by_id(event_options: Any) -> dict[str, Any]:
    if not isinstance(event_options, (list, tuple)):
        return {}
    options: dict[str, Any] = {}
    for item in event_options:
        event_id = item.get("event_id") if isinstance(item, dict) else getattr(item, "event_id", None)
        if isinstance(event_id, str) and event_id.strip():
            options[event_id.upper()] = item
    return options


def _option_value(option: Any, name: str, default: Any) -> Any:
    if isinstance(option, dict):
        return option.get(name, default)
    return getattr(option, name, default)


def build_kis_queries(
    query: str,
    query_plan: Any,
    *,
    event_options: Any = (),
    infer_sequence: bool = False,
    default_text_weight: float = 0.5,
    default_visual_weight: float = 0.5,
    default_use_rerank: bool = False,
    limit: int = 8,
) -> list[KISQuery]:
    """Build one KIS request per TRAKE event from the single temporal parse."""

    plan = query_plan if isinstance(query_plan, dict) else (
        _sequence_plan(query) if infer_sequence else None
    )
    plan = plan or _fallback_plan(query)
    shared_context = " ".join(str(plan.get("video_context", "")).split())
    raw_events = plan.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raw_events = _fallback_plan(query)["events"]

    options_by_id = _event_options_by_id(event_options)
    has_order_constraint = infer_sequence or any(
        bool(_option_value(option, "requires_after_previous", False))
        for option in options_by_id.values()
    )
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
        event_id = str(raw_event.get("event_id") or f"E{index}").upper()
        option = options_by_id.get(event_id)
        text_weight = float(_option_value(option, "text_weight", default_text_weight))
        visual_weight = float(_option_value(option, "visual_weight", default_visual_weight))
        if text_weight < 0 or visual_weight < 0 or text_weight + visual_weight <= 0:
            text_weight, visual_weight = 0.5, 0.5
        result.append(
            KISQuery(
                event_id=event_id,
                description=description or prompts[0],
                visual_prompt=prompts[0],
                prompt_variants=tuple(prompts[1:]),
                semantic_keywords=tuple(keywords),
                sequence_position=index if has_order_constraint else None,
                text_weight=text_weight,
                visual_weight=visual_weight,
                use_rerank=bool(_option_value(option, "use_rerank", default_use_rerank)),
                requires_after_previous=bool(
                    _option_value(
                        option,
                        "requires_after_previous",
                        infer_sequence and index > 1,
                    )
                ),
                verify_camera_motion=bool(
                    _option_value(option, "verify_camera_motion", False)
                ),
                motion_weight=float(_option_value(option, "motion_weight", 0.7)),
            )
        )
    return result or build_kis_queries(
        query,
        _fallback_plan(query),
        event_options=event_options,
        infer_sequence=infer_sequence,
        default_text_weight=default_text_weight,
        default_visual_weight=default_visual_weight,
        default_use_rerank=default_use_rerank,
        limit=limit,
    )


def video_id_from_frame_id(frame_id: str) -> str:
    marker = frame_id.upper().rfind("_F")
    return frame_id[:marker].upper() if marker > 0 else ""


def aggregate_kis_rankings(
    rankings: Sequence[tuple[KISQuery, Sequence[SearchHit]]],
    *,
    evidence_per_event: int = 3,
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Aggregate frame rankings with optional ordered-chain evidence."""

    if not rankings:
        return {}, {}

    per_event_scores: list[dict[str, float]] = []
    ordered_sequence = any(
        spec.requires_after_previous for spec, _ in rankings
    )
    sequence_candidates: list[dict[str, list[tuple[int, float, str]]]] = []
    sequence_evidence_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for spec, hits in rankings:
        max_hit_score = max((float(hit.score) for hit in hits), default=0.0)
        raw_video_scores: dict[str, list[float]] = {}
        event_sequence_candidates: dict[str, list[tuple[int, float, str]]] = {}
        event_evidence_counts: dict[str, int] = {}
        for rank, hit in enumerate(hits, start=1):
            video_id = (hit.video_name or video_id_from_frame_id(hit.frame_id)).upper()
            if not video_id:
                continue
            relative_score = float(hit.score) / max_hit_score if max_hit_score > 0 else 0.0
            raw_video_scores.setdefault(video_id, []).append(relative_score)
            metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
            caption = (
                metadata.get("detailed_caption_vi")
                or metadata.get("caption_vi")
                or metadata.get("detailed_caption")
                or metadata.get("caption")
                or ""
            )
            if ordered_sequence:
                position = hit.frame_index
                if not isinstance(position, int):
                    frame_match = re.search(r"_f(\d+)", hit.frame_id, flags=re.IGNORECASE)
                    position = int(frame_match.group(1)) if frame_match else -1
                if position >= 0:
                    event_sequence_candidates.setdefault(video_id, []).append(
                        (position, relative_score, hit.frame_id)
                    )
                    sequence_evidence_lookup[
                        (video_id, int(spec.sequence_position or 0), hit.frame_id)
                    ] = {
                        "event_id": spec.event_id,
                        "frame_id": hit.frame_id,
                        "rank": rank,
                        "score": round(relative_score, 8),
                        "thumbnail_url": hit.thumbnail_url,
                        "caption": str(caption)[:500],
                        "source_ranks": metadata.get("source_ranks", {}),
                        "sequence_position": spec.sequence_position,
                    }
            count = event_evidence_counts.get(video_id, 0)
            if count >= evidence_per_event:
                continue
            evidence.setdefault(video_id, []).append(
                {
                    "event_id": spec.event_id,
                    "frame_id": hit.frame_id,
                    "rank": rank,
                    "score": round(relative_score, 8),
                    "thumbnail_url": hit.thumbnail_url,
                    "caption": str(caption)[:500],
                    "source_ranks": metadata.get("source_ranks", {}),
                    "sequence_position": spec.sequence_position,
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
        if ordered_sequence:
            sequence_candidates.append(event_sequence_candidates)

    candidate_ids = set().union(*(scores for scores in per_event_scores))
    query_count = len(per_event_scores)
    base_scores: dict[str, float] = {}
    for video_id in candidate_ids:
        values = [event_scores.get(video_id, 0.0) for event_scores in per_event_scores]
        mean_score = sum(values) / query_count
        if query_count == 1:
            base_scores[video_id] = mean_score
            continue
        coverage = sum(value > 0 for value in values) / query_count
        base_scores[video_id] = 0.85 * mean_score + 0.15 * coverage

    if not ordered_sequence:
        return base_scores, evidence

    relation_indexes = [
        index
        for index, (spec, _) in enumerate(rankings)
        if index > 0 and spec.requires_after_previous
    ]
    relation_scores: dict[str, list[float]] = {video_id: [] for video_id in candidate_ids}
    selected_pairs: dict[str, set[tuple[int, str]]] = {
        video_id: set() for video_id in candidate_ids
    }
    for index in relation_indexes:
        raw_scores: dict[str, tuple[float, tuple[int, str], tuple[int, str]]] = {}
        for video_id in candidate_ids:
            previous = sorted(
                sequence_candidates[index - 1].get(video_id, []),
                key=lambda item: -item[1],
            )[:12]
            current = sorted(
                sequence_candidates[index].get(video_id, []),
                key=lambda item: -item[1],
            )[:12]
            best: tuple[float, tuple[int, str], tuple[int, str]] | None = None
            for previous_position, previous_score, previous_id in previous:
                for current_position, current_score, current_id in current:
                    if current_position <= previous_position:
                        continue
                    span = current_position - previous_position
                    compactness = 1.0 / (1.0 + span / 1000.0)
                    score = ((previous_score + current_score) / 2.0) * math.sqrt(compactness)
                    candidate = (
                        score,
                        (index, previous_id),
                        (index + 1, current_id),
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                raw_scores[video_id] = best
        maximum = max((item[0] for item in raw_scores.values()), default=0.0)
        for video_id in candidate_ids:
            found = raw_scores.get(video_id)
            normalized = found[0] / maximum if found is not None and maximum > 0 else 0.0
            relation_scores[video_id].append(normalized)
            if found is not None:
                selected_pairs[video_id].update(found[1:])

    scores: dict[str, float] = {}
    for video_id in candidate_ids:
        order_score = (
            sum(relation_scores[video_id]) / len(relation_indexes)
            if relation_indexes
            else 0.0
        )
        # Order is an explicit constraint. It strongly reranks only videos
        # that can satisfy every selected "after previous" relation.
        scores[video_id] = 0.55 * base_scores[video_id] + 0.45 * order_score
        selected = selected_pairs[video_id]
        video_evidence = evidence.setdefault(video_id, [])
        existing = {
            (item.get("sequence_position"), item.get("frame_id"))
            for item in video_evidence
        }
        for stage_position, frame_id in selected:
            if (stage_position, frame_id) in existing:
                continue
            selected_detail = sequence_evidence_lookup.get(
                (video_id, stage_position, frame_id)
            )
            if selected_detail:
                video_evidence.append(dict(selected_detail))
        for item in video_evidence:
            item["sequence_selected"] = (
                item.get("sequence_position"),
                item.get("frame_id"),
            ) in selected
            item["sequence_score"] = round(order_score, 8)
        video_evidence.sort(
            key=lambda item: (
                int(item.get("sequence_position") or 0),
                not bool(item.get("sequence_selected")),
                int(item.get("rank") or 0),
            )
        )
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
