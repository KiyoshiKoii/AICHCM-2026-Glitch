"""Same-video, ordered event retrieval for BTC E1..En questions."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from semantic_pipeline.retrieval.hierarchical_video_search import build_hierarchical_documents
from semantic_pipeline.retrieval.temporal_query_parser import (
    CONCEPT_STOPWORDS,
    ParsedTemporalQuery,
    TemporalEventQuery,
    fold_text,
    parse_temporal_query,
    query_tokens,
)
from semantic_pipeline.video_understanding.temporal_events import build_temporal_events, event_search_text


TEMPORAL_STOPWORDS = {
    "first", "earliest", "start", "starts", "started", "begin", "begins", "beginning",
    "last", "latest", "final", "end", "ends", "ended", "complete", "completed",
    "completely", "fully", "moment", "momentos", "khoanh", "khac", "dau", "tien",
    "lan", "cuoi", "cung", "bat", "dau", "hoan", "toan", "sau", "do", "then",
    "visible", "shown", "show", "see", "seen", "thay",
    "is", "are", "the", "a", "an", "about", "story", "time", "people", "facing",
}


@dataclass(frozen=True)
class TemporalCorpus:
    video: dict[str, Any]
    events: list[dict[str, Any]]
    frames: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _event_documents(
    timeline: dict[str, Any],
    frames: list[dict[str, Any]],
    video_id: str,
) -> list[dict[str, Any]]:
    raw_events = timeline.get("events")
    if isinstance(raw_events, list) and raw_events:
        return [dict(item) for item in raw_events if isinstance(item, dict) and item.get("event_id")]
    # Compatibility for pilots generated before video-timeline-v2.
    generated = build_temporal_events(
        timeline.get("segments", []),
        [_RuntimeFrameProxy(item) for item in frames],
        content_profile=str(timeline.get("content_profile", "general_event")),
    )
    return generated


class _RuntimeFrameProxy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.frame_id = payload["frame_id"]
        self.keyframe_n = int(payload["keyframe_n"])
        self.timestamp_ms = int(payload["timestamp_ms"])
        self.native_frame_idx = int(payload["native_frame_idx"])
        self.raw_metadata = {
            "caption": payload.get("visual_text", ""),
            "asr_text": payload.get("asr_text", ""),
            "ocr_text": payload.get("ocr_text", ""),
        }


def discover_temporal_corpus(
    *,
    output_root: Path,
    caption_dir: Path,
    asr_dir: Path,
    map_dir: Path,
    keyframe_dir: Path,
    batch_ids: Iterable[str] = (),
    video_ids: Iterable[str] = (),
) -> list[TemporalCorpus]:
    wanted_batches = {item.upper() for item in batch_ids}
    wanted_videos = {item.upper() for item in video_ids}
    corpora: list[TemporalCorpus] = []
    for pilot_dir in sorted(output_root.glob("L*/L*_V*/pilot")):
        video_id = pilot_dir.parent.name.upper()
        batch_id = pilot_dir.parent.parent.name.upper()
        if wanted_batches and batch_id not in wanted_batches:
            continue
        if wanted_videos and video_id not in wanted_videos and video_id.rsplit("_", 1)[-1] not in wanted_videos:
            continue
        try:
            documents = build_hierarchical_documents(
                pilot_dir=pilot_dir,
                caption_path=caption_dir / batch_id / f"{video_id}.json",
                asr_path=asr_dir / f"{video_id}.json",
                map_path=map_dir / f"{video_id}.csv",
                keyframe_dir=keyframe_dir / video_id,
            )
            timeline = _read_json(pilot_dir / "timeline.json")
            events = _event_documents(timeline, documents.frames, video_id)
            frame_by_n = {int(frame["keyframe_n"]): frame for frame in documents.frames}
            for event in events:
                event["video_id"] = video_id
                event.setdefault("keyframe_refs", [])
                event["search_text"] = event_search_text(event)
                event["_frames"] = [
                    frame_by_n[n]
                    for n in event.get("keyframe_refs", [])
                    if isinstance(n, int) and n in frame_by_n
                ]
            corpora.append(TemporalCorpus(documents.video, events, documents.frames))
        except (OSError, ValueError, json.JSONDecodeError):
            # A partially generated batch should not make valid videos
            # unsearchable.  Validation remains visible in each pilot report.
            continue
    if not corpora:
        raise FileNotFoundError(f"No valid temporal pilots found under {output_root}")
    return corpora


def _score(query: set[str], text: str) -> float:
    useful = query - TEMPORAL_STOPWORDS
    if not useful:
        useful = query
    matched = useful & query_tokens(text)
    return len(matched) / len(useful) if useful else 0.0


def _content_tokens(value: str, *, fold_accents: bool = False) -> list[str]:
    """Keep searchable terms in order while removing query boilerplate."""

    normalized = fold_text(value) if fold_accents else unicodedata.normalize("NFC", value.casefold())
    result: list[str] = []
    for token in re.findall(r"[^\W_]+", normalized):
        folded_token = fold_text(token)
        if (
            len(token) > 1
            and folded_token not in CONCEPT_STOPWORDS
            and folded_token not in TEMPORAL_STOPWORDS
        ):
            result.append(token)
    return result


def _tokens_fit(required: list[str], available: list[str]) -> bool:
    required_counts = Counter(required)
    available_counts = Counter(available)
    return all(available_counts[token] >= count for token, count in required_counts.items())


def _matches_concept_groups(
    concept_groups: tuple[tuple[str, ...], ...],
    text: str,
) -> tuple[float, list[str]]:
    """Return locally coherent concept coverage and supported aliases.

    Alternatives may be paraphrased and reordered, but their evidence must
    occur inside one compact window. This prevents unrelated mentions such as
    ``thanh pho``, ``Long Binh`` and ``den tho`` from collectively matching
    ``pho long den``.
    """

    if not concept_groups:
        return 0.0, []
    strict_text_sequence = _content_tokens(text)
    folded_text_sequence = _content_tokens(text, fold_accents=True)
    if not strict_text_sequence:
        return 0.0, []
    prepared_groups: list[list[tuple[str, list[str], bool]]] = []
    for alternatives in concept_groups:
        prepared_alternatives: list[tuple[str, list[str], bool]] = []
        for alternative in alternatives:
            normalized_alternative = unicodedata.normalize("NFC", alternative.casefold())
            fold_accents = fold_text(alternative) == normalized_alternative
            tokens = _content_tokens(alternative, fold_accents=fold_accents)
            if tokens:
                prepared_alternatives.append((alternative, tokens, fold_accents))
        if prepared_alternatives:
            prepared_groups.append(prepared_alternatives)
    if not prepared_groups:
        return 0.0, []
    minimum_required_terms = sum(
        min((len(tokens) for _, tokens, _ in alternatives), default=1)
        for alternatives in prepared_groups
    )
    # Allow natural modifiers and word-order changes without letting evidence
    # drift across unrelated facts in a long story.
    window_size = min(36, max(12, minimum_required_terms * 3))
    best_matches: list[str] = []
    for start in range(len(strict_text_sequence)):
        strict_window = strict_text_sequence[start : start + window_size]
        folded_window = folded_text_sequence[start : start + window_size]
        matches: list[str] = []
        for alternatives in prepared_groups:
            matched = next(
                (
                    alternative
                    for alternative, tokens, fold_accents in alternatives
                    if _tokens_fit(tokens, folded_window if fold_accents else strict_window)
                ),
                None,
            )
            if matched:
                matches.append(matched)
        if len(matches) > len(best_matches):
            best_matches = matches
        if len(best_matches) == len(prepared_groups):
            break
    return len(best_matches) / len(prepared_groups), best_matches


def _best_concept_match(
    concept_groups: tuple[tuple[str, ...], ...],
    evidence_units: Iterable[str],
) -> tuple[float, list[str]]:
    """Do not let concepts match across unrelated fields or events."""

    best: tuple[float, list[str]] = (0.0, [])
    for unit in evidence_units:
        candidate = _matches_concept_groups(concept_groups, unit)
        if candidate[0] > best[0]:
            best = candidate
        if best[0] >= 1.0:
            break
    return best


def _metadata_context_matches(
    video: dict[str, Any],
    context_groups: tuple[tuple[str, ...], ...],
) -> list[str]:
    """Expose metadata that supports context without boosting a lone token."""

    matches: list[str] = []
    for field in ("main_locations", "main_entities"):
        values = video.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            candidate = " ".join(str(value).split())
            coverage, _ = _matches_concept_groups(context_groups, candidate)
            if coverage > 0:
                matches.append(candidate)
    return list(dict.fromkeys(matches))


def _anchor_for(event: dict[str, Any], query: TemporalEventQuery) -> dict[str, Any] | None:
    anchors = [item for item in event.get("temporal_anchors", []) if isinstance(item, dict)]
    preferred = {
        "action_start": {"action_start", "first_contact"},
        "first_contact": {"first_contact", "action_start"},
        "state_complete": {"state_complete", "first_complete", "representative"},
        "last_complete": {"last_complete", "action_end"},
        "action_end": {"action_end", "last_complete"},
        "representative": {"representative"},
        "first_visible": {"first_visible", "action_start", "representative"},
    }.get(query.required_anchor, {"representative"})
    if query.required_anchor == "first_visible":
        frames = sorted(
            event.get("_frames", []),
            key=lambda item: (item["timestamp_ms"], item["keyframe_n"]),
        )
        best_score = max(
            (
                _score(
                    set(query.tokens),
                    " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
                )
                for frame in frames
            ),
            default=0.0,
        )
        if best_score > 0:
            threshold = best_score * 0.7
            for frame in frames:
                score = _score(
                    set(query.tokens),
                    " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
                )
                if score >= threshold:
                    return {
                        "anchor_type": "first_visible",
                        "frame_id": frame["frame_id"],
                        "keyframe_n": frame["keyframe_n"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "native_frame_idx": frame["native_frame_idx"],
                        "confidence": round(max(0.55, score), 6),
                    }
    for anchor in anchors:
        if anchor.get("anchor_type") in preferred:
            return anchor
    frames = sorted(event.get("_frames", []), key=lambda item: (item["timestamp_ms"], item["keyframe_n"]))
    if not frames:
        return None
    selected = frames[-1] if query.required_anchor in {"last_complete", "action_end"} else frames[0]
    return {
        "anchor_type": query.required_anchor,
        "frame_id": selected["frame_id"],
        "keyframe_n": selected["keyframe_n"],
        "timestamp_ms": selected["timestamp_ms"],
        "native_frame_idx": selected["native_frame_idx"],
        "confidence": 0.55,
    }


class TemporalEventSearch:
    def __init__(self, corpora: Iterable[TemporalCorpus], *, query_parser: Any | None = None) -> None:
        self.corpora = tuple(corpora)
        self.query_parser = query_parser

    @staticmethod
    def _event_evidence_units(event: dict[str, Any]) -> list[str]:
        return [
            str(event.get("description_vi", "")),
            str(event.get("search_text", "")),
            *[
                " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text"))
                for frame in event.get("_frames", [])
            ],
        ]

    def _event_concept_coverage(self, event: dict[str, Any], query: TemporalEventQuery) -> float:
        return _best_concept_match(
            query.required_concept_groups,
            self._event_evidence_units(event),
        )[0]

    def _rank_event(self, event: dict[str, Any], query: TemporalEventQuery) -> float:
        query_set = set(query.tokens)
        title_score = _score(query_set, str(event.get("description_vi", "")))
        event_score = _score(query_set, event.get("search_text", ""))
        frame_scores = [
            _score(
                query_set,
                " ".join(str(frame.get(field, "")) for field in ("visual_text", "asr_text", "ocr_text")),
            )
            for frame in event.get("_frames", [])
        ]
        frame_score = max(frame_scores, default=0.0)
        lexical_score = 0.55 * title_score + 0.25 * event_score + 0.20 * frame_score
        concept_score = self._event_concept_coverage(event, query)
        if query.required_concept_groups:
            return 0.25 * lexical_score + 0.75 * concept_score
        return lexical_score

    def _rank_video_context(
        self,
        corpus: TemporalCorpus,
        context_tokens: set[str],
        context_groups: tuple[tuple[str, ...], ...],
    ) -> tuple[float, list[str]]:
        """Rank a video from its summary/story corpus before event matching."""

        if not context_tokens:
            return 0.0, []
        video = corpus.video
        summary_score = _score(
            context_tokens,
            " ".join((str(video.get("summary_vi", "")), str(video.get("summary_en", "")))),
        )
        search_score = _score(context_tokens, str(video.get("search_text", "")))
        best_story_score = max(
            (_score(context_tokens, event.get("search_text", "")) for event in corpus.events),
            default=0.0,
        )
        base_score = 0.35 * summary_score + 0.35 * search_score + 0.30 * best_story_score
        video_concepts, _ = _best_concept_match(
            context_groups,
            (
                str(video.get("summary_vi", "")),
                str(video.get("summary_en", "")),
                str(video.get("search_text", "")),
            ),
        )
        best_event_concepts = max(
            (_matches_concept_groups(context_groups, event.get("search_text", ""))[0] for event in corpus.events),
            default=0.0,
        )
        concept_score = 0.35 * video_concepts + 0.65 * best_event_concepts
        metadata_matches = _metadata_context_matches(video, context_groups)
        if context_groups:
            return 0.35 * base_score + 0.65 * concept_score, metadata_matches
        return base_score, metadata_matches

    @staticmethod
    def _context_concept_coverage(
        corpus: TemporalCorpus,
        context_groups: tuple[tuple[str, ...], ...],
    ) -> float:
        video = corpus.video
        return _best_concept_match(
            context_groups,
            (
                str(video.get("summary_vi", "")),
                str(video.get("summary_en", "")),
                str(video.get("search_text", "")),
                *(str(event.get("search_text", "")) for event in corpus.events),
            ),
        )[0]

    def _best_review_candidate(
        self,
        corpus: TemporalCorpus,
        event_query: TemporalEventQuery,
    ) -> dict[str, Any] | None:
        """Return a video's best evidence frame without the answer threshold."""

        ranked = sorted(
            ((self._rank_event(event, event_query), event) for event in corpus.events),
            key=lambda item: (
                -item[0],
                int(item[1].get("start_ms", 0)),
                str(item[1].get("event_id", "")),
            ),
        )
        for score, event in ranked:
            anchor = _anchor_for(event, event_query)
            if anchor is None:
                continue
            return {
                "event_index": event_query.event_index,
                "source_label": event_query.source_label,
                "event_id": event["event_id"],
                "event_text": event.get("description_vi", ""),
                "anchor_type": anchor["anchor_type"],
                "frame_id": anchor["frame_id"],
                "keyframe_n": anchor["keyframe_n"],
                "native_frame_idx": anchor["native_frame_idx"],
                "timestamp_ms": anchor["timestamp_ms"],
                "score": round(score, 6),
                "confidence": anchor.get("confidence", event.get("confidence", 0.0)),
                "uncertain": bool(event.get("uncertain", True)),
                "reason_vi": event.get("description_vi", ""),
            }
        return None

    def search(
        self,
        query: str | ParsedTemporalQuery,
        *,
        top_k_videos: int = 20,
    ) -> dict[str, Any]:
        parsed = parse_temporal_query(query) if isinstance(query, str) else query
        if self.query_parser is not None:
            parsed = self.query_parser.parse(parsed)
        context_tokens = query_tokens(parsed.shared_context)
        ranked_videos: list[tuple[float, TemporalCorpus, list[dict[str, Any]], list[str]]] = []
        for corpus in self.corpora:
            context_score, context_matches = self._rank_video_context(
                corpus,
                context_tokens,
                parsed.context_concept_groups,
            )
            selected_events: list[dict[str, Any]] = []
            event_scores: list[float] = []
            used_event_ids: set[str] = set()
            for event_query in parsed.events:
                ranked = sorted(
                    (
                        (
                            self._rank_event(event, event_query)
                            - (0.15 if event.get("event_id") in used_event_ids else 0.0),
                            event,
                        )
                        for event in corpus.events
                    ),
                    key=lambda item: (-item[0], int(item[1].get("start_ms", 0)), str(item[1].get("event_id", ""))),
                )
                score, event = ranked[0] if ranked else (0.0, None)
                if ranked and event_query.required_anchor in {"action_start", "first_contact", "first_visible"}:
                    close_matches = [
                        item for item in ranked
                        if item[0] >= max(0.30, score * 0.80)
                    ]
                    if close_matches:
                        score, event = min(
                            close_matches,
                            key=lambda item: (
                                int(item[1].get("start_ms", 0)),
                                -item[0],
                                str(item[1].get("event_id", "")),
                            ),
                        )
                if event is None or score < 0.30:
                    event_scores.append(0.0)
                    continue
                anchor = _anchor_for(event, event_query)
                if anchor is None:
                    event_scores.append(0.0)
                    continue
                selected_events.append(
                    {
                        "event_index": event_query.event_index,
                        "source_label": event_query.source_label,
                        "event_id": event["event_id"],
                        "event_text": event.get("description_vi", ""),
                        "anchor_type": anchor["anchor_type"],
                        "frame_id": anchor["frame_id"],
                        "keyframe_n": anchor["keyframe_n"],
                        "native_frame_idx": anchor["native_frame_idx"],
                        "timestamp_ms": anchor["timestamp_ms"],
                        "score": round(score, 6),
                        "confidence": anchor.get("confidence", event.get("confidence", 0.0)),
                        "uncertain": bool(event.get("uncertain", True)),
                        "reason_vi": event.get("description_vi", ""),
                    }
                )
                used_event_ids.add(str(event.get("event_id", "")))
                event_scores.append(score)
            coverage = sum(score > 0 for score in event_scores) / max(1, len(parsed.events))
            ordering_bonus = 0.0
            timestamps = [item["timestamp_ms"] for item in selected_events]
            if timestamps and timestamps == sorted(timestamps):
                ordering_bonus = 0.1
            # The context is the video selector; events are resolved only
            # after that candidate video is identified.  This keeps a named
            # location in the summary from being outweighed by generic event
            # overlap in an unrelated news video.
            total = (
                0.65 * context_score
                + 0.20 * coverage
                + 0.10 * (sum(event_scores) / max(1, len(event_scores)))
                + 0.05 * ordering_bonus
            )
            total = min(1.0, max(0.0, total))
            ranked_videos.append((total, corpus, selected_events, context_matches))
        ranked_videos.sort(key=lambda item: (-item[0], str(item[1].video.get("video_id", ""))))
        summary_only = (
            len(parsed.events) == 1
            and parsed.events[0].text == parsed.shared_context
        )
        review_candidates: dict[str, dict[str, Any]] = {}
        if summary_only:
            eligible_videos = []
            for ranked_item in ranked_videos:
                corpus = ranked_item[1]
                if (
                    parsed.context_concept_groups
                    and self._context_concept_coverage(corpus, parsed.context_concept_groups) < 1.0
                ):
                    continue
                review_candidate = self._best_review_candidate(
                    corpus,
                    parsed.events[0],
                )
                if review_candidate is None:
                    continue
                review_candidates[str(corpus.video["video_id"])] = review_candidate
                eligible_videos.append(ranked_item)
            selected = eligible_videos[: max(1, top_k_videos)]
        else:
            selected = ranked_videos[: max(1, top_k_videos)]
        if not selected:
            return {"query": parsed.shared_context, "selected_video": None, "videos": [], "events": []}
        best_score, best_corpus, best_events, best_context_matches = selected[0]
        best_events.sort(key=lambda item: item["event_index"])
        candidates: list[dict[str, Any]] = []
        for video_score, corpus, events, context_matches in selected:
            best_event = review_candidates.get(str(corpus.video["video_id"]))
            if best_event is None:
                best_event = self._best_review_candidate(corpus, parsed.events[0])
            if best_event is None:
                continue
            candidates.append(
                {
                    **best_event,
                    "rank": len(candidates) + 1,
                    "video_id": corpus.video["video_id"],
                    "video_score": round(video_score, 6),
                    "matched_context_entities": context_matches,
                }
            )
        return {
            "query": query if isinstance(query, str) else parsed.shared_context,
            "mode": "temporal_event_search",
            "query_parsing": {
                "mode": getattr(self.query_parser, "mode", "deterministic"),
                "model": getattr(self.query_parser, "model", None),
            },
            "selected_video": {
                "video_id": best_corpus.video["video_id"],
                "score": round(best_score, 6),
                "matched_events": len(best_events),
                "total_events": len(parsed.events),
                "event_coverage": round(len(best_events) / max(1, len(parsed.events)), 6),
                "matched_context_entities": best_context_matches,
            },
            "events": best_events,
            # One best evidence frame per ranked video. This is the review
            # surface for summary-only searches; the selected video's E1..En
            # answers remain available separately in ``events``.
            "candidates": candidates,
            "videos": [
                {
                    "video_id": corpus.video["video_id"],
                    "score": round(score, 6),
                    "matched_events": len(events),
                    "total_events": len(parsed.events),
                }
                for score, corpus, events, context_matches in selected
            ],
        }
