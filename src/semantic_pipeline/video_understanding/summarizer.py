"""Evidence-grounded news summarization with a deterministic fallback."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

from .models import ASRSegment, MicroScene, RuntimeFrame, StoryCandidate

try:  # Optional at import time so offline validation remains lightweight.
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - depends on the developer environment.
    genai = None
    types = None


STOPWORDS = {
    "the",
    "and",
    "with",
    "this",
    "that",
    "trong",
    "của",
    "và",
    "cho",
    "một",
    "các",
    "đang",
    "được",
    "với",
}
TRANSITION_RE = re.compile(
    r"\b(tiếp theo|sau đó|chuyển sang|trong một diễn biến khác|bản tin tiếp theo)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {
        item.casefold()
        for item in WORD_RE.findall(text or "")
        if len(item) > 2 and item.casefold() not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _frame_text(frame: RuntimeFrame) -> str:
    raw = frame.raw_metadata
    parts = [
        str(raw.get("caption", "")),
        str(raw.get("detailed_caption", "")),
        str(raw.get("caption_vi", "")),
        str(raw.get("detailed_caption_vi", "")),
        str(raw.get("ocr_text", "")),
        str(raw.get("news_ticker_text", "")),
    ]
    for detection in raw.get("detections", []):
        if isinstance(detection, dict):
            parts.extend(
                str(detection.get(key, ""))
                for key in ("label", "description", "description_vi", "action")
            )
            attributes = detection.get("attributes", [])
            if isinstance(attributes, list):
                parts.extend(str(item) for item in attributes)
    return " ".join(parts)


def _scene_profile(scene: MicroScene, frame_by_n: dict[int, RuntimeFrame]) -> set[str]:
    return set().union(*(_words(_frame_text(frame_by_n[n])) for n in scene.frame_indices))


def _fallback_candidates(
    scenes: list[MicroScene],
    frames: list[RuntimeFrame],
    segments: dict[int, ASRSegment],
) -> list[StoryCandidate]:
    """Create useful, evidence-linked candidates when Gemini is disabled/unavailable."""

    frame_by_n = {frame.keyframe_n: frame for frame in frames}
    if not scenes:
        return []
    candidates: list[StoryCandidate] = []
    current_scenes: list[MicroScene] = [scenes[0]]
    current_profile = _scene_profile(scenes[0], frame_by_n)
    current_end = scenes[0].end_ms

    def close() -> None:
        nonlocal current_scenes, current_profile, current_end
        scene_ids = [scene.scene_id for scene in current_scenes]
        frame_numbers = [n for scene in current_scenes for n in scene.frame_indices]
        asr_indices = sorted({i for scene in current_scenes for i in scene.asr_segment_indices})
        text = " ".join(
            _frame_text(frame_by_n[n]) for n in frame_numbers[: min(len(frame_numbers), 8)]
        )
        speech = " ".join(segments[index].text for index in asr_indices)
        words = Counter(_words(f"{text} {speech}"))
        topics = [item for item, _ in words.most_common(8)]
        title = f"News story {len(candidates) + 1}"
        candidates.append(
            StoryCandidate(
                title=title,
                summary=" ".join((speech or text).split())[:600],
                topics=topics,
                entities=[],
                locations=[],
                scene_ids=scene_ids,
                asr_segment_indices=asr_indices,
                uncertain=True,
            )
        )
        current_scenes = []
        current_profile = set()
        current_end = 0

    for scene in scenes[1:]:
        profile = _scene_profile(scene, frame_by_n)
        speech = " ".join(segments[index].text for index in scene.asr_segment_indices)
        overlap = _jaccard(current_profile, profile)
        hard_transition = bool(TRANSITION_RE.search(speech))
        too_long = scene.end_ms - current_scenes[0].start_ms > 120_000
        if hard_transition or (overlap < 0.08 and too_long) or scene.start_ms - current_end > 12_000:
            close()
            current_scenes = [scene]
            current_profile = profile
            current_end = scene.end_ms
        else:
            current_scenes.append(scene)
            current_profile |= profile
            current_end = scene.end_ms
    close()
    return candidates


class NewsSummarizer:
    """Use Gemini when explicitly enabled; otherwise remain deterministic/offline."""

    def __init__(self, *, use_llm: bool, require_llm: bool = False) -> None:
        self.mode = "deterministic_fallback"
        self.model = "none"
        self.prompt_version = "news-summary-v1-fallback"
        self.client = None
        self._use_llm = use_llm
        self._require_llm = require_llm
        if not use_llm:
            return
        if genai is None or types is None:
            if require_llm:
                raise RuntimeError("google-genai is not installed but --require-llm was requested")
            return
        from semantic_pipeline.core.environment import get_env_value

        api_key = get_env_value("GEMINI_API_KEY")
        if not api_key:
            if require_llm:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            return
        self.client = genai.Client(api_key=api_key)
        self.model = get_env_value("GEMINI_VIDEO_SUMMARY_MODEL") or get_env_value(
            "GEMINI_VISUAL_MODEL"
        ) or "gemini-2.5-flash"
        self.mode = "gemini"
        self.prompt_version = "news-summary-v1"

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        if self.client is None or types is None:
            raise RuntimeError("LLM client is not available")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini summary response was empty")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Gemini summary response must be an object")
        return payload

    def summarize_windows(
        self,
        windows: list[dict[str, Any]],
        scenes: list[MicroScene],
        frames: list[RuntimeFrame],
        segments: dict[int, ASRSegment],
        *,
        require_llm: bool = False,
    ) -> list[StoryCandidate]:
        if self.client is None:
            return _fallback_candidates(scenes, frames, segments)
        scene_ids = {scene.scene_id for scene in scenes}
        candidates: list[StoryCandidate] = []
        for window in windows:
            prompt = self._window_prompt(window)
            try:
                payload = self._generate_json(prompt)
            except Exception:
                if require_llm:
                    raise
                self.mode = "deterministic_fallback_after_llm_error"
                continue
            raw_events = payload.get("events", [])
            if not isinstance(raw_events, list):
                continue
            allowed_scenes = set(window.get("scene_ids", [])) & scene_ids
            allowed_asr = {item["index"] for item in window.get("asr_segments", [])}
            for raw in raw_events:
                if not isinstance(raw, dict):
                    continue
                refs = [str(item) for item in raw.get("scene_refs", []) if str(item) in allowed_scenes]
                asr_refs: list[int] = []
                for item in raw.get("asr_segment_refs", []):
                    try:
                        parsed = int(item)
                    except (TypeError, ValueError):
                        continue
                    if parsed in allowed_asr:
                        asr_refs.append(parsed)
                if not refs and not asr_refs:
                    continue
                candidates.append(
                    StoryCandidate(
                        title=str(raw.get("title", "")).strip() or "Unlabelled news event",
                        summary=str(raw.get("summary", "")).strip(),
                        topics=_clean_strings(raw.get("topics")),
                        entities=_clean_strings(raw.get("entities")),
                        locations=_clean_strings(raw.get("locations")),
                        scene_ids=refs,
                        asr_segment_indices=asr_refs,
                        source_window_id=window.get("window_id"),
                        uncertain=bool(raw.get("uncertain", False)),
                    )
                )
        if not candidates:
            if self.client is not None and not require_llm:
                self.mode = "deterministic_fallback_after_llm_error"
            return _fallback_candidates(scenes, frames, segments)
        return candidates

    @staticmethod
    def _window_prompt(window: dict[str, Any]) -> str:
        return (
            "You are extracting evidence-grounded news events from a video window. "
            "Use only the supplied caption, objects, OCR and ASR. Do not invent facts, "
            "timestamps, or references. Return JSON with an events array; each event has "
            "title, summary, topics, entities, locations, scene_refs, asr_segment_refs, "
            "uncertain. Keep anchor and B-roll together when they discuss the same story. "
            "Split when the ASR explicitly changes topic.\n\n"
            + json.dumps(window, ensure_ascii=False)
        )

    def summarize_video(
        self,
        candidates: list[StoryCandidate],
        *,
        video_id: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        if self.client is not None and candidates:
            try:
                payload = self._generate_json(
                    "Create a concise bilingual video summary from these ordered, evidence-grounded "
                    "news stories. Return JSON with summary_vi, summary_en, main_topics, "
                    "main_entities, main_locations. Do not add facts.\n\n"
                    + json.dumps([candidate.__dict__ for candidate in candidates], ensure_ascii=False)
                )
                return {
                    "video_id": video_id,
                    "duration_ms": duration_ms,
                    "summary_vi": str(payload.get("summary_vi", "")).strip(),
                    "summary_en": str(payload.get("summary_en", "")).strip(),
                    "main_topics": _clean_strings(payload.get("main_topics")),
                    "main_entities": _clean_strings(payload.get("main_entities")),
                    "main_locations": _clean_strings(payload.get("main_locations")),
                }
            except Exception:
                if self._require_llm:
                    raise
                if self._use_llm:
                    # A transient provider error should still leave a usable,
                    # evidence-linked fallback unless --require-llm is active.
                    self.mode = "deterministic_fallback_after_llm_error"
        topics = _unique(item for candidate in candidates for item in candidate.topics)
        entities = _unique(item for candidate in candidates for item in candidate.entities)
        locations = _unique(item for candidate in candidates for item in candidate.locations)
        summary = " ".join(candidate.summary for candidate in candidates[:12]).strip()
        return {
            "video_id": video_id,
            "duration_ms": duration_ms,
            "summary_vi": summary[:2000],
            "summary_en": summary[:2000],
            "main_topics": topics[:20],
            "main_entities": entities[:20],
            "main_locations": locations[:20],
        }


def _clean_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique(str(item).strip() for item in value if str(item).strip())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.casefold()
        if normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return result
