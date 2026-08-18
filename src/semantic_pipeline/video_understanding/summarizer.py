"""Domain-agnostic, evidence-grounded video summarization."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any, Iterable

from semantic_pipeline.core.environment import get_env_value

from .models import ASRSegment, MicroScene, RuntimeFrame, StoryCandidate
from .llm_schemas import VIDEO_SUMMARY_SCHEMA, WINDOW_SUMMARY_SCHEMA

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
PROGRAM_INTRO_RE = re.compile(
    r"\b(chương trình tin tức|bản tin .*?giới thiệu các nội dung|tin tức 60 giây)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def load_video_summary_model() -> str:
    """Load the summary model from repository environment configuration only."""

    model_name = get_env_value("GEMINI_VIDEO_SUMMARY_MODEL") or get_env_value(
        "GEMINI_VISUAL_MODEL"
    )
    if not model_name:
        raise RuntimeError(
            "GEMINI_VIDEO_SUMMARY_MODEL or GEMINI_VISUAL_MODEL must be set in .env"
        )
    return model_name


def _terms(text: str) -> list[str]:
    return [
        item.casefold()
        for item in WORD_RE.findall(text or "")
        if len(item) > 2 and item.casefold() not in STOPWORDS
    ]


def _words(text: str) -> set[str]:
    return set(_terms(text))


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
    parts.extend(_spatial_phrases(raw))
    return " ".join(parts)


def _spatial_phrases(raw: dict[str, Any]) -> list[str]:
    """Resolve object-id relations into compact, searchable phrases."""

    labels = {
        str(item.get("object_id", "")): str(item.get("label", ""))
        for item in raw.get("detections", [])
        if isinstance(item, dict) and item.get("object_id")
    }
    phrases: list[str] = []
    for relation in raw.get("spatial_relations", []):
        if not isinstance(relation, dict):
            continue
        subject_id = str(relation.get("subject_id", "")).strip()
        predicate = str(relation.get("predicate", "")).strip().replace("_", " ")
        object_id = str(relation.get("object_id", "")).strip()
        if not subject_id or not predicate or not object_id:
            continue
        phrases.append(
            f"{labels.get(subject_id, subject_id)} {predicate} {labels.get(object_id, object_id)}"
        )
    return phrases


def _unique_text(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = " ".join(str(value).split())
        normalized = compact.casefold()
        if compact and normalized not in seen:
            seen.add(normalized)
            result.append(compact)
    return result


def _observable_facets(frames: Iterable[RuntimeFrame]) -> tuple[list[str], list[str], list[str]]:
    actions: list[str] = []
    objects: list[str] = []
    visual_states: list[str] = []
    for frame in frames:
        raw = frame.raw_metadata
        for detection in raw.get("detections", []):
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip()
            action = str(detection.get("action", "")).strip()
            if label:
                objects.append(label)
            if action:
                actions.append(f"{label}: {action}" if label else action)
        visual_states.extend(_spatial_phrases(raw))
    return (
        _unique_text(actions),
        _unique_text(objects),
        _unique_text(visual_states),
    )


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
        selected_frames = [frame_by_n[n] for n in frame_numbers if n in frame_by_n]
        descriptions = _unique_text(
            str(frame.raw_metadata.get("detailed_caption_vi", ""))
            or str(frame.raw_metadata.get("caption_vi", ""))
            or str(frame.raw_metadata.get("detailed_caption", ""))
            or str(frame.raw_metadata.get("caption", ""))
            for frame in selected_frames
        )
        text = " ".join(_frame_text(frame) for frame in selected_frames)
        speech = " ".join(segments[index].text for index in asr_indices)
        words = Counter(_terms(f"{text} {speech}"))
        topics = [item for item, _ in words.most_common(8)]
        actions, objects, visual_states = _observable_facets(selected_frames)
        title = next(
            (description[:160] for description in descriptions if description),
            f"Video segment {len(candidates) + 1}",
        )
        summary_parts = _unique_text([speech, *descriptions[:8]])
        candidates.append(
            StoryCandidate(
                title=title,
                summary=" ".join(summary_parts)[:1200],
                topics=topics,
                entities=[],
                locations=[],
                scene_ids=scene_ids,
                asr_segment_indices=asr_indices,
                uncertain=True,
                actions=actions[:20],
                objects=objects[:30],
                visual_states=visual_states[:30],
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


class VideoSummarizer:
    """Use Gemini when enabled; otherwise build a metadata-only summary."""

    def __init__(
        self,
        *,
        use_llm: bool,
        require_llm: bool = False,
        domain_hint: str = "general visual content",
        use_window_llm: bool = True,
    ) -> None:
        self.mode = "deterministic_fallback"
        self.model = "none"
        self.prompt_version = "general-video-summary-v3-fallback"
        self.domain_hint = " ".join(domain_hint.split()) or "general visual content"
        self.client = None
        self._last_request_at = 0.0
        self._request_interval_seconds = 4.2
        self._use_llm = use_llm
        self._require_llm = require_llm
        self._use_window_llm = use_window_llm
        self.request_count = 0
        if not use_llm:
            return
        if genai is None or types is None:
            if require_llm:
                raise RuntimeError("google-genai is not installed but --require-llm was requested")
            return
        try:
            self._request_interval_seconds = max(
                0.0,
                float(get_env_value("GEMINI_REQUEST_INTERVAL_SECONDS") or "4.2"),
            )
        except ValueError:
            self._request_interval_seconds = 4.2

        api_key = get_env_value("GEMINI_API_KEY")
        if not api_key:
            if require_llm:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            return
        self.client = genai.Client(api_key=api_key)
        self.model = load_video_summary_model()
        self.mode = "gemini" if use_window_llm else "gemini_final_only"
        self.prompt_version = (
            "general-video-summary-v3"
            if use_window_llm
            else "general-video-summary-v3-final-only"
        )

    def _generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        response_kind: str,
    ) -> dict[str, Any]:
        if self.client is None or types is None:
            raise RuntimeError("LLM client is not available")
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._request_interval_seconds:
            time.sleep(self._request_interval_seconds - elapsed)
        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        response = None
        for attempt, delay in enumerate((0.0, 5.0, 15.0, 30.0, 60.0)):
            if delay:
                time.sleep(delay)
            self._last_request_at = time.monotonic()
            try:
                self.request_count += 1
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as exc:
                message = str(exc)
                quota_error = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not quota_error or attempt == 4:
                    raise
        if response is None:  # pragma: no cover - defensive for unusual clients.
            raise RuntimeError(f"Gemini {response_kind} request returned no response")
        text = getattr(response, "text", None)
        if not text:
            raise ValueError(f"Gemini {response_kind} response was empty")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini {response_kind} response was not valid JSON") from exc
        return _normalize_payload(payload, response_kind)

    def summarize_windows(
        self,
        windows: list[dict[str, Any]],
        scenes: list[MicroScene],
        frames: list[RuntimeFrame],
        segments: dict[int, ASRSegment],
        *,
        require_llm: bool = False,
    ) -> list[StoryCandidate]:
        if self.client is None or not self._use_window_llm:
            return _fallback_candidates(scenes, frames, segments)
        scene_ids = {scene.scene_id for scene in scenes}
        candidates: list[StoryCandidate] = []
        for window in windows:
            prompt = self._window_prompt(window, domain_hint=self.domain_hint)
            try:
                payload = self._generate_json(
                    prompt,
                    schema=WINDOW_SUMMARY_SCHEMA,
                    response_kind="window",
                )
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
                title = str(raw.get("title", "")).strip()
                raw_summary = str(raw.get("summary", "")).strip()
                if _is_program_intro(title, raw_summary):
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
                        title=title or "Unlabelled semantic episode",
                        summary=raw_summary,
                        topics=_clean_strings(raw.get("topics")),
                        entities=_clean_strings(raw.get("entities")),
                        locations=_clean_strings(raw.get("locations")),
                        scene_ids=refs,
                        asr_segment_indices=asr_refs,
                        source_window_id=window.get("window_id"),
                        uncertain=bool(raw.get("uncertain", False)),
                        actions=_clean_strings(raw.get("actions")),
                        objects=_clean_strings(raw.get("objects")),
                        visual_states=_clean_strings(raw.get("visual_states")),
                    )
                )
        if not candidates:
            if self.client is not None and not require_llm:
                self.mode = "deterministic_fallback_after_llm_error"
            return _fallback_candidates(scenes, frames, segments)
        return candidates

    @staticmethod
    def _window_prompt(
        window: dict[str, Any],
        *,
        domain_hint: str = "general visual content",
    ) -> str:
        return (
            "You are extracting ordered, evidence-grounded semantic episodes from a video window. "
            f"The weak domain hint is {domain_hint!r}; visible metadata is authoritative. "
            "Use only the supplied frame captions, detections, object actions, spatial relations, "
            "OCR and ASR. Do not invent identities, motion, causality, timestamps, or references. "
            "A semantic episode is a coherent activity, situation, demonstration step, discussion, "
            "performance passage or scene—not an individual static frame. Use visual evidence even "
            "when ASR is absent. Merge adjacent frames that show the same episode; split on a real "
            "change of activity, subject, setting or topic. Describe observable actions and state/"
            "relation changes conservatively; do not claim an exact action boundary from sparse frames. "
            "Write every title and summary in natural Vietnamese. Return a JSON object with an events "
            "array; each event has title, summary, topics, entities, locations, actions, objects, "
            "visual_states, scene_refs, asr_segment_refs and uncertain. visual_states contains concise "
            "observable relations or conditions, not speculation.\n\n"
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
                    "Create a concise bilingual retrieval-oriented video summary from these ordered, "
                    "evidence-grounded semantic episodes. Cover the video's identity-defining subjects, "
                    "setting, objects, activities and coarse chronological progression. Do not turn "
                    "single-frame observations into unsupported temporal claims and do not add facts. "
                    "Return summary_vi, summary_en, main_topics, main_entities, main_locations, "
                    "main_actions, main_objects, main_visual_states and chronological_outline. "
                    "summary_vi must be natural Vietnamese; summary_en must be natural English and "
                    "must not copy summary_vi. chronological_outline must be short Vietnamese phrases "
                    "in observed order.\n\n"
                    + json.dumps([candidate.__dict__ for candidate in candidates], ensure_ascii=False),
                    schema=VIDEO_SUMMARY_SCHEMA,
                    response_kind="video summary",
                )
                summary_vi = str(payload.get("summary_vi", "")).strip()
                summary_en = str(payload.get("summary_en", "")).strip()
                if self._require_llm and summary_vi.casefold() == summary_en.casefold():
                    raise ValueError("Gemini bilingual summaries must not be identical")
                return {
                    "video_id": video_id,
                    "duration_ms": duration_ms,
                    "summary_vi": summary_vi,
                    "summary_en": summary_en,
                    "main_topics": _clean_strings(payload.get("main_topics")),
                    "main_entities": _clean_strings(payload.get("main_entities")),
                    "main_locations": _clean_strings(payload.get("main_locations")),
                    "main_actions": _clean_strings(payload.get("main_actions")),
                    "main_objects": _clean_strings(payload.get("main_objects")),
                    "main_visual_states": _clean_strings(payload.get("main_visual_states")),
                    "chronological_outline": _clean_strings(payload.get("chronological_outline")),
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
        actions = _unique(item for candidate in candidates for item in candidate.actions)
        objects = _unique(item for candidate in candidates for item in candidate.objects)
        visual_states = _unique(item for candidate in candidates for item in candidate.visual_states)
        summary = " ".join(candidate.summary for candidate in candidates[:12]).strip()
        return {
            "video_id": video_id,
            "duration_ms": duration_ms,
            "summary_vi": summary[:2000],
            "summary_en": summary[:2000],
            "main_topics": topics[:20],
            "main_entities": entities[:20],
            "main_locations": locations[:20],
            "main_actions": actions[:30],
            "main_objects": objects[:40],
            "main_visual_states": visual_states[:30],
            "chronological_outline": [
                candidate.title for candidate in candidates[:20] if candidate.title
            ],
        }


def _clean_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique(str(item).strip() for item in value if str(item).strip())


def _is_program_intro(title: str, summary: str) -> bool:
    """Exclude a broadcast opener that describes the show rather than a story."""

    combined = f"{title} {summary}"
    return bool(PROGRAM_INTRO_RE.search(combined)) and not any(
        marker in combined.casefold()
        for marker in ("khởi công", "tai nạn", "cháy", "lũ", "cứu hộ", "đấu giá")
    )


def _normalize_payload(payload: Any, response_kind: str) -> dict[str, Any]:
    """Normalize provider JSON while preserving the expected top-level contract."""

    if response_kind == "window" and isinstance(payload, list):
        # Compatibility with providers/models that honor JSON but ignore the
        # requested top-level object shape.
        payload = {"events": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini {response_kind} response must be an object")
    return payload


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.casefold()
        if normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return result


# Compatibility for existing imports while the rest of the pipeline migrates
# away from the original news-only naming.
NewsSummarizer = VideoSummarizer
