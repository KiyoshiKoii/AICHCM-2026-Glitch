"""Batch visual metadata extraction with Gemini.

One request processes a labelled group of keyframes and returns caption, OCR,
and object boxes.  The persisted output is deliberately compact; video/frame
coordinates are derived from ``frame_id`` by consumers that need them.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from ..core.compact_metadata import (
    CompactDetection,
    CompactSpatialRelation,
    CompactVisualRecord,
)
from ..core.environment import get_env_value
from ..core.frame_deduplication import (
    DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
    GlobalFrameDeduplication,
    load_global_frame_deduplication,
)
from ..core.frame_id import frame_id_from_path, parse_frame_id
from ..core.json_io import write_json_atomically
from ..core.request_limits import is_rate_limit_error, is_transient_service_error
from ..core.spatial_reasoning import (
    RawDetection,
    infer_spatial_relations,
    normalise_detections,
)
from ..core.visual_profiles import DEFAULT_YOUTUBE_METADATA_PATH, VisualContextResolver


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_SIZE = 100
DEFAULT_MAX_INLINE_BYTES = 14 * 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_REQUESTS_PER_MINUTE = 15
DEFAULT_DAILY_REQUEST_LIMIT = 500
MAX_TRANSIENT_RETRIES = 5
MAX_RATE_LIMIT_RETRIES = 5
MAX_ENRICHED_DETECTIONS = 5
MAX_DETECTIONS = 5
FORBIDDEN_VISUAL_TERMS = (
    "middle-aged",
    "elderly",
    "young man",
    "young woman",
    "old man",
    "old woman",
    "asian",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EDITORIAL_OVERLAY_LABELS = frozenset(
    {
        "logo",
        "watermark",
        "channel logo",
        "program logo",
        "lower third",
        "lower-third",
        "text overlay",
        "graphic overlay",
    }
)


class GeminiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeminiDetection(GeminiModel):
    label: str
    description: str = ""
    description_vi: str = ""
    attributes: list[str] = Field(default_factory=list)
    action: str = ""
    # Gemini's documented order: [ymin, xmin, ymax, xmax], in [0, 1000].
    box_2d: list[float] = Field(min_length=4, max_length=4)

    @field_validator("label")
    @classmethod
    def require_label(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if not value:
            raise ValueError("detection label must not be blank")
        return value

    @field_validator("description", "description_vi", "action", mode="before")
    @classmethod
    def normalise_optional_text(cls, value: Any) -> str:
        return value if isinstance(value, str) else ""

    @field_validator("attributes", mode="before")
    @classmethod
    def normalise_attributes(cls, value: Any) -> list[str]:
        return value if isinstance(value, list) else []


class GeminiFrameResult(GeminiModel):
    # ``slot`` is the authoritative request-local identity. A zero default
    # preserves compatibility with artifacts/tests produced before slots.
    slot: int = 0
    frame_id: str
    caption: str
    detailed_caption: str
    caption_vi: str = ""
    detailed_caption_vi: str = ""
    ocr_text: str = ""
    news_ticker_text: str = ""
    detections: list[GeminiDetection] = Field(default_factory=list)

    @field_validator("frame_id")
    @classmethod
    def require_frame_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("frame_id must not be blank")
        return value

    @field_validator("caption", "detailed_caption")
    @classmethod
    def require_caption(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if not value:
            raise ValueError("caption must not be blank")
        return value


class GeminiVisualBatch(GeminiModel):
    frames: list[GeminiFrameResult]


class GeminiOCRFrameResult(GeminiModel):
    slot: int = 0
    frame_id: str
    ocr_text: str = ""
    news_ticker_text: str = ""

    @field_validator("frame_id")
    @classmethod
    def require_frame_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("frame_id must not be blank")
        return value

    @field_validator("ocr_text", "news_ticker_text", mode="before")
    @classmethod
    def normalise_text(cls, value: Any) -> str:
        return value if isinstance(value, str) else ""


class GeminiOCRBatch(GeminiModel):
    frames: list[GeminiOCRFrameResult]


# Gemini's response-schema endpoint accepts an OpenAPI-schema subset.  Passing
# the Pydantic model itself serialises ``additionalProperties`` under this SDK,
# which Gemini 3.1 Flash Lite rejects.  Keep the transport schema deliberately
# simple and let the Pydantic models above perform strict local validation.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "frames": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "slot": {"type": "INTEGER"},
                    "frame_id": {"type": "STRING"},
                    "caption": {"type": "STRING"},
                    "detailed_caption": {"type": "STRING"},
                    "caption_vi": {"type": "STRING"},
                    "detailed_caption_vi": {"type": "STRING"},
                    "ocr_text": {"type": "STRING"},
                    "news_ticker_text": {"type": "STRING"},
                    "detections": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "label": {"type": "STRING"},
                                "description": {"type": "STRING"},
                                "description_vi": {"type": "STRING"},
                                "attributes": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"},
                                },
                                "action": {"type": "STRING"},
                                "box_2d": {
                                    "type": "ARRAY",
                                    "items": {"type": "NUMBER"},
                                },
                            },
                            "required": ["label", "box_2d"],
                        },
                    },
                },
                "required": [
                    "slot",
                    "frame_id",
                    "caption",
                    "detailed_caption",
                    "caption_vi",
                    "detailed_caption_vi",
                    "ocr_text",
                    "news_ticker_text",
                    "detections",
                ],
            },
        },
    },
    "required": ["frames"],
}


GEMINI_OCR_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "frames": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "slot": {"type": "INTEGER"},
                    "frame_id": {"type": "STRING"},
                    "ocr_text": {"type": "STRING"},
                    "news_ticker_text": {"type": "STRING"},
                },
                "required": ["slot", "frame_id", "ocr_text", "news_ticker_text"],
            },
        },
    },
    "required": ["frames"],
}


def sanitise_gemini_payload(response_text: str) -> tuple[Any, list[str]]:
    """Remove malformed detections without discarding valid frame metadata.

    Gemini occasionally emits a truncated ``box_2d`` such as ``[0, 0, 1000]``.
    A strict Pydantic parse would reject the entire response in that case,
    even though the frame caption and the other detections are usable.  There
    is no reliable geometric repair for a missing coordinate, so the safest
    policy is to drop only that detection and retain the frame.
    """
    payload = json.loads(response_text)
    if not isinstance(payload, dict):
        return payload, []

    malformed: list[str] = []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return payload, malformed
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        detections = frame.get("detections")
        if not isinstance(detections, list):
            frame["detections"] = []
            continue
        valid_detections: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            box = detection.get("box_2d") if isinstance(detection, dict) else None
            if not isinstance(box, list) or len(box) != 4:
                label = detection.get("label", "?") if isinstance(detection, dict) else "?"
                malformed.append(
                    f"frame_index={frame_index} detection_index={detection_index} "
                    f"label={label!r} box_2d={box!r}"
                )
                continue
            valid_detections.append(detection)
        frame["detections"] = valid_detections
    return payload, malformed


BASE_VISUAL_PROMPT = """You receive labelled keyframe images. Process every image independently.

For each supplied image return exactly one item. `slot` is the authoritative identity: copy the
integer Input Slot exactly. Return the matching frame_id too, but never use it to infer which
image a caption belongs to.
- caption: one short, retrieval-focused English sentence. Prioritize the number of clearly
  separable dominant foreground subjects, their left-to-right arrangement, distinctive visible
  attributes, clothing, and actions, followed by a short setting description. Avoid vague phrases
  such as "a group of people" when the foreground subjects can be counted and described. Omit
  ordinary lighting, vague atmosphere, and other non-discriminative details unless they are
  visually unusual or central to the scene. Do not add special tokens.
- detailed_caption: two to four factual English sentences. First state the shot type and number of
  dominant foreground subjects when clear. Then describe those subjects from left to right using
  visible attributes, clothing, pose, and actions. Describe secondary people and background context
  separately. Do not mention broadcaster graphics, watermarks, clocks, lower-thirds, or ticker text
  in captions; put visible text in the OCR fields. For scenes without dominant people, apply the same foreground-to-background ordering
  to important visible structures and objects. State only what is visible; do not infer names,
  roles, relationships, causes, or events.
- caption_vi: one short, retrieval-focused Vietnamese sentence generated directly from visual
  evidence in the image. Do not translate or paraphrase `caption`. Independently select the most
  discriminative visible details and write natural Vietnamese. Apply the same subject count,
  left-to-right arrangement, attributes, clothing, action, and concise setting priorities.
- detailed_caption_vi: two to four natural Vietnamese sentences generated directly from the image,
  not translated or paraphrased from `detailed_caption`. Independently describe the shot, dominant
  foreground subjects from left to right, and secondary background context. Use only visible
  evidence and follow the same no-inference restrictions as the English fields.
- ocr_text: transcribe only clearly visible text in its original language. Do not translate,
  infer missing characters, or include a reading of the image label. Return an empty string when
  no text is legible.
- news_ticker_text: return an empty string unless a later video-specific instruction asks for it.
- detections: return up to five of the most salient visible people, objects, or physical
  structures, ordered most salient first. If a real scene contains any clearly visible physical
  subject, return at least one useful box; do not use an empty array merely because the subject
  is large, diffuse, partly occluded, or not a conventional countable object. In addition to
  people and vehicles, box useful regions/structures such as damaged pavement, rubble, pipes,
  walls, roads, bridges, shorelines, buildings, boats, rivers, documents, or blueprints. Use a
  short lowercase English noun phrase for label. For detections only, exclude broadcaster logos,
  watermarks, lower-thirds, and other editorial overlays. Return an empty detections array only
  for a genuinely empty/blurred frame or a pure intro, transition, title-card, or abstract
  program graphic with no useful physical subject. Return box_2d as exactly four values:
  [ymin, xmin, ymax, xmax], normalized to integers or decimals from 0 to 1000. Never return a
  three-value or otherwise truncated box.
  Decide which detected objects have visually distinctive details useful for retrieval. For no
  more than five such objects in an image, additionally return:
  - description: one compact English noun phrase tying visible appearance, clothing, color, or
    other distinctive detail to this exact box.
  - description_vi: a natural Vietnamese description generated directly from the pixels, not
    translated from description, and tied to the same object.
  - attributes: at most six short lowercase English visual attributes suitable for later filtering.
  - action: one short lowercase English phrase for a directly visible action, or an empty string.
  If caption or detailed_caption explicitly mentions a distinctive attribute, clothing detail,
  color, or action for a detected object, copy that fact into the matching detection fields;
  do not leave the matching object enrichment empty.
  For an ordinary, background, partly hidden, or non-distinctive object, omit these fields or
  return empty values. Never infer identity, occupation, age, ethnicity, relationships, intent,
  or an action that is not visually evident. Do not describe apparent age or ethnicity; use a
  neutral label such as "person" when those properties are not directly evidenced. Keep returning useful boxes even when their semantic
  detail fields are empty.
  Do not output a confidence score. Do not output a box for the whole image unless the whole
  image is itself a physical object such as a document or screen.

Return JSON only and never combine evidence between images."""


NEWS_TICKER_INSTRUCTION = """For this news broadcast, `news_ticker_text` is required:
- Transcribe the small scrolling news ticker/crawl along the bottom edge of the frame.
- This field has priority even when the ticker is small. Preserve the visible Vietnamese or other
  source-language text exactly; transcribe only the currently visible segment and never infer the
  off-screen continuation.
- Do not put the channel watermark, clock, or program logo in this field.
- Keep `ocr_text` for other visible scene text; do not duplicate the ticker there.
- Return an empty string only when no ticker segment is visible or legible."""


OCR_ONLY_PROMPT = """OCR-only extraction: you receive labelled keyframe images. Process every image independently.

Return exactly one item for every supplied image. `slot` is the authoritative identity: copy the
integer Input Slot exactly and return the matching frame_id. Never combine evidence between images.

- `ocr_text`: transcribe only clearly legible scene text, signs, documents, screens, or charts in
  the original language. Do not translate, infer missing characters, or include the image label,
  broadcaster watermark, clock, logo, lower-third, or scrolling ticker. Return an empty string when
  no scene text is legible.
- `news_ticker_text`: only the currently visible scrolling ticker segment along the bottom edge of
  the frame. Preserve the source-language text exactly and never infer off-screen continuation. Do
  not include a watermark, clock, logo, or lower-third. Return an empty string when no ticker is
  visible or legible.

Return JSON only. Do not return captions, detections, explanations, or confidence scores."""


def build_visual_prompt(context=None) -> str:
    """Compose the shared extraction contract with an optional video profile."""
    if context is None:
        return BASE_VISUAL_PROMPT
    ticker_instruction = (
        f"\n\n{NEWS_TICKER_INSTRUCTION}"
        if context.profile.collection_code in {"L21", "L22"}
        else ""
    )
    return f"{BASE_VISUAL_PROMPT}{ticker_instruction}\n\n{context.prompt_context()}"


def build_ocr_prompt(context=None) -> str:
    """Compose the small OCR-only contract used for deduplicated frames."""

    if context is None:
        return OCR_ONLY_PROMPT
    ticker_instruction = (
        f"\n\n{NEWS_TICKER_INSTRUCTION}"
        if context.profile.collection_code in {"L21", "L22"}
        else ""
    )
    return f"{OCR_ONLY_PROMPT}{ticker_instruction}"


# Kept as a named constant for callers/tests that need the generic prompt.
VISUAL_PROMPT = BASE_VISUAL_PROMPT


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _strip_forbidden_visual_terms(value: str) -> tuple[str, set[str]]:
    """Remove demographic guesses that are prohibited by the visual contract."""

    text = value if isinstance(value, str) else ""
    flags: set[str] = set()
    terms = (*FORBIDDEN_VISUAL_TERMS, "trung niên", "cao tuổi", "người trẻ", "châu á")
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE):
            text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", "", text, flags=re.IGNORECASE)
            flags.add("removed_inferred_demographic")
    return _clean_text(text), flags


def load_gemini_api_key() -> str | None:
    """Read the configured key without requiring callers to export it manually."""
    return get_env_value("GEMINI_API_KEY")


def load_gemini_visual_model() -> str:
    """Load the extraction/repair model selected in `.env`."""
    return get_env_value("GEMINI_VISUAL_MODEL") or DEFAULT_MODEL


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _raw_detection_from_gemini(
    detection: GeminiDetection,
    width: int,
    height: int,
) -> RawDetection | None:
    ymin, xmin, ymax, xmax = detection.box_2d
    values = [ymin, xmin, ymax, xmax]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(value))) for value in values)
    if xmin >= xmax or ymin >= ymax:
        return None
    return RawDetection(
        label=detection.label,
        bbox=(xmin * width / 1000.0, ymin * height / 1000.0, xmax * width / 1000.0, ymax * height / 1000.0),
    )


def _is_editorial_overlay_label(label: str) -> bool:
    return " ".join(label.casefold().split()) in EDITORIAL_OVERLAY_LABELS


def _detection_key(
    label: str,
    bbox: tuple[float, float, float, float],
) -> tuple[str, tuple[float, float, float, float]]:
    return (
        " ".join(label.strip().casefold().split()),
        tuple(round(value, 6) for value in bbox),
    )


def _clean_attributes(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = _clean_text(value).casefold()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) == 6:
            break
    return result


def compact_record_from_gemini(
    result: GeminiFrameResult,
    image_path: Path,
    with_spatial: bool = True,
) -> CompactVisualRecord:
    """Validate Gemini boxes and map one response into the compact on-disk schema."""
    expected_frame_id = frame_id_from_path(image_path)
    if result.frame_id != expected_frame_id:
        raise ValueError(
            f"Gemini returned frame_id {result.frame_id!r}; expected {expected_frame_id!r}"
        )
    width, height = _image_size(image_path)
    raw_detections: list[RawDetection] = []
    quality_flags: set[str] = set()
    enriched_by_detection: dict[
        tuple[str, tuple[float, float, float, float]], GeminiDetection
    ] = {}
    enriched_count = 0
    for detection in result.detections:
        if _is_editorial_overlay_label(detection.label):
            continue
        raw = _raw_detection_from_gemini(detection, width, height)
        if raw is None:
            continue
        raw_detections.append(raw)
        has_semantic_detail = bool(
            _clean_text(detection.description)
            or _clean_text(detection.description_vi)
            or _clean_attributes(detection.attributes)
            or _clean_text(detection.action)
        )
        if has_semantic_detail and enriched_count < MAX_ENRICHED_DETECTIONS:
            normalized_bbox = (
                raw.bbox[0] / width,
                raw.bbox[1] / height,
                raw.bbox[2] / width,
                raw.bbox[3] / height,
            )
            enriched_by_detection.setdefault(
                _detection_key(raw.label, normalized_bbox), detection
            )
            enriched_count += 1
    detections = normalise_detections(
        raw_detections,
        width,
        height,
        max_detections=MAX_DETECTIONS,
    )
    relations = infer_spatial_relations(detections) if with_spatial else []
    caption, caption_flags = _strip_forbidden_visual_terms(result.caption)
    detailed_caption, detailed_flags = _strip_forbidden_visual_terms(result.detailed_caption)
    caption_vi, caption_vi_flags = _strip_forbidden_visual_terms(result.caption_vi)
    detailed_caption_vi, detailed_vi_flags = _strip_forbidden_visual_terms(
        result.detailed_caption_vi
    )
    caption = caption or "A visible scene is shown."
    detailed_caption = detailed_caption or "A visible scene is shown."
    quality_flags.update(caption_flags | detailed_flags | caption_vi_flags | detailed_vi_flags)

    def clean_detection_text(value: str) -> str:
        cleaned, flags = _strip_forbidden_visual_terms(value)
        quality_flags.update(flags)
        return cleaned

    def clean_detection_attributes(values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = clean_detection_text(value).casefold()
            if item and item not in cleaned:
                cleaned.append(item)
            if len(cleaned) == 6:
                break
        return cleaned

    return CompactVisualRecord(
        frame_id=expected_frame_id,
        visual_source_frame_id=expected_frame_id,
        ocr_source_frame_id=expected_frame_id,
        quality_flags=sorted(quality_flags),
        caption=caption,
        detailed_caption=detailed_caption,
        caption_vi=caption_vi,
        detailed_caption_vi=detailed_caption_vi,
        ocr_text=_clean_text(result.ocr_text),
        news_ticker_text=_clean_text(result.news_ticker_text),
        detections=[
            CompactDetection(
                object_id=item.object_id,
                label=item.label,
                bbox=item.bbox,
                description=(
                    clean_detection_text(detail.description) if detail is not None else ""
                ),
                description_vi=(
                    clean_detection_text(detail.description_vi) if detail is not None else ""
                ),
                attributes=(
                    clean_detection_attributes(detail.attributes) if detail is not None else []
                ),
                action=(
                    clean_detection_text(detail.action).casefold() if detail is not None else ""
                ),
            )
            for item in detections
            for detail in [enriched_by_detection.get(_detection_key(item.label, item.bbox))]
        ],
        spatial_relations=[
            CompactSpatialRelation(
                subject_id=item.subject_id,
                predicate=item.predicate.value,
                object_id=item.object_id,
            )
            for item in relations
        ],
    )


def canonicalize_response_frame_ids(
    frames: list[GeminiFrameResult],
    supplied_frame_ids: set[str],
) -> list[GeminiFrameResult]:
    """Accept harmless zero-padding differences while preserving batch identity.

    Gemini occasionally returns ``f00010`` for an input labelled ``f0010``.
    The video ID and numeric frame index still identify the same supplied image;
    all other mismatches remain errors in the strict response-set check.
    """
    supplied_by_identity = {
        (reference.video_name, reference.frame_index): frame_id
        for frame_id in supplied_frame_ids
        if (reference := parse_frame_id(frame_id))
    }
    canonicalized: list[GeminiFrameResult] = []
    for frame in frames:
        try:
            reference = parse_frame_id(frame.frame_id)
        except ValueError:
            canonicalized.append(frame)
            continue
        canonical_id = supplied_by_identity.get(
            (reference.video_name, reference.frame_index), frame.frame_id
        )
        canonicalized.append(frame.model_copy(update={"frame_id": canonical_id}))
    return canonicalized


def map_response_slots_to_frame_ids(
    frames: list[GeminiFrameResult],
    image_paths: list[Path],
) -> list[GeminiFrameResult]:
    """Use request-local slots, not Gemini-generated IDs, to bind output to images."""
    expected_slots = set(range(1, len(image_paths) + 1))
    returned_slots = [frame.slot for frame in frames]
    if len(returned_slots) != len(set(returned_slots)) or set(returned_slots) != expected_slots:
        return frames

    expected_by_slot = {
        slot: frame_id_from_path(path) for slot, path in enumerate(image_paths, start=1)
    }
    mismatched_ids = [
        f"slot={frame.slot}: {frame.frame_id} -> {expected_by_slot[frame.slot]}"
        for frame in frames
        if frame.frame_id != expected_by_slot[frame.slot]
    ]
    if mismatched_ids:
        print(
            "Gemini frame_id disagreed with valid input slots; using the slots as the "
            f"authoritative mapping. remapped={mismatched_ids}"
        )
    return [
        frame.model_copy(update={"frame_id": expected_by_slot[frame.slot]})
        for frame in frames
    ]


def map_ocr_response_slots_to_frame_ids(
    frames: list[GeminiOCRFrameResult],
    image_paths: list[Path],
) -> list[GeminiOCRFrameResult]:
    """Bind OCR-only results to images using request-local slots."""

    expected_slots = set(range(1, len(image_paths) + 1))
    returned_slots = [frame.slot for frame in frames]
    if len(returned_slots) != len(set(returned_slots)) or set(returned_slots) != expected_slots:
        return frames
    expected_by_slot = {
        slot: frame_id_from_path(path) for slot, path in enumerate(image_paths, start=1)
    }
    return [
        frame.model_copy(update={"frame_id": expected_by_slot[frame.slot]})
        for frame in frames
    ]


def canonicalize_ocr_response_frame_ids(
    frames: list[GeminiOCRFrameResult],
    supplied_frame_ids: set[str],
) -> list[GeminiOCRFrameResult]:
    """Accept harmless zero-padding differences in OCR-only responses."""

    supplied_by_identity = {
        (reference.video_name, reference.frame_index): frame_id
        for frame_id in supplied_frame_ids
        if (reference := parse_frame_id(frame_id))
    }
    canonicalized: list[GeminiOCRFrameResult] = []
    for frame in frames:
        try:
            reference = parse_frame_id(frame.frame_id)
        except ValueError:
            canonicalized.append(frame)
            continue
        canonical_id = supplied_by_identity.get(
            (reference.video_name, reference.frame_index), frame.frame_id
        )
        canonicalized.append(frame.model_copy(update={"frame_id": canonical_id}))
    return canonicalized


class GeminiVisualExtractor:
    def __init__(
        self,
        api_key: str | None,
        model_name: str = DEFAULT_MODEL,
        context_resolver: VisualContextResolver | None = None,
        youtube_metadata_path: str | Path = DEFAULT_YOUTUBE_METADATA_PATH,
        use_video_context: bool = True,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        if genai is None or types is None:
            raise RuntimeError("google-genai is not installed")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.use_video_context = use_video_context
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.max_output_tokens = max_output_tokens
        self.context_resolver = context_resolver or VisualContextResolver.from_youtube_metadata(
            youtube_metadata_path
        )

    @staticmethod
    def batch_paths(
        image_paths: Iterable[Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
        batch_key: Callable[[Path], str] | None = None,
    ) -> list[list[Path]]:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
        if max_inline_bytes < 1:
            raise ValueError("max_inline_bytes must be positive")

        batches: list[list[Path]] = []
        current: list[Path] = []
        current_bytes = 0
        current_key: str | None = None
        for path in image_paths:
            size = path.stat().st_size
            if size > max_inline_bytes:
                raise ValueError(f"Image exceeds inline payload budget: {path}")
            path_key = batch_key(path) if batch_key is not None else None
            if current and (
                len(current) >= batch_size
                or current_bytes + size > max_inline_bytes
                or path_key != current_key
            ):
                batches.append(current)
                current = []
                current_bytes = 0
                current_key = None
            current.append(path)
            current_bytes += size
            current_key = path_key
        if current:
            batches.append(current)
        return batches

    def extract_batch(
        self,
        image_paths: list[Path],
        with_spatial: bool = True,
        retry_missing_once: bool = True,
    ) -> list[CompactVisualRecord]:
        if not image_paths:
            return []

        supplied = {frame_id_from_path(path) for path in image_paths}
        if len(supplied) != len(image_paths):
            raise ValueError("Batch contains duplicate frame_id values")

        prompt = VISUAL_PROMPT
        if self.use_video_context:
            context = self.context_resolver.for_paths(image_paths)
            prompt = build_visual_prompt(context)
        contents: list[object] = [prompt]
        for slot, path in enumerate(image_paths, start=1):
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            contents.extend(
                [
                    f"Input Slot: {slot}. Frame ID: {frame_id_from_path(path)}. "
                    f"Return slot={slot} for this image.",
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
                ]
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError(
                "Gemini returned empty response text; the batch can be retried"
            )
        payload, malformed_detections = sanitise_gemini_payload(response_text)
        if malformed_detections:
            preview = "; ".join(malformed_detections[:3])
            suffix = " ..." if len(malformed_detections) > 3 else ""
            print(
                "Gemini returned malformed detections; dropped only those boxes "
                f"({len(malformed_detections)}): {preview}{suffix}"
            )
        result = GeminiVisualBatch.model_validate(payload)
        frames = map_response_slots_to_frame_ids(result.frames, image_paths)
        frames = canonicalize_response_frame_ids(frames, supplied)
        if len(supplied) == 1 and len(frames) == 1 and frames[0].frame_id not in supplied:
            # With one supplied image there is no ambiguity: Gemini sometimes
            # hallucinates an adjacent frame number, but its sole record still
            # belongs to the only image in this request.
            frames = [frames[0].model_copy(update={"frame_id": next(iter(supplied))})]
        returned = [item.frame_id for item in frames]
        if len(returned) != len(set(returned)):
            raise ValueError("Gemini returned duplicate frame_id values")
        missing = sorted(supplied - set(returned))
        unknown = sorted(set(returned) - supplied)
        if unknown:
            print(
                "Gemini returned unknown frame IDs; discarding them and retrying "
                f"the missing frames individually. missing={missing}, unknown={unknown}"
            )
            frames = [frame for frame in frames if frame.frame_id in supplied]
            returned = [item.frame_id for item in frames]
            missing = sorted(supplied - set(returned))

        paths_by_id = {frame_id_from_path(path): path for path in image_paths}
        by_id = {item.frame_id: item for item in frames}
        if missing:
            if not retry_missing_once:
                raise ValueError(f"Gemini omitted frame_id values after retry: {missing}")
            retry_records = [
                record
                for frame_id in missing
                for record in self.extract_batch(
                    [paths_by_id[frame_id]],
                    with_spatial=with_spatial,
                    retry_missing_once=False,
                )
            ]
            retry_by_id = {record.frame_id: record for record in retry_records}
            return [
                compact_record_from_gemini(by_id[frame_id], paths_by_id[frame_id], with_spatial)
                if frame_id in by_id
                else retry_by_id[frame_id]
                for frame_id in sorted(supplied)
            ]
        return [
            compact_record_from_gemini(by_id[frame_id], paths_by_id[frame_id], with_spatial)
            for frame_id in sorted(supplied)
        ]

    def extract_ocr_batch(
        self,
        image_paths: list[Path],
        retry_missing_once: bool = True,
    ) -> list[GeminiOCRFrameResult]:
        """Extract only frame-local OCR and ticker text for deduplicated frames."""

        if not image_paths:
            return []
        supplied = {frame_id_from_path(path) for path in image_paths}
        if len(supplied) != len(image_paths):
            raise ValueError("Batch contains duplicate frame_id values")

        prompt = OCR_ONLY_PROMPT
        if self.use_video_context:
            prompt = build_ocr_prompt(self.context_resolver.for_paths(image_paths))
        contents: list[object] = [prompt]
        for slot, path in enumerate(image_paths, start=1):
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            contents.extend(
                [
                    f"Input Slot: {slot}. Frame ID: {frame_id_from_path(path)}. "
                    f"Return slot={slot} for this image.",
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
                ]
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_OCR_RESPONSE_SCHEMA,
                temperature=0.0,
                max_output_tokens=min(self.max_output_tokens, 8_192),
            ),
        )
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("Gemini returned empty OCR response text; the batch can be retried")
        result = GeminiOCRBatch.model_validate(json.loads(response_text))
        frames = map_ocr_response_slots_to_frame_ids(result.frames, image_paths)
        frames = canonicalize_ocr_response_frame_ids(frames, supplied)
        if len(supplied) == 1 and len(frames) == 1 and frames[0].frame_id not in supplied:
            frames = [frames[0].model_copy(update={"frame_id": next(iter(supplied))})]

        returned = [item.frame_id for item in frames]
        if len(returned) != len(set(returned)):
            raise ValueError("Gemini returned duplicate OCR frame_id values")
        missing = sorted(supplied - set(returned))
        unknown = sorted(set(returned) - supplied)
        if unknown:
            frames = [frame for frame in frames if frame.frame_id in supplied]
            returned = [item.frame_id for item in frames]
            missing = sorted(supplied - set(returned))
        if missing:
            if not retry_missing_once:
                raise ValueError(f"Gemini omitted OCR frame_id values after retry: {missing}")
            paths_by_id = {frame_id_from_path(path): path for path in image_paths}
            retry_records = [
                record
                for frame_id in missing
                for record in self.extract_ocr_batch(
                    [paths_by_id[frame_id]], retry_missing_once=False
                )
            ]
            retry_by_id = {record.frame_id: record for record in retry_records}
            by_id = {item.frame_id: item for item in frames}
            return [by_id[frame_id] if frame_id in by_id else retry_by_id[frame_id] for frame_id in sorted(supplied)]
        by_id = {item.frame_id: item for item in frames}
        return [by_id[frame_id] for frame_id in sorted(supplied)]


def _load_existing(output_path: Path, resume: bool) -> dict[str, CompactVisualRecord]:
    if not output_path.exists():
        return {}
    if not resume:
        raise FileExistsError(f"{output_path} exists; pass --resume or --force")
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Existing output must be a JSON array")
    records = [CompactVisualRecord.model_validate(item) for item in raw]
    return {record.frame_id: record for record in records}


def caption_output_path(output_dir: str | Path, video_id: str) -> Path:
    """Return the per-video artifact path under ``caption/LXX``."""
    batch_id = video_id.split("_", maxsplit=1)[0]
    if not batch_id:
        raise ValueError(f"Cannot derive batch ID from video ID: {video_id!r}")
    return Path(output_dir) / batch_id / f"{video_id}.json"


def _copy_record_to_frame(
    record: CompactVisualRecord,
    frame_id: str,
    *,
    ocr_source_frame_id: str | None = None,
) -> CompactVisualRecord:
    is_duplicate = frame_id != record.frame_id
    flags = list(record.quality_flags)
    if is_duplicate and "visual_metadata_copied" not in flags:
        flags.append("visual_metadata_copied")
    return record.model_copy(
        update={
            "frame_id": frame_id,
            "visual_source_frame_id": record.frame_id,
            "ocr_source_frame_id": (
                ocr_source_frame_id
                if ocr_source_frame_id is not None
                else (frame_id if not is_duplicate else "")
            ),
            "quality_flags": sorted(set(flags)),
        }
    )


def _write_video_records(
    output_dir: Path,
    video_id: str,
    records: dict[str, CompactVisualRecord],
) -> None:
    write_json_atomically(
        caption_output_path(output_dir, video_id),
        [record.model_dump(mode="json") for _, record in sorted(records.items())],
        overwrite=True,
    )


def _validate_rate_controls(
    requests_per_minute: int,
    max_concurrent_requests: int,
    daily_request_limit: int,
    daily_requests_already_used: int,
) -> None:
    if requests_per_minute < 1:
        raise ValueError("requests per minute must be at least 1")
    if not 1 <= max_concurrent_requests <= requests_per_minute:
        raise ValueError("concurrent requests must be between 1 and requests per minute")


def run_extraction(
    input_dir: str | Path,
    output_dir: str | Path,
    api_key: str | None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
    resume: bool = False,
    force: bool = False,
    with_spatial: bool = True,
    youtube_metadata_path: str | Path = DEFAULT_YOUTUBE_METADATA_PATH,
    use_video_context: bool = True,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    global_filter_results_path: str | Path | None = DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_REQUESTS_PER_MINUTE,
    daily_request_limit: int = DEFAULT_DAILY_REQUEST_LIMIT,
    daily_requests_already_used: int = 0,
    request_budget_state_path: str | Path | None = None,
    video_prefix: str | None = None,
) -> dict[str, int]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path must be a directory: {output_dir}")
    _validate_rate_controls(
        requests_per_minute,
        max_concurrent_requests,
        daily_request_limit,
        daily_requests_already_used,
    )

    all_image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if video_prefix is not None:
        normalized_prefix = video_prefix.strip().upper()
        if not normalized_prefix:
            raise ValueError("video prefix cannot be empty")
        all_image_paths = [
            path
            for path in all_image_paths
            if parse_frame_id(frame_id_from_path(path)).video_name.startswith(
                f"{normalized_prefix}_"
            )
        ]
    all_paths_by_frame = {
        frame_id_from_path(path): path for path in all_image_paths
    }
    image_paths = all_image_paths
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        image_paths = image_paths[:limit]

    deduplication: GlobalFrameDeduplication | None = None
    if global_filter_results_path is not None:
        deduplication = load_global_frame_deduplication(global_filter_results_path)

    paths_by_video: dict[str, list[Path]] = {}
    for image_path in image_paths:
        video_id = parse_frame_id(frame_id_from_path(image_path)).video_name
        paths_by_video.setdefault(video_id, []).append(image_path)

    existing_by_video: dict[str, dict[str, CompactVisualRecord]] = {}
    pending_by_video: dict[str, dict[str, tuple[Path, list[str]]]] = {}
    pending_ocr_by_video: dict[str, list[Path]] = {}
    copied = 0
    for video_id, video_paths in paths_by_video.items():
        output_path = caption_output_path(output_dir, video_id)
        existing = {} if force else _load_existing(output_path, resume=resume)
        unexpected = [
            frame_id
            for frame_id in existing
            if parse_frame_id(frame_id).video_name != video_id
        ]
        if unexpected:
            raise ValueError(f"{output_path} contains frames from another video")
        existing_by_video[video_id] = existing

        paths_by_frame = {frame_id_from_path(path): path for path in video_paths}
        frame_ids_by_representative: dict[str, list[str]] = {}
        for frame_id in paths_by_frame:
            representative_id = (
                deduplication.representative_for(frame_id)
                if deduplication is not None
                else frame_id
            )
            frame_ids_by_representative.setdefault(representative_id, []).append(frame_id)

        pending_groups: dict[str, tuple[Path, list[str]]] = {}
        pending_ocr_paths: list[Path] = []
        for representative_id, member_ids in frame_ids_by_representative.items():
            source_record = existing.get(representative_id)
            if source_record is not None:
                for frame_id in member_ids:
                    previous_record = existing.get(frame_id)
                    copied_record = _copy_record_to_frame(source_record, frame_id)
                    if (
                        frame_id != representative_id
                        and previous_record is not None
                        and previous_record.ocr_source_frame_id == frame_id
                    ):
                        # A completed OCR-only result is frame-local and must survive
                        # a resume pass that refreshes the representative visual copy.
                        copied_record = copied_record.model_copy(
                            update={
                                "ocr_text": previous_record.ocr_text,
                                "news_ticker_text": previous_record.news_ticker_text,
                                "ocr_source_frame_id": frame_id,
                                "quality_flags": sorted(
                                    set(copied_record.quality_flags)
                                    | set(previous_record.quality_flags)
                                ),
                            }
                        )
                    if existing.get(frame_id) != copied_record:
                        existing[frame_id] = copied_record
                        copied += 1
            else:
                representative_path = paths_by_frame.get(representative_id)
                if representative_path is None:
                    representative_path = all_paths_by_frame.get(representative_id)
                if representative_path is None:
                    # The representative lies outside the selected input folder.
                    # Analyze a selected member rather than silently omitting it.
                    representative_path = paths_by_frame[member_ids[0]]
                pending_groups[frame_id_from_path(representative_path)] = (
                    representative_path,
                    member_ids,
                )

            for frame_id in member_ids:
                if frame_id == representative_id:
                    continue
                existing_record = existing.get(frame_id)
                if existing_record is not None and existing_record.ocr_source_frame_id == frame_id:
                    continue
                duplicate_path = paths_by_frame.get(frame_id) or all_paths_by_frame.get(frame_id)
                if duplicate_path is not None:
                    pending_ocr_paths.append(duplicate_path)
        pending_by_video[video_id] = pending_groups
        pending_ocr_by_video[video_id] = pending_ocr_paths
        _write_video_records(output_dir, video_id, existing)

    pending_paths = [
        request_path
        for groups in pending_by_video.values()
        for request_path, _ in groups.values()
    ]
    pending_ocr_paths = [path for paths in pending_ocr_by_video.values() for path in paths]
    if not pending_paths and not pending_ocr_paths:
        return {
            "processed": copied,
            "api_frames": 0,
            "total": sum(len(records) for records in existing_by_video.values()),
            "batches": 0,
            "daily_limit_reached": False,
            "daily_quota_reported": False,
        }

    extractor = GeminiVisualExtractor(
        api_key=api_key,
        model_name=model_name,
        youtube_metadata_path=youtube_metadata_path,
        use_video_context=use_video_context,
        max_output_tokens=max_output_tokens,
    )
    visual_batches: list[tuple[str, list[Path]]] = []
    for video_id in sorted(pending_by_video):
        request_paths = [request_path for request_path, _ in pending_by_video[video_id].values()]
        visual_batches.extend(
            (video_id, batch)
            for batch in extractor.batch_paths(
                request_paths,
                batch_size=batch_size,
                max_inline_bytes=max_inline_bytes,
            )
        )
    ocr_batches: list[tuple[str, list[Path]]] = []
    for video_id in sorted(pending_ocr_by_video):
        if not pending_ocr_by_video[video_id]:
            continue
        ocr_batches.extend(
            (video_id, batch)
            for batch in extractor.batch_paths(
                pending_ocr_by_video[video_id],
                batch_size=batch_size,
                max_inline_bytes=max_inline_bytes,
            )
        )

    total_batches = len(visual_batches) + len(ocr_batches)
    print(
        f"Starting extraction: total_batches={total_batches} "
        f"visual_batches={len(visual_batches)} ocr_batches={len(ocr_batches)}"
    )

    processed = copied
    api_frames = 0
    completed_batches = 0
    daily_limit_reached = False
    daily_quota_reported = False
    previous_window_started: float | None = None
    def run_batches(
        batches: list[tuple[str, list[Path]]],
        *,
        mode: str,
    ) -> None:
        nonlocal processed, api_frames, completed_batches
        next_batch_index = 0
        nonlocal previous_window_started
        transient_retry_counts: dict[tuple[str, str, tuple[str, ...]], int] = {}
        rate_retry_counts: dict[tuple[str, str, tuple[str, ...]], int] = {}

        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            while next_batch_index < len(batches):
                if previous_window_started is not None:
                    wait_seconds = 60 - (time.monotonic() - previous_window_started)
                    if wait_seconds > 0:
                        print(
                            f"RPM window complete; waiting {wait_seconds:.1f}s before the next window."
                        )
                        time.sleep(wait_seconds)

                window_size = min(requests_per_minute, len(batches) - next_batch_index)
                window = batches[next_batch_index : next_batch_index + window_size]
                previous_window_started = time.monotonic()
                if mode == "visual":
                    futures = {
                        executor.submit(
                            extractor.extract_batch,
                            batch,
                            with_spatial=with_spatial,
                        ): (video_id, batch)
                        for video_id, batch in window
                    }
                else:
                    futures = {
                        executor.submit(extractor.extract_ocr_batch, batch): (video_id, batch)
                        for video_id, batch in window
                    }
                next_batch_index += len(window)
                unexpected_error: Exception | None = None
                retry_batches: list[tuple[str, list[Path]]] = []

                for future in as_completed(futures):
                    video_id, batch = futures[future]
                    if future.cancelled():
                        continue
                    retry_key = (mode, video_id, tuple(frame_id_from_path(path) for path in batch))
                    try:
                        records = future.result()
                    except Exception as error:
                        if is_rate_limit_error(error):
                            attempt = rate_retry_counts.get(retry_key, 0) + 1
                            if attempt <= MAX_RATE_LIMIT_RETRIES:
                                rate_retry_counts[retry_key] = attempt
                                retry_batches.append((video_id, batch))
                                print(
                                    f"RPM quota reached for {video_id} ({mode}); "
                                    f"retry {attempt}/{MAX_RATE_LIMIT_RETRIES} in the next window."
                                )
                            else:
                                unexpected_error = RuntimeError(
                                    f"RPM quota remained unavailable after {MAX_RATE_LIMIT_RETRIES} "
                                    f"retries for {video_id} ({mode})"
                                )
                            continue
                        if is_transient_service_error(error):
                            attempt = transient_retry_counts.get(retry_key, 0) + 1
                            if attempt <= MAX_TRANSIENT_RETRIES:
                                transient_retry_counts[retry_key] = attempt
                                retry_batches.append((video_id, batch))
                                print(
                                    f"Gemini service unavailable for {video_id} ({mode}); "
                                    f"retry {attempt}/{MAX_TRANSIENT_RETRIES} in the next window."
                                )
                            else:
                                unexpected_error = RuntimeError(
                                    f"Gemini remained unavailable after {MAX_TRANSIENT_RETRIES} "
                                    f"retries for {video_id} ({mode})"
                                )
                            continue
                        unexpected_error = error
                        continue

                    existing = existing_by_video[video_id]
                    if mode == "visual":
                        for record in records:
                            _, member_ids = pending_by_video[video_id][record.frame_id]
                            for frame_id in member_ids:
                                updated = _copy_record_to_frame(record, frame_id)
                                if existing.get(frame_id) != updated:
                                    existing[frame_id] = updated
                                    processed += 1
                    else:
                        for record in records:
                            current = existing.get(record.frame_id)
                            if current is None:
                                raise ValueError(
                                    f"OCR result has no visual record for {record.frame_id}"
                                )
                            updated = current.model_copy(
                                update={
                                    "ocr_text": _clean_text(record.ocr_text),
                                    "news_ticker_text": _clean_text(record.news_ticker_text),
                                    "ocr_source_frame_id": record.frame_id,
                                }
                            )
                            if current != updated:
                                existing[record.frame_id] = updated
                                processed += 1
                    api_frames += len(records)
                    completed_batches += 1
                    _write_video_records(output_dir, video_id, existing)
                    transient_retry_counts.pop(retry_key, None)
                    rate_retry_counts.pop(retry_key, None)
                    print(
                        f"batch={completed_batches}/{total_batches} mode={mode} video={video_id} "
                        f"api_frames={api_frames} processed={processed} "
                        f"video_total={len(existing)}"
                    )

                if unexpected_error is not None:
                    raise unexpected_error
                if retry_batches:
                    batches.extend(retry_batches)

    # Complete visual representatives first so OCR-only jobs can merge into
    # already materialized duplicate records deterministically.
    run_batches(visual_batches, mode="visual")
    run_batches(ocr_batches, mode="ocr")

    return {
        "processed": processed,
        "api_frames": api_frames,
        "total": sum(len(records) for records in existing_by_video.values()),
        "batches": completed_batches,
        "daily_limit_reached": daily_limit_reached,
        "daily_quota_reported": daily_quota_reported,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact caption, OCR and spatial metadata with Gemini."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument(
        "--video-prefix",
        help="Only process videos whose IDs start with this batch prefix, for example L21",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metadata/caption"),
        help="Per-video JSON output root; files are written as LXX/LXX_VYYY.json",
    )
    parser.add_argument("--model", default=load_gemini_visual_model())
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Maximum Gemini requests started in each 60-second window",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Parallel Gemini requests (must not exceed --requests-per-minute)",
    )
    parser.add_argument(
        "--daily-request-limit",
        type=int,
        default=DEFAULT_DAILY_REQUEST_LIMIT,
        help="Legacy compatibility option; daily quota is checked manually and not enforced locally",
    )
    parser.add_argument(
        "--daily-requests-already-used",
        type=int,
        default=0,
        help="Legacy compatibility option; ignored because daily quota is checked manually",
    )
    parser.add_argument(
        "--request-budget-state",
        type=Path,
        help="Legacy compatibility option; no local daily budget state is written",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-inline-bytes", type=int, default=DEFAULT_MAX_INLINE_BYTES)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--youtube-metadata",
        type=Path,
        default=DEFAULT_YOUTUBE_METADATA_PATH,
        help="JSONL video context used to select the prompt hint",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--global-filter-results",
        type=Path,
        default=DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
        help="CSV that maps near-duplicate frames to a representative",
    )
    parser.add_argument(
        "--without-global-filter",
        action="store_true",
        help="Send every selected frame to Gemini instead of copying representative metadata",
    )
    parser.add_argument("--without-spatial", action="store_true")
    parser.add_argument(
        "--generic-prompt",
        action="store_true",
        help="Disable video-level context; intended for A/B prompt evaluation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_extraction(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            api_key=load_gemini_api_key(),
            model_name=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            max_inline_bytes=args.max_inline_bytes,
            resume=args.resume,
            force=args.force,
            with_spatial=not args.without_spatial,
            youtube_metadata_path=args.youtube_metadata,
            use_video_context=not args.generic_prompt,
            max_output_tokens=args.max_output_tokens,
            global_filter_results_path=(
                None if args.without_global_filter else args.global_filter_results
            ),
            requests_per_minute=args.requests_per_minute,
            max_concurrent_requests=args.max_concurrent_requests,
            daily_request_limit=args.daily_request_limit,
            daily_requests_already_used=args.daily_requests_already_used,
            request_budget_state_path=args.request_budget_state,
            video_prefix=args.video_prefix,
        )
    except Exception as exc:
        raise SystemExit(f"Gemini visual extraction failed: {exc}") from exc
    print(
        "Gemini visual extraction complete: "
        f"processed={summary['processed']}, api_frames={summary['api_frames']}, "
        f"batches={summary['batches']}, total={summary['total']}, "
        f"daily_limit_reached={summary.get('daily_limit_reached', False)}, "
        f"daily_quota_reported={summary.get('daily_quota_reported', False)}"
    )


if __name__ == "__main__":
    main()
