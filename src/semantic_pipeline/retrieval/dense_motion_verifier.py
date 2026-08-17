"""Fast runtime-dense temporal verifier using CLIP plus frame motion.

The sparse timeline still selects the video and candidate event.  This verifier
decodes that event densely from the raw video, batches all frames through CLIP,
and combines visual-query similarity with adjacent-frame change.  It performs
no autoregressive generation, so it is intended as the low-latency online path
while a VLM remains an optional fallback.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .qwen_video_verifier import TemporalVerificationRequest, TemporalVerificationResult
from .video_window_sampler import sample_video_window


_TEMPORAL_PREFIX_RE = re.compile(
    r"(?i)\b(?:khoảnh khắc|khoanh khac|đầu tiên|dau tien|first|earliest|moment|"
    r"completely|hoàn toàn|hoan toan)\b"
)


@dataclass(frozen=True)
class DenseMotionConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    translator_model_id: str = "Helsinki-NLP/opus-mt-vi-en"
    sample_fps: float = 2.0
    max_frames: int = 192
    batch_size: int = 64
    padding_ms: int = 2_000
    semantic_weight: float = 0.78
    motion_weight: float = 0.22

    @classmethod
    def from_environment(cls) -> "DenseMotionConfig":
        def integer(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def decimal(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            model_id=os.getenv("DENSE_TEMPORAL_CLIP_MODEL", cls.model_id),
            translator_model_id=os.getenv("DENSE_TEMPORAL_TRANSLATOR_MODEL", cls.translator_model_id),
            sample_fps=decimal("DENSE_TEMPORAL_FPS", cls.sample_fps),
            max_frames=integer("DENSE_TEMPORAL_MAX_FRAMES", cls.max_frames),
            batch_size=integer("DENSE_TEMPORAL_BATCH_SIZE", cls.batch_size),
            padding_ms=integer("DENSE_TEMPORAL_PADDING_MS", cls.padding_ms),
            semantic_weight=decimal("DENSE_TEMPORAL_SEMANTIC_WEIGHT", cls.semantic_weight),
            motion_weight=decimal("DENSE_TEMPORAL_MOTION_WEIGHT", cls.motion_weight),
        )


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def select_dense_boundary(
    semantic_scores: Sequence[float],
    motion_scores: Sequence[float],
    required_anchor: str,
    *,
    semantic_weight: float = 0.78,
    motion_weight: float = 0.22,
) -> tuple[int, float]:
    """Select a temporal index without depending on torch/OpenCV.

    Semantic similarity establishes event presence. Motion only resolves the
    boundary inside the high-semantic region, preventing a camera cut with no
    matching subject from winning solely because it moves strongly.
    """

    if not semantic_scores or len(semantic_scores) != len(motion_scores):
        raise ValueError("semantic and motion scores must be non-empty and aligned")
    semantic = _normalize(semantic_scores)
    motion = _normalize(motion_scores)
    peak = max(semantic)
    threshold = max(0.55, peak - 0.20)
    eligible = [index for index, score in enumerate(semantic) if score >= threshold]
    if not eligible:
        eligible = [max(range(len(semantic)), key=semantic.__getitem__)]
    combined = [
        semantic_weight * semantic[index] + motion_weight * motion[index]
        for index in range(len(semantic))
    ]

    if required_anchor in {"first_visible", "action_start", "first_contact", "state_complete"}:
        # Select the first high-semantic run. For actions/contact, prefer a
        # motion peak within the first second-sized neighborhood of that run.
        first = eligible[0]
        if required_anchor in {"first_visible", "state_complete"}:
            selected = first
        else:
            local = [index for index in eligible if index <= first + 1]
            selected = max(local, key=lambda index: (combined[index], -index))
    elif required_anchor in {"last_complete", "action_end"}:
        selected = eligible[-1]
    else:
        selected = max(eligible, key=lambda index: (combined[index], -index))
    return selected, max(0.0, min(1.0, combined[selected]))


class DenseMotionTemporalVerifier:
    """Runtime dense verifier with batched, non-generative inference."""

    # A dense scan is allowed to move throughout the retrieved event window.
    max_anchor_shift_ms: int | None = None

    def __init__(self, config: DenseMotionConfig | None = None) -> None:
        self.config = config or DenseMotionConfig.from_environment()
        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
        self._translator: Any | None = None
        self._translator_tokenizer: Any | None = None
        self._translations: dict[str, str] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("dense temporal verification requires torch, transformers and Pillow") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(self.config.model_id).to(device).eval()
        if device == "cuda":
            self._model = self._model.half()
            torch.backends.cudnn.benchmark = True
        self._processor = CLIPProcessor.from_pretrained(self.config.model_id)
        self._torch = torch
        self._device = device

    @staticmethod
    def _visual_query(event_text: str) -> str:
        cleaned = " ".join(_TEMPORAL_PREFIX_RE.sub(" ", event_text).split()).strip(" .,:;-")
        parenthetical = re.findall(r"\(([^()]*)\)", cleaned)
        english_hint = next(
            (
                item.strip()
                for item in reversed(parenthetical)
                if len(re.findall(r"[A-Za-z]{2,}", item)) >= 3
            ),
            "",
        )
        if english_hint:
            cleaned = english_hint
        return f"a video frame showing {cleaned or event_text}"

    def _english_visual_query(self, event_text: str) -> str:
        visual_query = self._visual_query(event_text)
        content = visual_query.removeprefix("a video frame showing ")
        if not any(ord(character) > 127 for character in content):
            return visual_query
        if content in self._translations:
            return f"a video frame showing {self._translations[content]}"
        try:
            from transformers import MarianMTModel, MarianTokenizer

            if self._translator is None or self._translator_tokenizer is None:
                self._translator_tokenizer = MarianTokenizer.from_pretrained(
                    self.config.translator_model_id
                )
                self._translator = MarianMTModel.from_pretrained(
                    self.config.translator_model_id
                ).eval()
            inputs = self._translator_tokenizer(
                [content],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            translated_ids = self._translator.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
            )
            translated = " ".join(
                self._translator_tokenizer.batch_decode(
                    translated_ids,
                    skip_special_tokens=True,
                )[0].split()
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            translated = content
        self._translations[content] = translated
        return f"a video frame showing {translated}"

    @staticmethod
    def _motion_scores(frames: Sequence[Any]) -> list[float]:
        import cv2

        scores = [0.0]
        previous = None
        for frame in frames:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if previous is not None:
                scores.append(float(cv2.absdiff(gray, previous).mean()) / 255.0)
            previous = gray
        return scores[: len(frames)]

    def _semantic_scores(self, frames: Sequence[Any], event_text: str) -> list[float]:
        self._load_model()
        assert self._torch is not None and self._model is not None
        assert self._processor is not None and self._device is not None
        from PIL import Image

        text_inputs = self._processor(
            text=[self._english_visual_query(event_text)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)
        with self._torch.inference_mode():
            text_features = self._model.get_text_features(**text_inputs)
            if hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        scores: list[float] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            images = [Image.fromarray(frame.image) for frame in batch]
            image_inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            if self._device == "cuda":
                image_inputs["pixel_values"] = image_inputs["pixel_values"].half()
            with self._torch.inference_mode():
                image_features = self._model.get_image_features(**image_inputs)
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                batch_scores = (image_features @ text_features.T).squeeze(-1).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores

    def verify(self, request: TemporalVerificationRequest) -> TemporalVerificationResult:
        started = time.perf_counter()
        start_ms = max(0, request.start_ms - self.config.padding_ms)
        end_ms = max(start_ms, request.end_ms + self.config.padding_ms)
        window = sample_video_window(
            request.video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            fps=self.config.sample_fps,
            max_frames=self.config.max_frames,
        )
        decoded_at = time.perf_counter()
        semantic = self._semantic_scores(window.frames, request.event_text)
        encoded_at = time.perf_counter()
        motion = self._motion_scores(window.frames)
        selected_index, confidence = select_dense_boundary(
            semantic,
            motion,
            request.required_anchor,
            semantic_weight=self.config.semantic_weight,
            motion_weight=self.config.motion_weight,
        )
        state = (
            "complete"
            if request.required_anchor in {"last_complete", "action_end"}
            else "transition"
            if request.required_anchor in {"action_start", "first_contact", "first_visible", "state_complete"}
            else "uncertain"
        )
        return TemporalVerificationResult(
            supported=True,
            timestamp_ms=window.frames[selected_index].timestamp_ms,
            confidence=confidence,
            selected_state=state,
            verifier=f"dense-motion:{self.config.model_id}",
            coarse_window_ms=(window.start_ms, window.end_ms),
            reason=(
                f"dense CLIP+motion frame {selected_index + 1}/{len(window.frames)}; "
                f"semantic={semantic[selected_index]:.4f}; motion={motion[selected_index]:.4f}; "
                f"decode_ms={(decoded_at - started) * 1_000:.0f}; "
                f"encode_ms={(encoded_at - decoded_at) * 1_000:.0f}"
            ),
        )
