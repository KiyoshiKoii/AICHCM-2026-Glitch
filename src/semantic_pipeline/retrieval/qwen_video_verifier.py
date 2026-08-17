"""Optional Qwen-VL verifier for fine temporal-event boundaries.

It is intentionally a reranker/verifier, not a dataset-wide embedder: callers
provide one candidate video interval and Qwen receives only sampled frames from
that interval.  Imports of the heavyweight model stack stay lazy so the normal
retrieval service remains usable without a local Qwen installation.
"""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .video_window_sampler import VideoWindow, bounded_window, sample_video_window


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_STATES = frozenset({"before", "transition", "after", "complete", "uncertain"})


@dataclass(frozen=True)
class TemporalVerificationRequest:
    video_id: str
    event_id: str
    event_text: str
    required_anchor: str
    video_path: Path
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TemporalVerificationResult:
    supported: bool
    timestamp_ms: int | None
    confidence: float
    selected_state: str
    verifier: str
    coarse_window_ms: tuple[int, int]
    refined_window_ms: tuple[int, int] | None = None
    reason: str = ""


class TemporalEventVerifier(Protocol):
    """Small interface that lets retrieval use Qwen or a deterministic fake."""

    def verify(self, request: TemporalVerificationRequest) -> TemporalVerificationResult:
        """Verify a candidate event and return its strongest temporal boundary."""


@dataclass(frozen=True)
class QwenVerifierConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    coarse_padding_ms: int = 8_000
    coarse_fps: float = 1.0
    coarse_max_frames: int = 8
    refine_radius_ms: int = 1_500
    refine_fps: float = 4.0
    refine_max_frames: int = 12
    image_max_side: int = 280
    max_new_tokens: int = 96
    load_in_4bit: bool = True
    cache_dir: Path = Path("data/processed/qwen_temporal_cache")

    @classmethod
    def from_environment(cls) -> "QwenVerifierConfig":
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

        value = os.getenv("QWEN_TEMPORAL_LOAD_IN_4BIT", "true").strip().casefold()
        return cls(
            model_id=os.getenv("QWEN_TEMPORAL_MODEL", cls.model_id),
            coarse_padding_ms=integer("QWEN_TEMPORAL_COARSE_PADDING_MS", cls.coarse_padding_ms),
            coarse_fps=decimal("QWEN_TEMPORAL_COARSE_FPS", cls.coarse_fps),
            coarse_max_frames=integer("QWEN_TEMPORAL_COARSE_MAX_FRAMES", cls.coarse_max_frames),
            refine_radius_ms=integer("QWEN_TEMPORAL_REFINE_RADIUS_MS", cls.refine_radius_ms),
            refine_fps=decimal("QWEN_TEMPORAL_REFINE_FPS", cls.refine_fps),
            refine_max_frames=integer("QWEN_TEMPORAL_REFINE_MAX_FRAMES", cls.refine_max_frames),
            image_max_side=integer("QWEN_TEMPORAL_IMAGE_MAX_SIDE", cls.image_max_side),
            max_new_tokens=integer("QWEN_TEMPORAL_MAX_NEW_TOKENS", cls.max_new_tokens),
            load_in_4bit=value not in {"0", "false", "no", "off"},
            cache_dir=Path(os.getenv("QWEN_TEMPORAL_CACHE_DIR", str(cls.cache_dir))),
        )


class QwenTemporalVerifier:
    """Two-pass local Qwen-VL verifier using coarse then dense frame sampling."""

    def __init__(self, config: QwenVerifierConfig | None = None) -> None:
        self.config = config or QwenVerifierConfig.from_environment()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def verify(self, request: TemporalVerificationRequest) -> TemporalVerificationResult:
        cached = self._read_cache(request)
        if cached is not None:
            return cached
        coarse_start, coarse_end = bounded_window(
            center_ms=(request.start_ms + request.end_ms) // 2,
            radius_ms=max(
                self.config.coarse_padding_ms,
                (request.end_ms - request.start_ms) // 2 + self.config.coarse_padding_ms,
            ),
            duration_ms=self._video_duration_ms(request.video_path),
        )
        coarse = sample_video_window(
            request.video_path,
            start_ms=coarse_start,
            end_ms=coarse_end,
            fps=self.config.coarse_fps,
            max_frames=self.config.coarse_max_frames,
        )
        coarse_payload = self._infer(request, coarse)
        if not coarse_payload["supported"] or coarse_payload["timestamp_ms"] is None:
            result = TemporalVerificationResult(
                supported=False,
                timestamp_ms=None,
                confidence=coarse_payload["confidence"],
                selected_state=coarse_payload["selected_state"],
                verifier=f"qwen:{self.config.model_id}",
                coarse_window_ms=(coarse.start_ms, coarse.end_ms),
                reason=coarse_payload["reason"],
            )
            self._write_cache(request, result)
            return result

        refine_start, refine_end = bounded_window(
            center_ms=coarse_payload["timestamp_ms"],
            radius_ms=self.config.refine_radius_ms,
            duration_ms=coarse.duration_ms,
        )
        refined = sample_video_window(
            request.video_path,
            start_ms=refine_start,
            end_ms=refine_end,
            fps=self.config.refine_fps,
            max_frames=self.config.refine_max_frames,
        )
        refined_payload = self._infer(request, refined)
        selected = refined_payload if refined_payload["supported"] else coarse_payload
        result = TemporalVerificationResult(
            supported=selected["supported"],
            timestamp_ms=selected["timestamp_ms"],
            confidence=selected["confidence"],
            selected_state=selected["selected_state"],
            verifier=f"qwen:{self.config.model_id}",
            coarse_window_ms=(coarse.start_ms, coarse.end_ms),
            refined_window_ms=(refined.start_ms, refined.end_ms),
            reason=selected["reason"],
        )
        self._write_cache(request, result)
        return result

    def _cache_path(self, request: TemporalVerificationRequest) -> Path:
        """Keep only deterministic model/request identity in the cache key."""

        key_payload = {
            "model_id": self.config.model_id,
            "load_in_4bit": self.config.load_in_4bit,
            "video_id": request.video_id,
            "event_id": request.event_id,
            "event_text": request.event_text,
            "required_anchor": request.required_anchor,
            "start_ms": request.start_ms,
            "end_ms": request.end_ms,
            "coarse_fps": self.config.coarse_fps,
            "coarse_max_frames": self.config.coarse_max_frames,
            "coarse_padding_ms": self.config.coarse_padding_ms,
            "refine_fps": self.config.refine_fps,
            "refine_max_frames": self.config.refine_max_frames,
            "refine_radius_ms": self.config.refine_radius_ms,
            "image_max_side": self.config.image_max_side,
        }
        digest = sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.config.cache_dir / request.video_id / f"{digest}.json"

    def _read_cache(self, request: TemporalVerificationRequest) -> TemporalVerificationResult | None:
        path = self._cache_path(request)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return TemporalVerificationResult(
                supported=bool(payload["supported"]),
                timestamp_ms=int(payload["timestamp_ms"]) if payload.get("timestamp_ms") is not None else None,
                confidence=float(payload["confidence"]),
                selected_state=str(payload["selected_state"]),
                verifier=str(payload["verifier"]),
                coarse_window_ms=tuple(int(value) for value in payload["coarse_window_ms"]),
                refined_window_ms=(
                    tuple(int(value) for value in payload["refined_window_ms"])
                    if payload.get("refined_window_ms") is not None
                    else None
                ),
                reason=str(payload.get("reason", "")),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _write_cache(self, request: TemporalVerificationRequest, result: TemporalVerificationResult) -> None:
        path = self._cache_path(request)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "supported": result.supported,
                        "timestamp_ms": result.timestamp_ms,
                        "confidence": result.confidence,
                        "selected_state": result.selected_state,
                        "verifier": result.verifier,
                        "coarse_window_ms": list(result.coarse_window_ms),
                        "refined_window_ms": list(result.refined_window_ms) if result.refined_window_ms else None,
                        "reason": result.reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError:
            # Cache is a latency optimization; an unwritable cache must not
            # turn a valid temporal answer into a failed request.
            return

    @staticmethod
    def _video_duration_ms(video_path: Path) -> int:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("Qwen temporal verification requires opencv-python") from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open raw video: {video_path}")
        try:
            return max(
                0,
                round(
                    1_000
                    * capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    / max(capture.get(cv2.CAP_PROP_FPS), 1.0)
                ),
            )
        finally:
            capture.release()

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "Qwen temporal verification requires torch, transformers and bitsandbytes; "
                "install the optional qwen extras first"
            ) from exc
        model_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
        if self.config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        self._torch = torch

    def _infer(self, request: TemporalVerificationRequest, window: VideoWindow) -> dict[str, Any]:
        self._load_model()
        assert self._model is not None and self._processor is not None and self._torch is not None
        prompt = self._prompt(request, window)
        from PIL import Image

        images = [self._resize_image(Image.fromarray(frame.image)) for frame in window.frames]
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": prompt},
        ]}]
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_response(response, window)

    def _resize_image(self, image: Any) -> Any:
        """Bound total visual tokens so Qwen-VL fits an 8 GB GPU.

        Qwen's image encoder otherwise retains each raw HD keyframe.  With a
        frame sequence that explodes the multimodal attention context, even
        though the 4-bit language weights themselves fit in VRAM.
        """

        max_side = max(28, int(self.config.image_max_side))
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        target_width = max(28, round(width * scale / 28) * 28)
        target_height = max(28, round(height * scale / 28) * 28)
        if (target_width, target_height) == image.size:
            return image
        from PIL import Image

        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _prompt(request: TemporalVerificationRequest, window: VideoWindow) -> str:
        frame_lines = "\n".join(
            f"{index}: {frame.timestamp_ms} ms"
            for index, frame in enumerate(window.frames)
        )
        return (
            "You verify a temporal video event from ordered frames. Use only visible evidence. "
            "The requested event is:\n"
            f"{request.event_text}\n"
            f"Requested boundary: {request.required_anchor}.\n"
            "Choose the one frame index that best answers the requested boundary. "
            "For first_visible/action_start/first_contact/state_complete choose the earliest supported frame; "
            "for last_complete choose the latest supported frame. If unsupported, set supported=false and "
            "selected_frame_index=null. Return JSON only with exactly these fields: supported (boolean), "
            "selected_frame_index (integer or null), selected_state (before|transition|after|complete|uncertain), "
            "confidence (0..1), reason (short Vietnamese or English).\n"
            "Frame index to timestamp mapping:\n"
            f"{frame_lines}"
        )

    @staticmethod
    def _parse_response(response: str, window: VideoWindow) -> dict[str, Any]:
        match = _JSON_BLOCK_RE.search(response)
        if match is None:
            raise ValueError("Qwen temporal verifier did not return a JSON object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen temporal verifier returned invalid JSON") from exc
        supported = bool(payload.get("supported", False))
        raw_index = payload.get("selected_frame_index")
        selected_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else None
        if selected_index is None or not 0 <= selected_index < len(window.frames):
            supported = False
            selected_timestamp_ms = None
        else:
            selected_timestamp_ms = window.frames[selected_index].timestamp_ms
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        state = str(payload.get("selected_state", "uncertain")).strip().casefold()
        return {
            "supported": supported,
            "timestamp_ms": selected_timestamp_ms,
            "confidence": max(0.0, min(1.0, confidence)),
            "selected_state": state if state in _VALID_STATES else "uncertain",
            "reason": " ".join(str(payload.get("reason", "")).split())[:400],
        }
