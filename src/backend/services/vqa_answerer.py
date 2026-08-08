import logging
from pathlib import Path
from typing import Protocol

from backend.schemas.search import SearchHit
from backend.schemas.vqa import VQAAnswer, VQABatchAnswer

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


logger = logging.getLogger("uvicorn.error")


class VQAAnswerer(Protocol):
    @property
    def available(self) -> bool: ...

    async def answer_batch(
        self,
        question: str,
        hits: list[SearchHit],
    ) -> dict[str, VQAAnswer]: ...


class GeminiVQAAnswerer:
    """Answers one question for a labelled batch of retrieved keyframes."""

    def __init__(self, api_key: str | None, model_name: str) -> None:
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key and genai else None
        if self.client:
            logger.info("[GeminiVQAAnswerer] Initialized with model %s", model_name)
        else:
            logger.warning("[GeminiVQAAnswerer] Not initialized. Missing API key or google-genai.")

    @property
    def available(self) -> bool:
        return self.client is not None

    @staticmethod
    def _image_path(frame_id: str) -> Path | None:
        if "_f" not in frame_id:
            return None

        video_name, frame_part = frame_id.rsplit("_f", 1)
        try:
            frame_number = int(frame_part)
        except ValueError:
            return None

        project_root = Path(__file__).resolve().parents[3]
        return project_root / "data" / "keyframes" / video_name / f"{frame_number:03d}.jpg"

    async def answer_batch(
        self,
        question: str,
        hits: list[SearchHit],
    ) -> dict[str, VQAAnswer]:
        if not self.client or not hits:
            return {}

        contents: list = []
        valid_frame_ids: set[str] = set()
        for hit in hits:
            image_path = self._image_path(hit.frame_id)
            if image_path is None or not image_path.is_file():
                logger.warning("[GeminiVQAAnswerer] Image not found for %s", hit.frame_id)
                continue
            contents.extend([
                f"Frame ID: {hit.frame_id}",
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"),
            ])
            valid_frame_ids.add(hit.frame_id)

        if not valid_frame_ids:
            return {}

        prompt = (
            "You will receive labelled images. Answer the same question independently "
            "for every image, using only visual information in that image. Do not combine "
            "evidence across images. Return concise Vietnamese or English answers with no "
            "explanation. If an image does not provide enough evidence, answer exactly "
            "'unknown'. Each returned frame_id must exactly match one supplied Frame ID.\n\n"
            f"Question: {question}"
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[prompt, *contents],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VQABatchAnswer,
                    temperature=0.0,
                ),
            )
            batch = VQABatchAnswer.model_validate_json(response.text)
            answers: dict[str, VQAAnswer] = {}
            for item in batch.answers:
                if item.frame_id in valid_frame_ids and item.frame_id not in answers:
                    answers[item.frame_id] = VQAAnswer(
                        answer=item.answer,
                        confidence=item.confidence,
                    )
            return answers
        except Exception as exc:
            logger.error("[GeminiVQAAnswerer] Batch request failed: %s", exc)
            return {}
