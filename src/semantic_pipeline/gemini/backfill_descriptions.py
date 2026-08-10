"""Backfill missing object descriptions from existing Gemini captions.

This tool intentionally uses only the already stored caption and detailed caption as evidence.
It never invents visual details that are absent from those fields and leaves every unrelated
metadata field untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field, field_validator

from ..core.json_io import write_json_atomically
from .extractor import DEFAULT_MODEL, load_gemini_api_key, load_gemini_visual_model

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - exercised only without the optional SDK.
    genai = None
    types = None


BACKFILL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "frames": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "frame_id": {"type": "STRING"},
                    "objects": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "object_id": {"type": "STRING"},
                                "description": {"type": "STRING"},
                                "description_vi": {"type": "STRING"},
                                "attributes": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "action": {"type": "STRING"},
                            },
                            "required": [
                                "object_id",
                                "description",
                                "description_vi",
                                "attributes",
                                "action",
                            ],
                        },
                    },
                },
                "required": ["frame_id", "objects"],
            },
        }
    },
    "required": ["frames"],
}


class DescriptionUpdate(BaseModel):
    object_id: str
    description: str = ""
    description_vi: str = ""
    attributes: list[str] = Field(default_factory=list)
    action: str = ""

    @field_validator("object_id", "description", "description_vi", "action", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return " ".join(value.split()) if isinstance(value, str) else ""

    @field_validator("attributes", mode="before")
    @classmethod
    def clean_attributes(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = " ".join(item.split()).casefold()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result[:6]


class DescriptionFrame(BaseModel):
    frame_id: str
    objects: list[DescriptionUpdate] = Field(default_factory=list)


class DescriptionBatch(BaseModel):
    frames: list[DescriptionFrame]


PROMPT = """You repair missing object enrichment in existing video metadata.

Use ONLY facts explicitly present in each frame's caption or detailed_caption. Do not inspect
the image, infer an attribute from an object label, or add an occupation, identity, relationship,
or action that the captions do not state.

For every listed object, return one update with the same object_id. If the captions explicitly
tie a color, clothing detail, appearance, or action to that object, copy it into description,
description_vi, attributes, or action as appropriate. Keep fields empty when the captions do not
support a detail. Return one frame entry for every input frame and JSON only.
"""


def _caption_context(record: dict[str, Any]) -> dict[str, Any]:
    missing_objects = []
    for detection in record.get("detections", []):
        if not isinstance(detection, dict):
            continue
        if not (
            detection.get("description")
            and detection.get("description_vi")
            and detection.get("attributes")
            and detection.get("action")
        ):
            missing_objects.append(
                {
                    "object_id": detection.get("object_id", ""),
                    "label": detection.get("label", ""),
                }
            )
    return {
        "frame_id": record.get("frame_id", ""),
        "caption": record.get("caption", ""),
        "detailed_caption": record.get("detailed_caption", ""),
        "detailed_caption_vi": record.get("detailed_caption_vi", ""),
        "objects": missing_objects,
    }


def _needs_backfill(record: dict[str, Any]) -> bool:
    return bool(_caption_context(record)["objects"])


def build_prompt(records: Sequence[dict[str, Any]]) -> str:
    return PROMPT + "\n\nINPUT FRAMES:\n" + json.dumps(
        [_caption_context(record) for record in records],
        ensure_ascii=False,
        indent=2,
    )


def apply_updates(records: list[dict[str, Any]], payload: dict[str, Any]) -> int:
    parsed = DescriptionBatch.model_validate(payload)
    by_frame = {str(item.get("frame_id")): item for item in records}
    updated = 0
    for frame in parsed.frames:
        record = by_frame.get(frame.frame_id)
        if record is None:
            continue
        detections = record.get("detections")
        if not isinstance(detections, list):
            continue
        updates = {item.object_id: item for item in frame.objects}
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            update = updates.get(detection.get("object_id"))
            if update is None:
                continue
            changed = False
            for field in ("description", "description_vi", "attributes", "action"):
                current = detection.get(field)
                value = getattr(update, field)
                if not current and value:
                    detection[field] = value
                    changed = True
            if changed:
                updated += 1
    return updated


def _load_video(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"metadata must be a JSON array: {path}")
    return raw


def run_backfill(
    *,
    metadata_path: str | Path,
    api_key: str | None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 25,
    frame_ids: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    path = Path(metadata_path)
    records = _load_video(path)
    candidates = [
        record
        for record in records
        if _needs_backfill(record)
        and (frame_ids is None or record.get("frame_id") in frame_ids)
    ]
    summary = {"candidates": len(candidates), "updated_objects": 0, "requests": 0}
    if dry_run or not candidates:
        return summary
    if not api_key or genai is None or types is None:
        raise RuntimeError("Gemini SDK/API key is unavailable")

    client = genai.Client(api_key=api_key)
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        response = client.models.generate_content(
            model=model_name,
            contents=build_prompt(batch),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BACKFILL_SCHEMA,
            ),
        )
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("Gemini returned an empty description backfill response")
        summary["updated_objects"] += apply_updates(records, json.loads(response_text))
        summary["requests"] += 1
        write_json_atomically(path, records, overwrite=True)
        if offset + batch_size < len(candidates):
            time.sleep(60 / 15)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", default=load_gemini_visual_model())
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--frame-id", action="append", dest="frame_ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = run_backfill(
        metadata_path=args.metadata,
        api_key=load_gemini_api_key(),
        model_name=args.model,
        batch_size=args.batch_size,
        frame_ids=set(args.frame_ids) if args.frame_ids else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
