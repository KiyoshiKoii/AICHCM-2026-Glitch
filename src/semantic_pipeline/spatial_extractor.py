"""Florence-2 object detection and deterministic spatial reasoning.

The heavy vision model is isolated from the Task 1 extractor so this stage does
not load PaddleOCR, VietOCR, or the translation model. Florence returns pixel
boxes; this module validates and normalises them before deriving relations.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PIL import Image

try:
    from migrate_metadata import write_json_atomically
    from schemas import (
        DEFAULT_METADATA_PATH,
        Detection,
        FrameMetadata,
        SpatialRelation,
        load_metadata_file,
        validate_metadata_records,
    )
except ImportError:
    from .migrate_metadata import write_json_atomically
    from .schemas import (
        DEFAULT_METADATA_PATH,
        Detection,
        FrameMetadata,
        SpatialRelation,
        load_metadata_file,
        validate_metadata_records,
    )

FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE_OD_TASK = "<OD>"
SPATIAL_RULE_VERSION = "spatial-v2"
LABEL_GROUNDING_VERSION = "label-grounding-v1"
DEFAULT_INPUT_PATH = DEFAULT_METADATA_PATH.with_name("metadata_entities.json")
DEFAULT_OUTPUT_PATH = DEFAULT_METADATA_PATH.with_name("metadata_spatial.json")
DEFAULT_IMAGE_DIR = DEFAULT_METADATA_PATH.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Florence's normal OD decode does not expose calibrated per-box probabilities.
# Keep this explicit proxy below 1.0 rather than presenting certainty as a score.
DEFAULT_DETECTION_CONFIDENCE = 0.5


@dataclass(frozen=True)
class LabelGroundingRule:
    """A detector confusion that may be corrected only with frame evidence."""

    detector_labels: frozenset[str]
    target_label: str
    evidence_terms: tuple[str, ...]


# These are model-level confusions observed across content types, not frame IDs.
# A rule is inert unless its target concept occurs in entities/caption/OCR.
LABEL_GROUNDING_RULES = (
    LabelGroundingRule(
        frozenset({"dice"}),
        "domino",
        ("domino", "dominos", "dominoes"),
    ),
    LabelGroundingRule(
        frozenset({"vase"}),
        "sack",
        ("sack", "sacks", "money bag", "money bags"),
    ),
    LabelGroundingRule(
        frozenset({"christmas tree", "tower"}),
        "triangle",
        ("triangle", "triangles"),
    ),
    LabelGroundingRule(
        frozenset({"toy"}),
        "grid",
        ("grid", "table cells"),
    ),
)

# Florence often names a full-frame document container after its visual shape.
# For boxes covering nearly the whole image, prefer an entity extracted from the
# same frame.  Priority is deliberately narrow and deterministic.
FULL_FRAME_CONTAINER_LABELS = frozenset(
    {"poster", "whiteboard", "book", "picture frame"}
)
FULL_FRAME_CONTEXT_LABELS = (
    ("screenshot", ("screenshot",)),
    ("worksheet", ("worksheet",)),
    ("website", ("website", "webpage")),
    ("text editor", ("text editor",)),
    ("computer screen", ("computer screen",)),
    ("notebook", ("notebook",)),
    ("page", ("page",)),
)


@dataclass(frozen=True)
class RawDetection:
    label: str
    # Pixel coordinates [x1, y1, x2, y2] in the original image.
    bbox: tuple[float, float, float, float]
    confidence: float | None = None


class ObjectDetector(Protocol):
    model: str

    def detect(self, image: Image.Image) -> list[RawDetection]: ...


class FlorenceObjectDetector:
    """Minimal Florence-only `<OD>` adapter."""

    def __init__(
        self,
        model_id: str = FLORENCE_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.torch = torch
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self.device == "auto":
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = model_id
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.florence = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.florence.eval()

    def detect(self, image: Image.Image) -> list[RawDetection]:
        image = image.convert("RGB")
        inputs = self.processor(
            text=FLORENCE_OD_TASK,
            images=image,
            return_tensors="pt",
        ).to(self.device, self.dtype)
        with self.torch.inference_mode():
            generated_ids = self.florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=FLORENCE_OD_TASK,
            image_size=image.size,  # PIL order is (width, height).
        )
        result = parsed.get(FLORENCE_OD_TASK)
        if not isinstance(result, dict):
            raise RuntimeError("Florence response is missing the <OD> result")
        boxes = result.get("bboxes", [])
        labels = result.get("labels", [])
        if len(boxes) != len(labels):
            raise RuntimeError(
                "Florence returned different numbers of labels and bounding boxes"
            )

        raw: list[RawDetection] = []
        for label, box in zip(labels, boxes):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            raw.append(
                RawDetection(
                    label=str(label),
                    bbox=tuple(float(coordinate) for coordinate in box),
                )
            )
        return raw


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _clean_label(label: str) -> str:
    return " ".join(label.strip().casefold().split())


def _object_id_base(label: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return base or "object"


def normalise_detections(
    raw_detections: list[RawDetection],
    image_width: int,
    image_height: int,
    default_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    min_box_area: float = 0.0001,
    duplicate_iou: float = 0.95,
    max_detections: int = 50,
) -> list[Detection]:
    """Clamp pixel boxes, drop noise/duplicates, and create deterministic IDs."""
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not 0 <= default_confidence <= 1:
        raise ValueError("default_confidence must be between 0 and 1")
    if not 0 <= min_box_area < 1:
        raise ValueError("min_box_area must be in [0, 1)")
    if not 0 <= duplicate_iou <= 1:
        raise ValueError("duplicate_iou must be between 0 and 1")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")

    candidates: list[tuple[str, tuple[float, float, float, float], float]] = []
    for raw in raw_detections:
        label = _clean_label(raw.label)
        if not label or len(raw.bbox) != 4 or not all(
            math.isfinite(value) for value in raw.bbox
        ):
            continue
        px1, py1, px2, py2 = raw.bbox
        x1 = min(1.0, max(0.0, px1 / image_width))
        y1 = min(1.0, max(0.0, py1 / image_height))
        x2 = min(1.0, max(0.0, px2 / image_width))
        y2 = min(1.0, max(0.0, py2 / image_height))
        if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) < min_box_area:
            continue
        confidence = (
            default_confidence if raw.confidence is None else raw.confidence
        )
        if not math.isfinite(confidence):
            continue
        confidence = min(1.0, max(0.0, confidence))
        box = (x1, y1, x2, y2)
        if any(
            previous_label == label and bbox_iou(previous_box, box) >= duplicate_iou
            for previous_label, previous_box, _ in candidates
        ):
            continue
        candidates.append((label, box, confidence))

    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1], item[1][2]))
    counts: defaultdict[str, int] = defaultdict(int)
    detections: list[Detection] = []
    for label, box, confidence in candidates[:max_detections]:
        base = _object_id_base(label)
        occurrence = counts[base]
        counts[base] += 1
        detections.append(
            Detection(
                object_id=f"{base}_{occurrence}",
                label=label,
                bbox=box,
                confidence=confidence,
            )
        )
    return detections


def _contains_evidence(text: str, term: str) -> bool:
    """Match a phrase on token boundaries without fuzzy or semantic guessing."""
    return re.search(
        rf"(?<!\w){re.escape(term.casefold())}(?!\w)", text.casefold()
    ) is not None


def _grounding_evidence(
    record: FrameMetadata,
    terms: tuple[str, ...],
    *,
    entities_only: bool = False,
) -> list[str]:
    evidence: list[str] = []
    for entity in record.entities.objects:
        if any(_contains_evidence(entity, term) for term in terms):
            evidence.append(f"entities.objects:{entity}")
    if not entities_only:
        for source_name, source_text in (
            ("caption", record.caption),
            ("ocr_text", record.ocr_text),
        ):
            for term in terms:
                if _contains_evidence(source_text, term):
                    evidence.append(f"{source_name}:{term}")
                    break
    # Keep persisted provenance compact and deterministic.
    return list(dict.fromkeys(evidence))[:3]


def _grounded_label(
    detection: Detection,
    record: FrameMetadata,
) -> tuple[str, list[str]]:
    detector_label = detection.raw_label or detection.label
    for rule in LABEL_GROUNDING_RULES:
        if detector_label in rule.detector_labels:
            evidence = _grounding_evidence(record, rule.evidence_terms)
            if evidence:
                return rule.target_label, evidence

    x1, y1, x2, y2 = detection.bbox
    box_area = (x2 - x1) * (y2 - y1)
    if detector_label in FULL_FRAME_CONTAINER_LABELS and box_area >= 0.8:
        for target_label, terms in FULL_FRAME_CONTEXT_LABELS:
            evidence = _grounding_evidence(record, terms, entities_only=True)
            if evidence and target_label != detector_label:
                return target_label, evidence
    return detector_label, []


def ground_detection_labels(
    detections: list[Detection],
    record: FrameMetadata,
) -> tuple[list[Detection], int]:
    """Correct explicit detector confusions using independent frame evidence.

    The original detector label and exact evidence source are persisted.  The
    function is idempotent: re-running always starts from ``raw_label``.
    """
    grounded: list[Detection] = []
    correction_count = 0
    for detection in detections:
        detector_label = detection.raw_label or detection.label
        target_label, evidence = _grounded_label(detection, record)
        if target_label == detector_label:
            grounded.append(
                detection.model_copy(
                    update={
                        "label": detector_label,
                        "raw_label": None,
                        "label_source": "detector",
                        "label_evidence": [],
                    }
                )
            )
            continue
        correction_count += 1
        grounded.append(
            detection.model_copy(
                update={
                    "label": target_label,
                    "raw_label": detector_label,
                    "label_source": "context_grounding",
                    "label_evidence": evidence,
                }
            )
        )
    return grounded, correction_count


def _axis_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    intersection = max(
        0.0, min(first_end, second_end) - max(first_start, second_start)
    )
    smaller = min(first_end - first_start, second_end - second_start)
    return intersection / smaller if smaller > 0 else 0.0


def _overlap_over_smaller_box(first: Detection, second: Detection) -> float:
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _relation(
    subject: Detection,
    predicate: str,
    object_: Detection,
    confidence: float,
) -> SpatialRelation:
    return SpatialRelation.model_validate(
        {
            "subject_id": subject.object_id,
            "subject_label": subject.label,
            "predicate": predicate,
            "object_id": object_.object_id,
            "object_label": object_.label,
            "confidence": round(min(1.0, max(0.0, confidence)), 6),
        }
    )


def infer_spatial_relations(
    detections: list[Detection],
    min_axis_gap: float = 0.01,
    min_axis_overlap: float = 0.2,
    min_overlap_ratio: float = 0.2,
) -> list[SpatialRelation]:
    """Derive conservative reciprocal relations from normalised boxes."""
    if not 0 <= min_axis_gap < 1:
        raise ValueError("min_axis_gap must be in [0, 1)")
    if not 0 <= min_axis_overlap <= 1:
        raise ValueError("min_axis_overlap must be between 0 and 1")
    if not 0 <= min_overlap_ratio <= 1:
        raise ValueError("min_overlap_ratio must be between 0 and 1")

    relations: list[SpatialRelation] = []
    for first_index, first in enumerate(detections):
        for second in detections[first_index + 1 :]:
            ax1, ay1, ax2, ay2 = first.bbox
            bx1, by1, bx2, by2 = second.bbox
            base_confidence = min(first.confidence, second.confidence)

            vertical_projection = _axis_overlap(ay1, ay2, by1, by2)
            if vertical_projection >= min_axis_overlap:
                if bx1 - ax2 >= min_axis_gap:
                    strength = 0.5 + 0.25 * min((bx1 - ax2) / 0.25, 1.0) + 0.25 * vertical_projection
                    relations.extend(
                        [
                            _relation(first, "left_of", second, base_confidence * strength),
                            _relation(second, "right_of", first, base_confidence * strength),
                        ]
                    )
                elif ax1 - bx2 >= min_axis_gap:
                    strength = 0.5 + 0.25 * min((ax1 - bx2) / 0.25, 1.0) + 0.25 * vertical_projection
                    relations.extend(
                        [
                            _relation(second, "left_of", first, base_confidence * strength),
                            _relation(first, "right_of", second, base_confidence * strength),
                        ]
                    )

            horizontal_projection = _axis_overlap(ax1, ax2, bx1, bx2)
            if horizontal_projection >= min_axis_overlap:
                if by1 - ay2 >= min_axis_gap:
                    strength = 0.5 + 0.25 * min((by1 - ay2) / 0.25, 1.0) + 0.25 * horizontal_projection
                    relations.extend(
                        [
                            _relation(first, "above", second, base_confidence * strength),
                            _relation(second, "below", first, base_confidence * strength),
                        ]
                    )
                elif ay1 - by2 >= min_axis_gap:
                    strength = 0.5 + 0.25 * min((ay1 - by2) / 0.25, 1.0) + 0.25 * horizontal_projection
                    relations.extend(
                        [
                            _relation(second, "above", first, base_confidence * strength),
                            _relation(first, "below", second, base_confidence * strength),
                        ]
                    )

            overlap_ratio = _overlap_over_smaller_box(first, second)
            if overlap_ratio >= min_overlap_ratio:
                overlap_confidence = base_confidence * overlap_ratio
                relations.extend(
                    [
                        _relation(first, "overlapping", second, overlap_confidence),
                        _relation(second, "overlapping", first, overlap_confidence),
                    ]
                )
    return relations


def _image_index(image_dir: str | Path) -> dict[str, Path]:
    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    index: dict[str, Path] = {}
    for path in image_dir.rglob("*.*"):
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            stem = path.stem
            if "_f" in stem:
                frame_id = stem
            else:
                video_name = path.parent.name
                frame_id = f"{video_name}_f{int(stem):04d}"
            if frame_id in index:
                raise ValueError(
                    f"Multiple images have frame_id {frame_id!r}: "
                    f"{index[frame_id]} and {path}"
                )
            index[frame_id] = path
    return index


def preflight_inputs(
    metadata_path: str | Path,
    image_dir: str | Path,
    frame_ids: set[str] | None = None,
) -> tuple[list[FrameMetadata], dict[str, Path]]:
    records = validate_metadata_records(load_metadata_file(metadata_path))
    known_ids = {record.frame_id for record in records}
    unknown = (frame_ids or set()) - known_ids
    if unknown:
        raise ValueError(f"Unknown --frame-id values: {sorted(unknown)}")
    images = _image_index(image_dir)
    required = frame_ids or known_ids
    missing = sorted(required - images.keys())
    if missing:
        raise FileNotFoundError(f"Missing images for frame_ids: {missing}")
    return records, images


def enrich_spatial_records(
    records: list[FrameMetadata],
    detector: ObjectDetector,
    image_loader: Callable[[FrameMetadata], Image.Image],
    frame_ids: set[str] | None = None,
    limit: int | None = None,
    overwrite_existing: bool = False,
    default_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    min_box_area: float = 0.0001,
    duplicate_iou: float = 0.95,
    max_detections: int = 50,
    min_axis_gap: float = 0.01,
    min_axis_overlap: float = 0.2,
    min_overlap_ratio: float = 0.2,
    on_progress: Callable[[int, list[FrameMetadata]], None] | None = None,
) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    known_ids = {record.frame_id for record in records}
    unknown = (frame_ids or set()) - known_ids
    if unknown:
        raise ValueError(f"Unknown frame_ids: {sorted(unknown)}")

    processed = skipped = eligible = total_detections = total_relations = 0
    total_grounded_labels = 0
    for index, record in enumerate(records):
        if frame_ids is not None and record.frame_id not in frame_ids:
            continue
        if record.processing.detection_model is not None and not overwrite_existing:
            skipped += 1
            continue
        if limit is not None and eligible >= limit:
            continue
        eligible += 1

        image = image_loader(record)
        try:
            raw = detector.detect(image)
            detections = normalise_detections(
                raw,
                image.width,
                image.height,
                default_confidence=default_confidence,
                min_box_area=min_box_area,
                duplicate_iou=duplicate_iou,
                max_detections=max_detections,
            )
        finally:
            image.close()
        detections, grounded_labels = ground_detection_labels(detections, record)
        relations = infer_spatial_relations(
            detections,
            min_axis_gap=min_axis_gap,
            min_axis_overlap=min_axis_overlap,
            min_overlap_ratio=min_overlap_ratio,
        )
        processing = record.processing.model_copy(
            update={
                "detection_model": detector.model,
                "label_grounding_version": LABEL_GROUNDING_VERSION,
                "spatial_rule_version": SPATIAL_RULE_VERSION,
                "spatial_processed_at": datetime.now(timezone.utc),
            }
        )
        records[index] = record.model_copy(
            update={
                "detections": detections,
                "spatial_relations": relations,
                "processing": processing,
            }
        )
        processed += 1
        total_detections += len(detections)
        total_relations += len(relations)
        total_grounded_labels += grounded_labels
        if on_progress is not None:
            on_progress(processed, records)

    return {
        "processed": processed,
        "skipped": skipped,
        "total": len(records),
        "detections": total_detections,
        "relations": total_relations,
        "grounded_labels": total_grounded_labels,
    }


def reground_spatial_records(
    records: list[FrameMetadata],
    min_axis_gap: float = 0.01,
    min_axis_overlap: float = 0.2,
    min_overlap_ratio: float = 0.2,
) -> dict:
    """Apply current label grounding/rules to existing detector output."""
    processed = total_detections = total_relations = total_grounded_labels = 0
    for index, record in enumerate(records):
        if not record.detections:
            continue
        detections, grounded_labels = ground_detection_labels(
            record.detections, record
        )
        relations = infer_spatial_relations(
            detections,
            min_axis_gap=min_axis_gap,
            min_axis_overlap=min_axis_overlap,
            min_overlap_ratio=min_overlap_ratio,
        )
        processing = record.processing.model_copy(
            update={
                "label_grounding_version": LABEL_GROUNDING_VERSION,
                "spatial_rule_version": SPATIAL_RULE_VERSION,
                "spatial_processed_at": datetime.now(timezone.utc),
            }
        )
        records[index] = record.model_copy(
            update={
                "detections": detections,
                "spatial_relations": relations,
                "processing": processing,
            }
        )
        processed += 1
        total_detections += len(detections)
        total_relations += len(relations)
        total_grounded_labels += grounded_labels
    return {
        "processed": processed,
        "total": len(records),
        "detections": total_detections,
        "relations": total_relations,
        "grounded_labels": total_grounded_labels,
    }


def _dump_records(records: list[FrameMetadata]) -> list[dict]:
    return [record.model_dump(mode="json") for record in records]


def reground_spatial_file(
    input_path: str | Path,
    output_path: str | Path,
    force: bool = False,
    **reasoning_options,
) -> dict:
    """Re-ground existing detections without loading Florence again.

    In-place migration is safe because ``write_json_atomically`` writes a
    validated complete replacement; a different existing output still needs
    ``force=True``.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path != input_path and output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} exists; pass --force or choose another output"
        )
    records = validate_metadata_records(load_metadata_file(input_path))
    summary = reground_spatial_records(records, **reasoning_options)
    # Re-validate model-copy updates before persisting the corpus.
    validated = validate_metadata_records(_dump_records(records))
    write_json_atomically(output_path, _dump_records(validated), overwrite=True)
    return summary


def enrich_spatial_file(
    input_path: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
    detector: ObjectDetector,
    frame_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
    overwrite_existing: bool = False,
    checkpoint_every: int = 5,
    **reasoning_options,
) -> dict:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    source_records, images = preflight_inputs(input_path, image_dir, frame_ids)
    if resume:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot resume; output does not exist: {output_path}")
        records = validate_metadata_records(load_metadata_file(output_path))
        if {record.frame_id for record in records} != {
            record.frame_id for record in source_records
        }:
            raise ValueError("resume output frame_ids do not match the input metadata")
    else:
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} exists; pass --resume, --force, or choose another output"
            )
        records = source_records

    def load_image(record: FrameMetadata) -> Image.Image:
        with Image.open(images[record.frame_id]) as source:
            return source.convert("RGB")

    def checkpoint(processed: int, current: list[FrameMetadata]) -> None:
        if processed % checkpoint_every == 0:
            write_json_atomically(output_path, _dump_records(current), overwrite=True)

    try:
        summary = enrich_spatial_records(
            records,
            detector,
            load_image,
            frame_ids=frame_ids,
            limit=limit,
            overwrite_existing=overwrite_existing,
            on_progress=checkpoint,
            **reasoning_options,
        )
    except Exception:
        write_json_atomically(output_path, _dump_records(records), overwrite=True)
        raise
    write_json_atomically(output_path, _dump_records(records), overwrite=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect objects with Florence-2 and derive spatial relations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=os.getenv("FLORENCE_MODEL", FLORENCE_MODEL_ID))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--frame-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--reground-existing",
        action="store_true",
        help=(
            "Re-apply contextual label grounding and spatial rules to existing "
            "detections without loading Florence; --input may equal --output."
        ),
    )
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-box-area", type=float, default=0.0001)
    parser.add_argument("--min-axis-gap", type=float, default=0.01)
    parser.add_argument("--min-axis-overlap", type=float, default=0.2)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.2)
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=DEFAULT_DETECTION_CONFIDENCE,
        help="Proxy confidence for Florence OD, whose standard decode has no box scores.",
    )
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="Validate metadata/image matching without loading Florence.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    frame_ids = set(args.frame_id) if args.frame_id else None
    try:
        if args.check_input:
            records, images = preflight_inputs(args.input, args.image_dir, frame_ids)
            selected = frame_ids or {record.frame_id for record in records}
            print(
                f"Spatial input ready: records={len(records)}, "
                f"selected_images={len(selected)}, image_dir={args.image_dir.resolve()}"
            )
            return

        reasoning_options = {
            "min_axis_gap": args.min_axis_gap,
            "min_axis_overlap": args.min_axis_overlap,
            "min_overlap_ratio": args.min_overlap_ratio,
        }
        if args.reground_existing:
            summary = reground_spatial_file(
                args.input,
                args.output,
                force=args.force,
                **reasoning_options,
            )
            print(
                "Spatial re-grounding complete: "
                f"processed={summary['processed']}, "
                f"grounded_labels={summary['grounded_labels']}, "
                f"detections={summary['detections']}, "
                f"relations={summary['relations']}, "
                f"total={summary['total']}, output={args.output.resolve()}"
            )
            return

        detector = FlorenceObjectDetector(model_id=args.model, device=args.device)
        summary = enrich_spatial_file(
            input_path=args.input,
            image_dir=args.image_dir,
            output_path=args.output,
            detector=detector,
            frame_ids=frame_ids,
            limit=args.limit,
            resume=args.resume,
            force=args.force,
            overwrite_existing=args.overwrite_existing,
            checkpoint_every=args.checkpoint_every,
            default_confidence=args.detection_confidence,
            min_box_area=args.min_box_area,
            max_detections=args.max_detections,
            **reasoning_options,
        )
        print(
            "Spatial extraction complete: "
            f"processed={summary['processed']}, skipped={summary['skipped']}, "
            f"grounded_labels={summary['grounded_labels']}, "
            f"detections={summary['detections']}, relations={summary['relations']}, "
            f"total={summary['total']}, output={args.output.resolve()}"
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Spatial extraction failed: {exc}\n")


if __name__ == "__main__":
    main()
