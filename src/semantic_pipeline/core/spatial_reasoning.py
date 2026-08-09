"""Local normalization and geometry for Gemini-produced object boxes.

This module intentionally contains no detector model or legacy metadata schema.
Gemini supplies pixel evidence; these pure functions normalize it and derive
conservative relations for the compact visual artifact.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum


class SpatialPredicate(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    OVERLAPPING = "overlapping"


@dataclass(frozen=True)
class RawDetection:
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    object_id: str
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 0.5


@dataclass(frozen=True)
class SpatialRelation:
    subject_id: str
    predicate: SpatialPredicate
    object_id: str


def bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _clean_label(label: str) -> str:
    return " ".join(label.strip().casefold().split())


def _object_id_base(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_") or "object"


def normalise_detections(
    raw_detections: list[RawDetection],
    image_width: int,
    image_height: int,
    min_box_area: float = 0.0001,
    duplicate_iou: float = 0.95,
    max_detections: int = 50,
) -> list[Detection]:
    """Clamp boxes, discard malformed/duplicate output, and create stable IDs."""
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not 0 <= min_box_area < 1 or not 0 <= duplicate_iou <= 1 or max_detections < 1:
        raise ValueError("invalid detection normalization configuration")

    candidates: list[tuple[str, tuple[float, float, float, float]]] = []
    for raw in raw_detections:
        label = _clean_label(raw.label)
        if not label or not all(math.isfinite(value) for value in raw.bbox):
            continue
        px1, py1, px2, py2 = raw.bbox
        box = (
            min(1.0, max(0.0, px1 / image_width)),
            min(1.0, max(0.0, py1 / image_height)),
            min(1.0, max(0.0, px2 / image_width)),
            min(1.0, max(0.0, py2 / image_height)),
        )
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) < min_box_area:
            continue
        if any(previous_label == label and bbox_iou(previous_box, box) >= duplicate_iou for previous_label, previous_box in candidates):
            continue
        candidates.append((label, box))

    candidates = sorted(candidates[:max_detections], key=lambda item: (item[0], *item[1]))
    counts: defaultdict[str, int] = defaultdict(int)
    detections: list[Detection] = []
    for label, box in candidates:
        base = _object_id_base(label)
        detections.append(Detection(f"{base}_{counts[base]}", label, box))
        counts[base] += 1
    return detections


def _axis_overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    smaller = min(first_end - first_start, second_end - second_start)
    return intersection / smaller if smaller > 0 else 0.0


def _overlap_over_smaller_box(first: Detection, second: Detection) -> float:
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller if smaller > 0 else 0.0


def _relation(subject: Detection, predicate: SpatialPredicate, object_: Detection) -> SpatialRelation:
    return SpatialRelation(subject.object_id, predicate, object_.object_id)


def infer_spatial_relations(
    detections: list[Detection],
    min_axis_gap: float = 0.01,
    min_axis_overlap: float = 0.2,
    min_overlap_ratio: float = 0.2,
    max_spatial_box_area: float = 0.95,
) -> list[SpatialRelation]:
    """Derive reciprocal relations while ignoring near-full-frame boxes."""
    if not 0 <= min_axis_gap < 1 or not 0 <= min_axis_overlap <= 1 or not 0 <= min_overlap_ratio <= 1:
        raise ValueError("invalid spatial relation configuration")
    spatial = [
        item for item in detections
        if (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]) <= max_spatial_box_area
    ]
    relations: list[SpatialRelation] = []
    for index, first in enumerate(spatial):
        for second in spatial[index + 1:]:
            ax1, ay1, ax2, ay2 = first.bbox
            bx1, by1, bx2, by2 = second.bbox
            if _axis_overlap(ay1, ay2, by1, by2) >= min_axis_overlap:
                if bx1 - ax2 >= min_axis_gap:
                    relations.extend([_relation(first, SpatialPredicate.LEFT_OF, second), _relation(second, SpatialPredicate.RIGHT_OF, first)])
                elif ax1 - bx2 >= min_axis_gap:
                    relations.extend([_relation(second, SpatialPredicate.LEFT_OF, first), _relation(first, SpatialPredicate.RIGHT_OF, second)])
            if _axis_overlap(ax1, ax2, bx1, bx2) >= min_axis_overlap:
                if by1 - ay2 >= min_axis_gap:
                    relations.extend([_relation(first, SpatialPredicate.ABOVE, second), _relation(second, SpatialPredicate.BELOW, first)])
                elif ay1 - by2 >= min_axis_gap:
                    relations.extend([_relation(second, SpatialPredicate.ABOVE, first), _relation(first, SpatialPredicate.BELOW, second)])
            if _overlap_over_smaller_box(first, second) >= min_overlap_ratio:
                relations.extend([_relation(first, SpatialPredicate.OVERLAPPING, second), _relation(second, SpatialPredicate.OVERLAPPING, first)])
    return relations
