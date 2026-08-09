"""Minimal persisted schema for Gemini visual metadata.

The compact file intentionally stores only fields that cannot be recovered from
``frame_id`` or recomputed at an API boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .frame_id import parse_frame_id


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]


class CompactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompactDetection(CompactModel):
    object_id: NonEmptyString
    label: NonEmptyString
    bbox: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]

    @model_validator(mode="after")
    def validate_box_order(self) -> "CompactDetection":
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self


class CompactSpatialRelation(CompactModel):
    subject_id: NonEmptyString
    predicate: Literal["left_of", "right_of", "above", "below", "overlapping"]
    object_id: NonEmptyString

    @model_validator(mode="after")
    def reject_self_relation(self) -> "CompactSpatialRelation":
        if self.subject_id == self.object_id:
            raise ValueError("a spatial relation cannot reference the same object twice")
        return self


class CompactVisualRecord(CompactModel):
    frame_id: NonEmptyString
    caption: NonEmptyString
    # Kept separate from the short retrieval caption so a caller can display or
    # index richer visual context without making the primary caption noisy.
    detailed_caption: str = ""
    ocr_text: str = ""
    # L21/L22 only: scrolling news crawl at the bottom of the broadcast frame.
    news_ticker_text: str = ""
    detections: list[CompactDetection] = Field(default_factory=list)
    spatial_relations: list[CompactSpatialRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "CompactVisualRecord":
        parse_frame_id(self.frame_id)
        known_ids = {item.object_id for item in self.detections}
        if len(known_ids) != len(self.detections):
            raise ValueError("detection object_id values must be unique within a frame")
        for relation in self.spatial_relations:
            missing = {relation.subject_id, relation.object_id} - known_ids
            if missing:
                raise ValueError(
                    "spatial relation references unknown detection ids: "
                    f"{sorted(missing)}"
                )
        return self
