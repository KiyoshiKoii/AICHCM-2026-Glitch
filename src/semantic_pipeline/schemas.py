"""Versioned metadata schema for the Dev 2 semantic pipeline.

The schema accepts the current Task 1 records unchanged: fields introduced for
Task 4 have safe defaults.  Elasticsearch mappings and future extractors should
use ``FrameMetadata`` as their shared source of truth.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "sample_frames" / "metadata.json"

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]


class TimeOfDay(str, Enum):
    UNKNOWN = "unknown"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class SceneSetting(str, Enum):
    UNKNOWN = "unknown"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class SpatialPredicate(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    OVERLAPPING = "overlapping"


class StrictModel(BaseModel):
    """Reject accidental fields instead of silently dropping misspelled data."""

    model_config = ConfigDict(extra="forbid")


class FrameEntities(StrictModel):
    time_of_day: TimeOfDay = TimeOfDay.UNKNOWN
    setting: SceneSetting = SceneSetting.UNKNOWN
    locations: list[NonEmptyString] = Field(default_factory=list)
    objects: list[NonEmptyString] = Field(default_factory=list)
    actions: list[NonEmptyString] = Field(default_factory=list)
    colors: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("locations", "objects", "actions", "colors")
    @classmethod
    def normalise_entity_values(cls, values: list[str]) -> list[str]:
        """Canonicalise LLM output so harmless casing/duplicates do not fail a batch."""
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalised = value.strip().casefold()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(normalised)
        return unique


class Detection(StrictModel):
    object_id: NonEmptyString
    label: NonEmptyString
    # ``raw_label`` preserves the detector output when contextual evidence
    # corrects a known visual-label confusion.  Old metadata remains valid:
    # detector-only records use the defaults below.
    raw_label: NonEmptyString | None = None
    label_source: Literal["detector", "context_grounding"] = "detector"
    label_evidence: list[NonEmptyString] = Field(default_factory=list)
    # Normalised [x1, y1, x2, y2], measured from the image's top-left corner.
    bbox: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    confidence: Confidence

    @field_validator("label_evidence")
    @classmethod
    def normalise_label_evidence(cls, values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalised = value.strip().casefold()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(normalised)
        return unique

    @model_validator(mode="after")
    def validate_box_order(self) -> "Detection":
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        if self.label_source == "detector" and (
            self.raw_label is not None or self.label_evidence
        ):
            raise ValueError(
                "detector labels cannot declare grounding provenance"
            )
        if self.label_source == "context_grounding" and (
            self.raw_label is None
            or self.raw_label.casefold() == self.label.casefold()
            or not self.label_evidence
        ):
            raise ValueError(
                "context-grounded labels require a different raw_label and evidence"
            )
        return self


class SpatialRelation(StrictModel):
    subject_id: NonEmptyString
    subject_label: NonEmptyString | None = None
    predicate: SpatialPredicate
    object_id: NonEmptyString
    object_label: NonEmptyString | None = None
    confidence: Confidence

    @model_validator(mode="after")
    def reject_self_relation(self) -> "SpatialRelation":
        if self.subject_id == self.object_id:
            raise ValueError("a spatial relation cannot reference the same object twice")
        if (self.subject_label is None) != (self.object_label is None):
            raise ValueError("spatial relation labels must be both present or both absent")
        return self


class ProcessingMetadata(StrictModel):
    caption_model: str | None = None
    ocr_models: list[NonEmptyString] = Field(default_factory=list)
    entity_model: str | None = None
    prompt_version: str | None = None
    processed_at: datetime | None = None
    detection_model: str | None = None
    label_grounding_version: str | None = None
    spatial_rule_version: str | None = None
    spatial_processed_at: datetime | None = None
    spatial_index_policy: str | None = None
    spatial_relations_raw_count: Annotated[int, Field(ge=0)] | None = None
    spatial_relations_indexed_count: Annotated[int, Field(ge=0)] | None = None


class FrameMetadata(StrictModel):
    """Canonical persisted document for one video frame."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    frame_id: NonEmptyString
    video_name: NonEmptyString
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_ms: Annotated[int, Field(ge=0)] | None = None

    caption: str
    ocr_text: str = ""
    ocr_text_raw: str = ""

    entities: FrameEntities = Field(default_factory=FrameEntities)
    detections: list[Detection] = Field(default_factory=list)
    spatial_relations: list[SpatialRelation] = Field(default_factory=list)
    processing: ProcessingMetadata = Field(default_factory=ProcessingMetadata)

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> "FrameMetadata":
        expected_suffix = f"_f{self.frame_index:04d}"
        if not self.frame_id.endswith(expected_suffix):
            raise ValueError(
                f"frame_id must end with {expected_suffix!r} for frame_index={self.frame_index}"
            )

        video_stem = Path(self.video_name).stem
        expected_frame_id = f"{video_stem}{expected_suffix}"
        if self.frame_id != expected_frame_id:
            raise ValueError(
                f"frame_id {self.frame_id!r} does not match video_name; "
                f"expected {expected_frame_id!r}"
            )

        detection_ids = [detection.object_id for detection in self.detections]
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("detection object_id values must be unique within a frame")

        known_ids = set(detection_ids)
        labels_by_id = {
            detection.object_id: detection.label for detection in self.detections
        }
        for relation in self.spatial_relations:
            missing = {relation.subject_id, relation.object_id} - known_ids
            if missing:
                raise ValueError(
                    "spatial relation references unknown detection ids: "
                    f"{sorted(missing)}"
                )
            if relation.subject_label is not None and (
                relation.subject_label.casefold()
                != labels_by_id[relation.subject_id].casefold()
                or relation.object_label.casefold()
                != labels_by_id[relation.object_id].casefold()
            ):
                raise ValueError("spatial relation labels do not match detection ids")
        return self


def load_metadata_file(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> list[dict]:
    path = Path(metadata_path)
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError(f"{path} must contain a JSON array")
    if not raw_records:
        raise ValueError(f"{path} contains no frame records")
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError(f"every item in {path} must be a JSON object")
    return raw_records


def validate_metadata_records(raw_records: list[dict]) -> list[FrameMetadata]:
    """Validate records and report the failing frame/position in one error."""
    validated: list[FrameMetadata] = []
    seen_frame_ids: set[str] = set()
    for position, raw_record in enumerate(raw_records):
        label = raw_record.get("frame_id", f"position {position}")
        try:
            record = FrameMetadata.model_validate(raw_record)
        except ValidationError as exc:
            raise ValueError(f"Invalid metadata record {label!r}:\n{exc}") from exc
        if record.frame_id in seen_frame_ids:
            raise ValueError(f"Duplicate frame_id in metadata: {record.frame_id}")
        seen_frame_ids.add(record.frame_id)
        validated.append(record)
    return validated


def validate_metadata_file(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> list[FrameMetadata]:
    return validate_metadata_records(load_metadata_file(metadata_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate metadata against schema v1.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    raw_records = load_metadata_file(args.metadata)
    legacy_count = sum("schema_version" not in record for record in raw_records)
    try:
        records = validate_metadata_records(raw_records)
    except ValueError as exc:
        parser.exit(status=1, message=f"Validation failed: {exc}\n")
    print(
        f"Validated {len(records)} records against metadata schema v{SCHEMA_VERSION}. "
        f"Legacy records using defaults: {legacy_count}."
    )


if __name__ == "__main__":
    main()
