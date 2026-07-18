"""Tests for the canonical Task 4 metadata schema and legacy migration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from migrate_metadata import migrate_file, migrate_records, write_json_atomically
from schemas import (
    CodeMetadata,
    Detection,
    FrameMetadata,
    SCHEMA_VERSION,
    SpatialRelation,
    validate_metadata_file,
    validate_metadata_records,
)

SEMANTIC_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = SEMANTIC_DIR / "sample_frames" / "metadata.json"


def minimal_record(**updates) -> dict:
    record = {
        "frame_id": "vid01_f0001",
        "video_name": "vid01.mp4",
        "frame_index": 1,
        "caption": "A person stands to the left of a car.",
        "ocr_text": "PARKING",
        "ocr_text_raw": "BÃI ĐỖ XE",
    }
    record.update(updates)
    return record


class TestLegacyCompatibility:
    def test_current_metadata_validates_without_modification(self):
        records = validate_metadata_file(METADATA_PATH)
        assert len(records) == 24
        assert all(record.schema_version == SCHEMA_VERSION for record in records)
        assert all(record.entities.time_of_day.value == "unknown" for record in records)
        assert all(record.code.language == "unknown" for record in records)
        assert all(record.detections == [] for record in records)

    def test_migration_adds_all_v1_defaults(self):
        migrated = migrate_records([minimal_record()])
        record = migrated[0]
        assert record["schema_version"] == "1.0"
        assert record["timestamp_ms"] is None
        assert record["entities"]["setting"] == "unknown"
        assert record["code"]["language"] == "unknown"
        assert record["detections"] == []
        assert record["spatial_relations"] == []
        assert "processing" in record

    def test_migrate_file_never_overwrites_source(self):
        # The equality guard runs before any input read/write, so no temp file is needed.
        input_path = Path("same-metadata-path.json")
        with pytest.raises(ValueError, match="must differ"):
            migrate_file(input_path, input_path)

    def test_write_requires_force_for_existing_output(self, monkeypatch):
        output_path = Path("already-existing-metadata.json")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        with pytest.raises(FileExistsError):
            write_json_atomically(output_path, [minimal_record()])


class TestGeometryAndRelations:
    def test_valid_detection_and_relation(self):
        record = FrameMetadata.model_validate(
            minimal_record(
                detections=[
                    {
                        "object_id": "person_0",
                        "label": "person",
                        "bbox": [0.1, 0.2, 0.3, 0.9],
                        "confidence": 0.8,
                    },
                    {
                        "object_id": "car_0",
                        "label": "car",
                        "bbox": [0.5, 0.3, 0.95, 0.85],
                        "confidence": 0.9,
                    },
                ],
                spatial_relations=[
                    {
                        "subject_id": "person_0",
                        "predicate": "left_of",
                        "object_id": "car_0",
                        "confidence": 0.75,
                    }
                ],
            )
        )
        assert isinstance(record.detections[0], Detection)
        assert isinstance(record.spatial_relations[0], SpatialRelation)

    @pytest.mark.parametrize(
        "bbox",
        [
            [-0.1, 0.2, 0.3, 0.9],
            [0.1, 0.2, 1.1, 0.9],
            [0.5, 0.2, 0.3, 0.9],
            [0.1, 0.9, 0.3, 0.2],
        ],
    )
    def test_rejects_invalid_bbox(self, bbox):
        with pytest.raises(ValidationError):
            Detection(
                object_id="person_0", label="person", bbox=bbox, confidence=0.8
            )

    def test_rejects_confidence_outside_zero_one(self):
        with pytest.raises(ValidationError):
            Detection(
                object_id="person_0",
                label="person",
                bbox=[0.1, 0.2, 0.3, 0.9],
                confidence=1.1,
            )

    def test_rejects_relation_to_unknown_object(self):
        with pytest.raises(ValidationError, match="unknown detection ids"):
            FrameMetadata.model_validate(
                minimal_record(
                    spatial_relations=[
                        {
                            "subject_id": "person_0",
                            "predicate": "left_of",
                            "object_id": "car_0",
                            "confidence": 0.75,
                        }
                    ]
                )
            )

    def test_rejects_duplicate_detection_ids(self):
        detection = {
            "object_id": "person_0",
            "label": "person",
            "bbox": [0.1, 0.2, 0.3, 0.9],
            "confidence": 0.8,
        }
        with pytest.raises(ValidationError, match="must be unique"):
            FrameMetadata.model_validate(
                minimal_record(detections=[detection, detection])
            )


class TestDocumentIntegrity:
    def test_rejects_frame_id_video_mismatch(self):
        with pytest.raises(ValidationError, match="does not match video_name"):
            FrameMetadata.model_validate(minimal_record(video_name="other.mp4"))

    def test_rejects_frame_id_index_mismatch(self):
        with pytest.raises(ValidationError, match="must end with"):
            FrameMetadata.model_validate(minimal_record(frame_index=2))

    def test_rejects_duplicate_frame_ids_across_file(self):
        with pytest.raises(ValueError, match="Duplicate frame_id"):
            validate_metadata_records([minimal_record(), minimal_record()])

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            FrameMetadata.model_validate(minimal_record(capton="misspelled"))


class TestCodeMetadata:
    def test_accepts_auditable_sql_classification(self):
        code = CodeMetadata(
            language="sql",
            statement_type="select",
            patterns=["SELECT", "from", "select"],
            search_terms=["SQL query"],
            evidence=["ocr_text:select"],
            classifier_version="code-rules-v1",
        )
        assert code.patterns == ["select", "from"]
        assert code.search_terms == ["sql query"]

    def test_rejects_sql_without_provenance(self):
        with pytest.raises(ValidationError, match="SQL classification requires"):
            CodeMetadata(language="sql")

    def test_rejects_statement_type_for_unknown_language(self):
        with pytest.raises(ValidationError, match="cannot declare a statement type"):
            CodeMetadata(statement_type="select")
