"""Unit tests for Florence box normalisation and pure spatial geometry."""

from pathlib import Path

from PIL import Image
import pytest
from pydantic import ValidationError

from schemas import Detection, FrameMetadata
from spatial_extractor import (
    SPATIAL_RULE_VERSION,
    RawDetection,
    enrich_spatial_file,
    enrich_spatial_records,
    ground_detection_labels,
    infer_spatial_relations,
    normalise_detections,
)


def frame(frame_index: int = 1, detection_model=None) -> FrameMetadata:
    return FrameMetadata.model_validate(
        {
            "frame_id": f"vid01_f{frame_index:04d}",
            "video_name": "vid01.mp4",
            "frame_index": frame_index,
            "caption": "A person is standing to the left of a car.",
            "ocr_text": "",
            "processing": {"detection_model": detection_model},
        }
    )


def detection(object_id, label, bbox, confidence=0.8) -> Detection:
    return Detection(
        object_id=object_id,
        label=label,
        bbox=bbox,
        confidence=confidence,
    )


class FakeDetector:
    model = "fake-florence-od"

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        assert image.size == (100, 100)
        return [
            RawDetection("Person", (10, 20, 30, 80), 0.8),
            RawDetection("Car", (60, 20, 95, 80), 0.9),
        ]


class TestBoxNormalisation:
    def test_clamps_filters_deduplicates_and_assigns_deterministic_ids(self):
        result = normalise_detections(
            [
                RawDetection("Person", (60, 10, 90, 90), 0.9),
                RawDetection("car", (-10, 20, 20, 80), None),
                RawDetection(" person ", (10, 10, 40, 90), 0.8),
                RawDetection("person", (10, 10, 40, 90), 0.7),
                RawDetection("noise", (1, 1, 1.01, 1.01), 0.5),
                RawDetection("invalid", (50, 50, 40, 80), 0.5),
            ],
            image_width=100,
            image_height=100,
        )

        assert [item.object_id for item in result] == [
            "car_0",
            "person_0",
            "person_1",
        ]
        assert result[0].bbox == (0.0, 0.2, 0.2, 0.8)
        assert result[0].confidence == 0.5
        assert result[1].bbox[0] == 0.1
        assert result[2].bbox[0] == 0.6

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"default_confidence": 1.1},
            {"min_box_area": -0.1},
            {"duplicate_iou": 1.1},
            {"max_detections": 0},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            normalise_detections([], 100, 100, **kwargs)


class TestSpatialGeometry:
    def test_left_right_and_above_below_are_reciprocal(self):
        person = detection("person_0", "person", (0.1, 0.1, 0.3, 0.6))
        car = detection("car_0", "car", (0.6, 0.1, 0.9, 0.6))
        sign = detection("sign_0", "sign", (0.1, 0.75, 0.3, 0.95))

        relations = infer_spatial_relations([person, car, sign])
        triples = {
            (item.subject_id, item.predicate.value, item.object_id)
            for item in relations
        }

        assert ("person_0", "left_of", "car_0") in triples
        assert ("car_0", "right_of", "person_0") in triples
        assert ("person_0", "above", "sign_0") in triples
        assert ("sign_0", "below", "person_0") in triples
        assert all(item.subject_label and item.object_label for item in relations)

    def test_overlap_uses_smaller_box_and_is_reciprocal(self):
        notebook = detection("notebook_0", "notebook", (0.2, 0.2, 0.8, 0.8))
        hand = detection("hand_0", "hand", (0.3, 0.3, 0.5, 0.5))

        relations = infer_spatial_relations([notebook, hand])
        triples = {
            (item.subject_id, item.predicate.value, item.object_id)
            for item in relations
        }

        assert triples == {
            ("notebook_0", "overlapping", "hand_0"),
            ("hand_0", "overlapping", "notebook_0"),
        }

    def test_diagonal_boxes_without_axis_projection_do_not_create_noise(self):
        first = detection("first_0", "first", (0.05, 0.05, 0.2, 0.2))
        second = detection("second_0", "second", (0.8, 0.8, 0.95, 0.95))
        assert infer_spatial_relations([first, second]) == []


class TestContextualLabelGrounding:
    def test_preserves_btc_detector_provenance_without_correction(self):
        car = Detection(
            object_id="car_0",
            label="car",
            mid="/m/0k4j",
            label_source="btc_detector",
            bbox=(0.1, 0.2, 0.3, 0.5),
            confidence=0.9,
        )
        grounded, count = ground_detection_labels([car], frame())
        assert count == 0
        assert grounded[0].label_source == "btc_detector"

    def test_corrects_known_confusion_only_when_frame_has_evidence(self):
        record = frame()
        record = record.model_copy(
            update={"ocr_text": "Six sacks are lined up in a row."}
        )
        vase = detection("vase_0", "vase", (0.1, 0.2, 0.3, 0.5))

        grounded, count = ground_detection_labels([vase], record)

        assert count == 1
        assert grounded[0].label == "sack"
        assert grounded[0].raw_label == "vase"
        assert grounded[0].label_source == "context_grounding"
        assert grounded[0].label_evidence == ["ocr_text:sacks"]

    def test_does_not_guess_without_target_concept_evidence(self):
        vase = detection("vase_0", "vase", (0.1, 0.2, 0.3, 0.5))
        grounded, count = ground_detection_labels([vase], frame())
        assert count == 0
        assert grounded[0] == vase

    def test_full_frame_container_uses_entity_evidence_and_is_idempotent(self):
        record = frame().model_copy(
            update={
                "entities": frame().entities.model_copy(
                    update={"objects": ["screenshot", "text"]}
                )
            }
        )
        poster = detection("poster_0", "poster", (0.001, 0.001, 0.999, 0.999))

        first, first_count = ground_detection_labels([poster], record)
        second, second_count = ground_detection_labels(first, record)

        assert first_count == second_count == 1
        assert first == second
        assert first[0].label == "screenshot"
        assert first[0].raw_label == "poster"
        assert first[0].label_evidence == ["entities.objects:screenshot"]


class TestBatchPipeline:
    @staticmethod
    def image_loader(record):
        return Image.new("RGB", (100, 100), "white")

    def test_enriches_detections_relations_and_provenance(self):
        records = [frame()]
        summary = enrich_spatial_records(
            records,
            FakeDetector(),
            self.image_loader,
        )

        assert summary == {
            "processed": 1,
            "skipped": 0,
            "total": 1,
            "detections": 2,
            "relations": 2,
            "grounded_labels": 0,
        }
        assert [item.label for item in records[0].detections] == ["car", "person"]
        assert records[0].processing.detection_model == "fake-florence-od"
        assert records[0].processing.spatial_rule_version == SPATIAL_RULE_VERSION
        assert records[0].processing.spatial_processed_at is not None

    def test_limit_counts_only_selected_unprocessed_records(self):
        records = [frame(1, "done"), frame(2), frame(3)]
        detector = FakeDetector()
        summary = enrich_spatial_records(
            records,
            detector,
            self.image_loader,
            frame_ids={"vid01_f0001", "vid01_f0002", "vid01_f0003"},
            limit=1,
        )

        assert summary["processed"] == 1
        assert summary["skipped"] == 1
        assert detector.calls == 1
        assert records[1].processing.detection_model == "fake-florence-od"
        assert records[2].processing.detection_model is None

    def test_unknown_frame_selection_fails_before_model_call(self):
        detector = FakeDetector()
        with pytest.raises(ValueError, match="Unknown frame_ids"):
            enrich_spatial_records(
                [frame()],
                detector,
                self.image_loader,
                frame_ids={"missing_f0001"},
            )
        assert detector.calls == 0

    def test_file_pipeline_never_overwrites_source(self):
        same_path = Path("same-spatial-metadata.json")
        with pytest.raises(ValueError, match="must differ"):
            enrich_spatial_file(
                same_path,
                Path("images"),
                same_path,
                FakeDetector(),
            )


def test_schema_rejects_relation_labels_that_disagree_with_detection_ids():
    with pytest.raises(ValidationError, match="labels do not match"):
        FrameMetadata.model_validate(
            {
                "frame_id": "vid01_f0001",
                "video_name": "vid01.mp4",
                "frame_index": 1,
                "caption": "A person is left of a car.",
                "detections": [
                    {
                        "object_id": "person_0",
                        "label": "person",
                        "bbox": [0.1, 0.2, 0.3, 0.8],
                        "confidence": 0.8,
                    },
                    {
                        "object_id": "car_0",
                        "label": "car",
                        "bbox": [0.6, 0.2, 0.9, 0.8],
                        "confidence": 0.8,
                    },
                ],
                "spatial_relations": [
                    {
                        "subject_id": "person_0",
                        "subject_label": "car",
                        "predicate": "left_of",
                        "object_id": "car_0",
                        "object_label": "person",
                        "confidence": 0.7,
                    }
                ],
            }
        )


def test_schema_rejects_untraceable_context_grounded_label():
    with pytest.raises(ValidationError, match="require a different raw_label"):
        Detection.model_validate(
            {
                "object_id": "sack_0",
                "label": "sack",
                "label_source": "context_grounding",
                "bbox": [0.1, 0.2, 0.3, 0.5],
                "confidence": 0.5,
            }
        )
