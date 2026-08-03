"""Tests for the manually annotated spatial acceptance benchmark."""

import pytest

from benchmark_spatial import (
    evaluate_spatial_records,
    expand_expected_relations,
    load_ground_truth,
)
from schemas import FrameMetadata


def ground_truth():
    return {
        "version": "1.0",
        "iou_threshold": 0.5,
        "frames": [
            {
                "frame_id": "vid01_f0001",
                "objects": [
                    {
                        "object_id": "person_truth",
                        "label": "person",
                        "label_aliases": ["human"],
                        "bbox": [0.1, 0.2, 0.3, 0.8],
                    },
                    {
                        "object_id": "car_truth",
                        "label": "car",
                        "label_aliases": [],
                        "bbox": [0.6, 0.2, 0.9, 0.8],
                    },
                ],
                "relations": [],
                "ordered_groups": [
                    {
                        "object_ids": ["person_truth", "car_truth"],
                        "forward": "left_of",
                        "reverse": "right_of",
                    }
                ],
            }
        ],
    }


def predicted_record(person_label="person"):
    return FrameMetadata.model_validate(
        {
            "frame_id": "vid01_f0001",
            "video_name": "vid01.mp4",
            "frame_index": 1,
            "caption": "A person is left of a car.",
            "detections": [
                {
                    "object_id": "person_0",
                    "label": person_label,
                    "bbox": [0.1, 0.2, 0.3, 0.8],
                    "confidence": 0.5,
                },
                {
                    "object_id": "car_0",
                    "label": "car",
                    "bbox": [0.6, 0.2, 0.9, 0.8],
                    "confidence": 0.5,
                },
            ],
            "spatial_relations": [
                {
                    "subject_id": "person_0",
                    "subject_label": person_label,
                    "predicate": "left_of",
                    "object_id": "car_0",
                    "object_label": "car",
                    "confidence": 0.4,
                },
                {
                    "subject_id": "car_0",
                    "subject_label": "car",
                    "predicate": "right_of",
                    "object_id": "person_0",
                    "object_label": person_label,
                    "confidence": 0.4,
                },
            ],
        }
    )


def test_expands_ordered_group_into_reciprocal_pairs():
    relations = expand_expected_relations(ground_truth()["frames"][0])
    assert relations == {
        ("person_truth", "left_of", "car_truth"),
        ("car_truth", "right_of", "person_truth"),
    }


def test_repository_gold_set_keeps_minimum_manual_coverage():
    truth = load_ground_truth()
    assert truth["annotation_summary"]["frames"] >= 10
    assert truth["annotation_summary"]["objects"] >= 30
    assert truth["annotation_summary"]["relation_evaluated_frames"] >= 3


def test_perfect_geometry_and_labels_pass_acceptance():
    report = evaluate_spatial_records([predicted_record()], ground_truth())
    assert report["acceptance"]["passed"] is True
    assert report["metrics"]["localization"]["f1"] == 1.0
    assert report["metrics"]["relation_geometry"]["f1"] == 1.0
    assert report["metrics"]["queryable_relations"]["recall"] == 1.0


def test_wrong_label_does_not_hide_correct_geometry_but_warns():
    report = evaluate_spatial_records(
        [predicted_record(person_label="statue")], ground_truth()
    )
    assert report["metrics"]["localization"]["recall"] == 1.0
    assert report["metrics"]["relation_geometry"]["recall"] == 1.0
    assert report["metrics"]["labels"]["accuracy_on_localized_objects"] == 0.5
    assert report["metrics"]["queryable_relations"]["recall"] == 0.0
    assert report["acceptance"]["warnings"]


def test_missing_annotated_frame_is_rejected():
    with pytest.raises(ValueError, match="missing annotated"):
        evaluate_spatial_records([], ground_truth())


def test_object_only_frame_expands_coverage_without_changing_relation_metrics():
    truth = ground_truth()
    truth["frames"].append(
        {
            "frame_id": "vid01_f0002",
            "objects": [
                {
                    "object_id": "flower_truth",
                    "label": "flower",
                    "label_aliases": [],
                    "bbox": [0.1, 0.1, 0.3, 0.3],
                }
            ],
            "relations_evaluated": False,
            "relations": [],
            "ordered_groups": [],
        }
    )
    object_only = FrameMetadata.model_validate(
        {
            "frame_id": "vid01_f0002",
            "video_name": "vid01.mp4",
            "frame_index": 2,
            "caption": "A flower.",
            "detections": [
                {
                    "object_id": "flower_0",
                    "label": "flower",
                    "bbox": [0.1, 0.1, 0.3, 0.3],
                    "confidence": 0.5,
                }
            ],
        }
    )

    report = evaluate_spatial_records(
        [predicted_record(), object_only], truth
    )

    assert report["annotation_coverage"] == {
        "frames": 2,
        "objects": 3,
        "relation_evaluated_frames": 1,
    }
    assert report["metrics"]["relation_geometry"]["predicted"] == 2
    assert report["frames"][1]["predicted_relations"] is None


def test_invalid_relation_group_is_rejected():
    frame = ground_truth()["frames"][0]
    frame["ordered_groups"][0]["forward"] = "diagonal_to"
    with pytest.raises(ValueError, match="Invalid ordered"):
        expand_expected_relations(frame)
