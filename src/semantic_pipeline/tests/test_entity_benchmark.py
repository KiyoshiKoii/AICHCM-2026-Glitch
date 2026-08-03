"""Tests for the independent entity visual-truth benchmark."""

from pathlib import Path

import pytest

from benchmark_entities import (
    DEFAULT_GROUND_TRUTH_PATH,
    evaluate_entity_records,
    load_entity_ground_truth,
)
from schemas import FrameMetadata


def _record(entities: dict | None = None) -> FrameMetadata:
    return FrameMetadata.model_validate(
        {
            "frame_id": "vid01_f0001",
            "video_name": "vid01.mp4",
            "frame_index": 1,
            "caption": "Fixture caption is not used as visual ground truth.",
            "entities": entities or {},
        }
    )


def _ground_truth(entities: dict | None = None) -> dict:
    return {
        "version": "1.0",
        "annotation_method": "unit-test pixels",
        "frames": [
            {
                "frame_id": "vid01_f0001",
                "entities": entities
                or {
                    "time_of_day": "unknown",
                    "setting": "unknown",
                    "locations": [],
                    "objects": [],
                    "actions": [],
                    "colors": [],
                },
            }
        ],
    }


def _evaluate(record: FrameMetadata, truth: dict) -> dict:
    return evaluate_entity_records(
        [record],
        truth,
        min_annotated_frames=1,
        min_overall_micro_f1=0.0,
        min_searchable_micro_f1=0.0,
        min_object_recall=0.0,
    )


def test_aliases_match_one_to_one_and_redundant_prediction_is_false_positive():
    record = _record(
        {
            "objects": ["bar chart", "chart"],
        }
    )
    truth = _ground_truth(
        {
            "time_of_day": "unknown",
            "setting": "unknown",
            "locations": [],
            "objects": [
                {"value": "bar chart", "aliases": ["bar graph", "chart"]}
            ],
            "actions": [],
            "colors": [],
        }
    )

    report = _evaluate(record, truth)
    objects = report["metrics"]["fields"]["objects"]
    assert objects["counts"] == {"tp": 1, "fp": 1, "fn": 0}
    assert objects["micro"]["precision"] == 0.5
    assert objects["micro"]["recall"] == 1.0
    assert report["error_analysis"]["objects"]["top_false_positives"] == [
        {"value": "chart", "count": 1}
    ]


def test_empty_array_is_known_negative_but_null_annotation_is_skipped():
    record = _record(
        {
            "locations": ["office"],
            "objects": ["screen"],
            "actions": [],
            "colors": [],
        }
    )
    truth = _ground_truth(
        {
            "time_of_day": "unknown",
            "setting": "unknown",
            "locations": [],
            "objects": None,
            "actions": [],
            "colors": [],
        }
    )

    report = _evaluate(record, truth)
    locations = report["metrics"]["fields"]["locations"]
    objects = report["metrics"]["fields"]["objects"]
    assert locations["counts"] == {"tp": 0, "fp": 1, "fn": 0}
    assert locations["negative_handling"] == {
        "gold_negative_frames": 1,
        "correct_negative_frames": 0,
        "false_positive_negative_frames": 1,
    }
    assert objects["evaluated_frames"] == 0
    assert objects["skipped_annotations"] == 1


def test_explicit_unknown_is_a_scored_scalar_class():
    report = _evaluate(_record(), _ground_truth())
    for field in ("time_of_day", "setting"):
        metric = report["metrics"]["fields"][field]
        assert metric["counts"] == {"tp": 1, "fp": 0, "fn": 0}
        assert metric["unknown_handling"]["explicit_unknown_gold"] == 1
        assert metric["unknown_handling"]["correct_unknown_predictions"] == 1


def test_ignored_prediction_is_neither_true_nor_false_positive():
    truth = _ground_truth()
    truth["frames"][0]["ignored_predictions"] = {"objects": ["ambiguous shape"]}
    record = _record({"objects": ["ambiguous shape"]})

    report = _evaluate(record, truth)
    frame_objects = report["frames"][0]["fields"]["objects"]
    assert frame_objects["ignored_predictions"] == ["ambiguous shape"]
    assert frame_objects["tp"] == frame_objects["fp"] == frame_objects["fn"] == 0


def test_missing_annotated_frame_is_rejected():
    with pytest.raises(ValueError, match="missing annotated"):
        evaluate_entity_records([], _ground_truth(), min_annotated_frames=1)


def test_default_ground_truth_has_independent_twelve_frame_coverage():
    ground_truth = load_entity_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    assert len(ground_truth["frames"]) == 12
    assert "pixels" in ground_truth["annotation_method"]
    semantic_dir = Path(DEFAULT_GROUND_TRUTH_PATH).parent
    assert all(
        (semantic_dir / frame["image_file"]).is_file()
        for frame in ground_truth["frames"]
    )


def test_coverage_gate_fails_for_tiny_fixture_without_hiding_metrics():
    report = evaluate_entity_records(
        [_record()],
        _ground_truth(),
        min_annotated_frames=10,
        min_overall_micro_f1=0.0,
        min_searchable_micro_f1=0.0,
        min_object_recall=0.0,
    )
    coverage = report["acceptance"]["checks"]["annotated_frame_coverage"]
    assert coverage["passed"] is False
    assert report["acceptance"]["passed"] is False
    assert report["metrics"]["overall"]["micro"]["f1"] == 1.0
