"""BTC object adapter tests: axis order, score pruning, and class-wise NMS."""

import json

import pytest

from btc_objects import filter_btc_detections, load_btc_object_file
from spatial_extractor import RawDetection


def raw(label, mid, score, box):
    return RawDetection(
        label=label,
        mid=mid,
        confidence=score,
        bbox=box,
        label_source="btc_detector",
    )


def test_threshold_nms_is_class_wise_and_keeps_real_scores():
    detections = filter_btc_detections(
        [
            raw("Car", "/m/car", 0.90, (0.1, 0.2, 0.5, 0.8)),
            raw("Car", "/m/car", 0.80, (0.11, 0.2, 0.51, 0.8)),
            raw("Bus", "/m/bus", 0.70, (0.11, 0.2, 0.51, 0.8)),
            raw("Person", "/m/person", 0.19, (0.6, 0.1, 0.8, 0.9)),
        ]
    )

    assert [(item.label, item.confidence) for item in detections] == [
        ("bus", 0.70),
        ("car", 0.90),
    ]
    assert all(item.label_source == "btc_detector" for item in detections)
    assert all(item.mid and item.grid_cell and item.area for item in detections)


def test_btc_yxyx_box_is_converted_to_xyxy(tmp_path):
    source = tmp_path / "001.json"
    source.write_text(
        json.dumps(
            {
                "detection_scores": ["0.9"],
                "detection_class_names": ["/m/car"],
                "detection_class_entities": ["Car"],
                "detection_boxes": [["0.2", "0.1", "0.8", "0.5"]],
            }
        ),
        encoding="utf-8",
    )
    detections, stats = load_btc_object_file(source)
    assert detections[0].bbox == (0.1, 0.2, 0.5, 0.8)
    assert stats == {"raw": 1, "above_threshold": 1, "filtered": 1}


def test_zero_detection_frame_does_not_crash(tmp_path):
    source = tmp_path / "001.json"
    source.write_text(
        json.dumps(
            {
                "detection_scores": [],
                "detection_class_names": [],
                "detection_class_entities": [],
                "detection_boxes": [],
            }
        ),
        encoding="utf-8",
    )
    assert load_btc_object_file(source)[0] == []


def test_rejects_parallel_array_mismatch(tmp_path):
    source = tmp_path / "001.json"
    source.write_text(
        json.dumps(
            {
                "detection_scores": ["0.9"],
                "detection_class_names": [],
                "detection_class_entities": ["Car"],
                "detection_boxes": [["0", "0", "1", "1"]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different lengths"):
        load_btc_object_file(source)
