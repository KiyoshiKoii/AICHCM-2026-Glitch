"""Evaluate Florence detections and spatial rules against manual annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from migrate_metadata import write_json_atomically
    from schemas import FrameMetadata, validate_metadata_file
    from spatial_extractor import DEFAULT_OUTPUT_PATH, bbox_iou
except ImportError:
    from .migrate_metadata import write_json_atomically
    from .schemas import FrameMetadata, validate_metadata_file
    from .spatial_extractor import DEFAULT_OUTPUT_PATH, bbox_iou

SEMANTIC_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_PATH = SEMANTIC_DIR / "spatial_ground_truth.json"
DEFAULT_REPORT_PATH = SEMANTIC_DIR / "spatial_benchmark_report_v1.json"
VALID_PREDICATES = {"left_of", "right_of", "above", "below", "overlapping"}


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH_PATH) -> dict:
    ground_truth = json.loads(Path(path).read_text(encoding="utf-8"))
    if ground_truth.get("version") != "1.0":
        raise ValueError("spatial ground truth must use version 1.0")
    threshold = ground_truth.get("iou_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    frames = ground_truth.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("spatial ground truth must contain a non-empty frames list")
    frame_ids = [frame.get("frame_id") for frame in frames]
    if not all(isinstance(frame_id, str) and frame_id for frame_id in frame_ids):
        raise ValueError("every annotated frame requires frame_id")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("annotated frame_id values must be unique")
    object_count = 0
    for frame in frames:
        objects = frame.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("every annotated frame requires at least one object")
        object_ids = [item.get("object_id") for item in objects]
        if not all(isinstance(object_id, str) and object_id for object_id in object_ids):
            raise ValueError("every annotated object requires object_id")
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("annotated object_id values must be unique per frame")
        for item in objects:
            box = item.get("bbox")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(value, (int, float)) for value in box)
                or not all(0 <= value <= 1 for value in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"Invalid annotated bbox: {item}")
        object_count += len(objects)
    ground_truth["annotation_summary"] = {
        "frames": len(frames),
        "objects": object_count,
        "relation_evaluated_frames": sum(
            frame.get("relations_evaluated", True) for frame in frames
        ),
    }
    return ground_truth


def expand_expected_relations(frame: dict) -> set[tuple[str, str, str]]:
    object_ids = {item["object_id"] for item in frame.get("objects", [])}
    relations: set[tuple[str, str, str]] = set()
    for relation in frame.get("relations", []):
        triple = (
            relation.get("subject_id"),
            relation.get("predicate"),
            relation.get("object_id"),
        )
        if (
            triple[0] not in object_ids
            or triple[2] not in object_ids
            or triple[0] == triple[2]
            or triple[1] not in VALID_PREDICATES
        ):
            raise ValueError(f"Invalid annotated spatial relation: {relation}")
        relations.add(triple)

    for group in frame.get("ordered_groups", []):
        ordered_ids = group.get("object_ids", [])
        forward = group.get("forward")
        reverse = group.get("reverse")
        if (
            len(ordered_ids) < 2
            or len(ordered_ids) != len(set(ordered_ids))
            or not set(ordered_ids) <= object_ids
            or forward not in VALID_PREDICATES
            or reverse not in VALID_PREDICATES
        ):
            raise ValueError(f"Invalid ordered relation group: {group}")
        for left_index, subject_id in enumerate(ordered_ids):
            for object_id in ordered_ids[left_index + 1 :]:
                relations.add((subject_id, forward, object_id))
                relations.add((object_id, reverse, subject_id))
    return relations


def _accepted_labels(item: dict) -> set[str]:
    values = [item["label"], *item.get("label_aliases", [])]
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def _match_detections(
    record: FrameMetadata,
    annotated_objects: list[dict],
    iou_threshold: float,
) -> tuple[dict[str, str], dict[str, bool], list[dict]]:
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, predicted in enumerate(record.detections):
        for truth_index, truth in enumerate(annotated_objects):
            score = bbox_iou(predicted.bbox, tuple(truth["bbox"]))
            if score >= iou_threshold:
                candidates.append((score, predicted_index, truth_index))
    candidates.sort(reverse=True)

    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    predicted_to_truth: dict[str, str] = {}
    label_matches: dict[str, bool] = {}
    matches: list[dict] = []
    for score, predicted_index, truth_index in candidates:
        if predicted_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(predicted_index)
        used_truth.add(truth_index)
        predicted = record.detections[predicted_index]
        truth = annotated_objects[truth_index]
        label_match = predicted.label.casefold() in _accepted_labels(truth)
        predicted_to_truth[predicted.object_id] = truth["object_id"]
        label_matches[predicted.object_id] = label_match
        matches.append(
            {
                "predicted_id": predicted.object_id,
                "predicted_label": predicted.label,
                "truth_id": truth["object_id"],
                "truth_label": truth["label"],
                "iou": score,
                "label_match": label_match,
            }
        )
    return predicted_to_truth, label_matches, matches


def evaluate_spatial_records(
    records: list[FrameMetadata],
    ground_truth: dict,
    min_localization_recall: float = 0.9,
    min_relation_geometry_f1: float = 0.9,
) -> dict:
    records_by_id = {record.frame_id: record for record in records}
    annotated_ids = {frame["frame_id"] for frame in ground_truth["frames"]}
    missing = sorted(annotated_ids - records_by_id.keys())
    if missing:
        raise ValueError(f"Metadata is missing annotated frame_ids: {missing}")
    iou_threshold = float(ground_truth["iou_threshold"])

    total_truth_objects = total_predicted_objects = total_matches = 0
    total_label_matches = 0
    expected_relation_total = predicted_relation_total = relation_true_positive = 0
    queryable_true_positive = 0
    frame_reports: list[dict] = []
    relation_evaluated_frames = 0

    for annotated_frame in ground_truth["frames"]:
        record = records_by_id[annotated_frame["frame_id"]]
        objects = annotated_frame.get("objects", [])
        relations_evaluated = annotated_frame.get("relations_evaluated", True)
        expected_relations = (
            expand_expected_relations(annotated_frame)
            if relations_evaluated
            else set()
        )
        mapping, label_matches, matches = _match_detections(
            record, objects, iou_threshold
        )
        predicted_relations: set[tuple[str, str, str]] = set()
        queryable_relations: set[tuple[str, str, str]] = set()
        if relations_evaluated:
            relation_evaluated_frames += 1
            for relation in record.spatial_relations:
                subject_truth = mapping.get(relation.subject_id)
                object_truth = mapping.get(relation.object_id)
                if subject_truth is None or object_truth is None:
                    continue
                triple = (subject_truth, relation.predicate.value, object_truth)
                predicted_relations.add(triple)
                if (
                    label_matches[relation.subject_id]
                    and label_matches[relation.object_id]
                ):
                    queryable_relations.add(triple)

        relation_tp = len(predicted_relations & expected_relations)
        queryable_tp = len(queryable_relations & expected_relations)
        total_truth_objects += len(objects)
        total_predicted_objects += len(record.detections)
        total_matches += len(matches)
        total_label_matches += sum(item["label_match"] for item in matches)
        if relations_evaluated:
            expected_relation_total += len(expected_relations)
            predicted_relation_total += len(record.spatial_relations)
            relation_true_positive += relation_tp
            queryable_true_positive += queryable_tp
        frame_reports.append(
            {
                "frame_id": record.frame_id,
                "truth_objects": len(objects),
                "predicted_objects": len(record.detections),
                "localized_objects": len(matches),
                "correct_labels": sum(item["label_match"] for item in matches),
                "relations_evaluated": relations_evaluated,
                "expected_relations": (
                    len(expected_relations) if relations_evaluated else None
                ),
                "predicted_relations": (
                    len(record.spatial_relations) if relations_evaluated else None
                ),
                "correct_geometry_relations": (
                    relation_tp if relations_evaluated else None
                ),
                "correct_queryable_relations": (
                    queryable_tp if relations_evaluated else None
                ),
                "matches": matches,
            }
        )

    localization_precision = _safe_divide(total_matches, total_predicted_objects)
    localization_recall = _safe_divide(total_matches, total_truth_objects)
    relation_precision = _safe_divide(
        relation_true_positive, predicted_relation_total
    )
    relation_recall = _safe_divide(relation_true_positive, expected_relation_total)
    queryable_precision = _safe_divide(
        queryable_true_positive, predicted_relation_total
    )
    queryable_recall = _safe_divide(
        queryable_true_positive, expected_relation_total
    )
    metrics = {
        "localization": {
            "precision": localization_precision,
            "recall": localization_recall,
            "f1": _f1(localization_precision, localization_recall),
            "matched": total_matches,
            "predicted": total_predicted_objects,
            "ground_truth": total_truth_objects,
        },
        "labels": {
            "accuracy_on_localized_objects": _safe_divide(
                total_label_matches, total_matches
            ),
            "correct": total_label_matches,
            "localized": total_matches,
        },
        "relation_geometry": {
            "precision": relation_precision,
            "recall": relation_recall,
            "f1": _f1(relation_precision, relation_recall),
            "correct": relation_true_positive,
            "predicted": predicted_relation_total,
            "ground_truth": expected_relation_total,
        },
        "queryable_relations": {
            "precision": queryable_precision,
            "recall": queryable_recall,
            "f1": _f1(queryable_precision, queryable_recall),
            "correct": queryable_true_positive,
            "predicted": predicted_relation_total,
            "ground_truth": expected_relation_total,
        },
    }
    checks = {
        "localization_recall": {
            "value": localization_recall,
            "minimum": min_localization_recall,
            "passed": localization_recall >= min_localization_recall,
        },
        "relation_geometry_f1": {
            "value": metrics["relation_geometry"]["f1"],
            "minimum": min_relation_geometry_f1,
            "passed": metrics["relation_geometry"]["f1"]
            >= min_relation_geometry_f1,
        },
    }
    warnings = []
    if metrics["labels"]["accuracy_on_localized_objects"] < 0.8:
        warnings.append(
            "Florence localized objects but mislabeled some classes; exact spatial "
            "label filters need a stronger detector or label-grounding stage."
        )
    return {
        "ground_truth_version": ground_truth["version"],
        "annotated_frames": len(ground_truth["frames"]),
        "iou_threshold": iou_threshold,
        "metrics": metrics,
        "annotation_coverage": {
            "frames": len(ground_truth["frames"]),
            "objects": total_truth_objects,
            "relation_evaluated_frames": relation_evaluated_frames,
        },
        "acceptance": {
            "passed": all(check["passed"] for check in checks.values()),
            "checks": checks,
            "warnings": warnings,
        },
        "frames": frame_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark spatial metadata against manual ground truth."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-localization-recall", type=float, default=0.9)
    parser.add_argument("--min-relation-geometry-f1", type=float, default=0.9)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        ground_truth = load_ground_truth(args.ground_truth)
        records = validate_metadata_file(args.metadata)
        report = evaluate_spatial_records(
            records,
            ground_truth,
            min_localization_recall=args.min_localization_recall,
            min_relation_geometry_f1=args.min_relation_geometry_f1,
        )
        report.update(
            {
                "metadata_path": str(args.metadata.resolve()),
                "ground_truth_path": str(args.ground_truth.resolve()),
            }
        )
        write_json_atomically(args.output, report, overwrite=True)
    except Exception as exc:
        parser.exit(status=1, message=f"Spatial benchmark failed: {exc}\n")

    metrics: dict[str, Any] = report["metrics"]
    print(
        "Spatial benchmark | "
        f"frames={report['annotated_frames']} | "
        f"localization_recall={metrics['localization']['recall']:.4f} | "
        f"label_accuracy={metrics['labels']['accuracy_on_localized_objects']:.4f} | "
        f"relation_geometry_f1={metrics['relation_geometry']['f1']:.4f} | "
        f"queryable_relation_recall={metrics['queryable_relations']['recall']:.4f}"
    )
    for warning in report["acceptance"]["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved report to {args.output.resolve()}")
    if not report["acceptance"]["passed"]:
        parser.exit(status=2, message="Spatial benchmark acceptance failed.\n")


if __name__ == "__main__":
    main()
