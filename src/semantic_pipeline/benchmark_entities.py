"""Evaluate extracted entities against independent visual annotations.

The ground truth in :mod:`entity_visual_ground_truth.json` is derived from the
sample-frame pixels, not from captions, OCR, or model output.  This benchmark
therefore measures whether searchable metadata describes what is actually
visible instead of merely repeating an upstream caption.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from common import paths
except ImportError:  # pragma: no cover - package import from repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import paths

try:
    from migrate_metadata import write_json_atomically
    from schemas import FrameMetadata, validate_metadata_file
    from entity_extractor import DEFAULT_OUTPUT_PATH
except ImportError:
    from .migrate_metadata import write_json_atomically
    from .schemas import FrameMetadata, validate_metadata_file
    from .entity_extractor import DEFAULT_OUTPUT_PATH


SEMANTIC_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_PATH = SEMANTIC_DIR / "entity_visual_ground_truth.json"
DEFAULT_REPORT_PATH = paths.reports_dir() / "entity_benchmark_report_v1.json"

SCALAR_FIELDS = ("time_of_day", "setting")
SET_FIELDS = ("locations", "objects", "actions", "colors")
ENTITY_FIELDS = (*SCALAR_FIELDS, *SET_FIELDS)
SCALAR_VALUES = {
    "time_of_day": {"unknown", "morning", "afternoon", "evening", "night"},
    "setting": {"unknown", "indoor", "outdoor"},
}


def _safe_divide(numerator: int, denominator: int) -> float:
    """Use the standard information-retrieval empty-set convention."""
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def _truth_item(raw_item: str | dict) -> dict[str, Any]:
    if isinstance(raw_item, str):
        value = raw_item
        aliases: list[str] = []
    elif isinstance(raw_item, dict):
        value = raw_item.get("value")
        aliases = raw_item.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(f"invalid entity aliases: {raw_item!r}")
    else:
        raise ValueError(f"entity annotation must be a string/object: {raw_item!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"entity annotation requires a non-empty value: {raw_item!r}")
    accepted = {_normalise(value), *(_normalise(alias) for alias in aliases)}
    return {"value": value.strip().casefold(), "accepted": accepted}


def _validate_frame(frame: dict) -> None:
    frame_id = frame.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("every entity annotation requires frame_id")
    entities = frame.get("entities")
    if not isinstance(entities, dict):
        raise ValueError(f"annotated frame {frame_id!r} requires entities")
    unexpected = set(entities) - set(ENTITY_FIELDS)
    missing = set(ENTITY_FIELDS) - set(entities)
    if unexpected or missing:
        raise ValueError(
            f"annotated frame {frame_id!r} has invalid fields; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    for field in SCALAR_FIELDS:
        value = entities[field]
        # null means the human annotator deliberately excluded an ambiguous field.
        if value is not None and value not in SCALAR_VALUES[field]:
            raise ValueError(f"invalid {field} annotation in {frame_id!r}: {value!r}")
    for field in SET_FIELDS:
        values = entities[field]
        # null is supported as an unscored annotation, distinct from [] (negative).
        if values is None:
            continue
        if not isinstance(values, list):
            raise ValueError(f"{field} in {frame_id!r} must be an array or null")
        parsed = [_truth_item(item) for item in values]
        canonical = [_normalise(item["value"]) for item in parsed]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f"duplicate {field} annotations in {frame_id!r}")

    ignored = frame.get("ignored_predictions", {})
    if not isinstance(ignored, dict) or set(ignored) - set(ENTITY_FIELDS):
        raise ValueError(f"invalid ignored_predictions in {frame_id!r}")
    for field, values in ignored.items():
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(
                f"ignored_predictions.{field} in {frame_id!r} must be strings"
            )


def load_entity_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH_PATH) -> dict:
    ground_truth = json.loads(Path(path).read_text(encoding="utf-8"))
    if ground_truth.get("version") != "1.0":
        raise ValueError("entity visual ground truth must use version 1.0")
    frames = ground_truth.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("entity visual ground truth requires a non-empty frames list")
    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError("every entity visual annotation must be an object")
    for frame in frames:
        _validate_frame(frame)
    frame_ids = [frame["frame_id"] for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("entity visual annotation frame_id values must be unique")
    return ground_truth


def _maximum_matches(
    predicted: list[str], truth_items: list[dict[str, Any]]
) -> dict[int, int]:
    """Return a maximum one-to-one match using per-truth accepted aliases."""
    truth_to_prediction: dict[int, int] = {}

    def augment(predicted_index: int, visited_truth: set[int]) -> bool:
        value = _normalise(predicted[predicted_index])
        for truth_index, truth in enumerate(truth_items):
            if truth_index in visited_truth or value not in truth["accepted"]:
                continue
            visited_truth.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or augment(previous, visited_truth):
                truth_to_prediction[truth_index] = predicted_index
                return True
        return False

    for predicted_index in range(len(predicted)):
        augment(predicted_index, set())
    return {
        predicted_index: truth_index
        for truth_index, predicted_index in truth_to_prediction.items()
    }


def _predicted_values(record: FrameMetadata, field: str) -> list[str]:
    value = getattr(record.entities, field)
    if field in SCALAR_FIELDS:
        return [value.value]
    return list(value)


def _evaluate_field(
    predicted: list[str],
    truth: str | list[str | dict] | None,
    ignored_values: list[str],
) -> dict[str, Any]:
    if truth is None:
        return {
            "skipped": True,
            "reason": "annotation explicitly marked unscored (null)",
            "predicted": predicted,
        }

    raw_truth = [truth] if isinstance(truth, str) else truth
    truth_items = [_truth_item(item) for item in raw_truth]
    ignored_normalised = {_normalise(value) for value in ignored_values}
    scored_predictions = [
        value for value in predicted if _normalise(value) not in ignored_normalised
    ]
    ignored_predictions = [
        value for value in predicted if _normalise(value) in ignored_normalised
    ]
    matches = _maximum_matches(scored_predictions, truth_items)
    matched_truth = set(matches.values())
    tp = len(matches)
    fp = len(scored_predictions) - tp
    fn = len(truth_items) - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "skipped": False,
        "gold": [item["value"] for item in truth_items],
        "predicted": scored_predictions,
        "ignored_predictions": ignored_predictions,
        "matches": [
            {
                "predicted": scored_predictions[predicted_index],
                "gold": truth_items[truth_index]["value"],
            }
            for predicted_index, truth_index in sorted(matches.items())
        ],
        "false_positives": [
            value
            for predicted_index, value in enumerate(scored_predictions)
            if predicted_index not in matches
        ],
        "false_negatives": [
            item["value"]
            for truth_index, item in enumerate(truth_items)
            if truth_index not in matched_truth
        ],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def evaluate_entity_records(
    records: list[FrameMetadata],
    ground_truth: dict,
    min_annotated_frames: int = 10,
    min_overall_micro_f1: float = 0.5,
    min_searchable_micro_f1: float = 0.4,
    min_object_recall: float = 0.5,
) -> dict:
    if min_annotated_frames < 1:
        raise ValueError("min_annotated_frames must be at least 1")
    for threshold_name, threshold in {
        "min_overall_micro_f1": min_overall_micro_f1,
        "min_searchable_micro_f1": min_searchable_micro_f1,
        "min_object_recall": min_object_recall,
    }.items():
        if not 0 <= threshold <= 1:
            raise ValueError(f"{threshold_name} must be in [0, 1]")

    records_by_id = {record.frame_id: record for record in records}
    annotated_ids = {frame["frame_id"] for frame in ground_truth["frames"]}
    missing = sorted(annotated_ids - records_by_id.keys())
    if missing:
        raise ValueError(f"metadata is missing annotated frame_ids: {missing}")

    aggregates: dict[str, dict[str, Any]] = {
        field: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "frame_precision": [],
            "frame_recall": [],
            "frame_f1": [],
            "evaluated_frames": 0,
            "skipped_annotations": 0,
            "gold_negative_frames": 0,
            "correct_negative_frames": 0,
            "false_positive_negative_frames": 0,
            "explicit_unknown_gold": 0,
            "correct_unknown_predictions": 0,
        }
        for field in ENTITY_FIELDS
    }
    frame_reports: list[dict[str, Any]] = []

    for annotated_frame in ground_truth["frames"]:
        record = records_by_id[annotated_frame["frame_id"]]
        ignored_by_field = annotated_frame.get("ignored_predictions", {})
        field_reports: dict[str, dict[str, Any]] = {}
        for field in ENTITY_FIELDS:
            truth = annotated_frame["entities"][field]
            predicted = _predicted_values(record, field)
            field_report = _evaluate_field(
                predicted, truth, ignored_by_field.get(field, [])
            )
            field_reports[field] = field_report
            aggregate = aggregates[field]
            if field_report["skipped"]:
                aggregate["skipped_annotations"] += 1
                continue

            aggregate["evaluated_frames"] += 1
            for metric in ("tp", "fp", "fn"):
                aggregate[metric] += field_report[metric]
            for metric in ("precision", "recall", "f1"):
                aggregate[f"frame_{metric}"].append(field_report[metric])

            if field in SET_FIELDS and not field_report["gold"]:
                aggregate["gold_negative_frames"] += 1
                if field_report["predicted"]:
                    aggregate["false_positive_negative_frames"] += 1
                else:
                    aggregate["correct_negative_frames"] += 1
            if field in SCALAR_FIELDS and truth == "unknown":
                aggregate["explicit_unknown_gold"] += 1
                if field_report["tp"] == 1:
                    aggregate["correct_unknown_predictions"] += 1

        frame_reports.append(
            {"frame_id": record.frame_id, "fields": field_reports}
        )

    field_metrics: dict[str, dict[str, Any]] = {}
    for field, aggregate in aggregates.items():
        tp, fp, fn = aggregate["tp"], aggregate["fp"], aggregate["fn"]
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        metrics = {
            "counts": {"tp": tp, "fp": fp, "fn": fn},
            "micro": {
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            },
            "macro_by_frame": {
                "precision": _mean(aggregate["frame_precision"]),
                "recall": _mean(aggregate["frame_recall"]),
                "f1": _mean(aggregate["frame_f1"]),
            },
            "evaluated_frames": aggregate["evaluated_frames"],
            "skipped_annotations": aggregate["skipped_annotations"],
        }
        if field in SET_FIELDS:
            metrics["negative_handling"] = {
                "gold_negative_frames": aggregate["gold_negative_frames"],
                "correct_negative_frames": aggregate["correct_negative_frames"],
                "false_positive_negative_frames": aggregate[
                    "false_positive_negative_frames"
                ],
            }
        else:
            metrics["unknown_handling"] = {
                "explicit_unknown_gold": aggregate["explicit_unknown_gold"],
                "correct_unknown_predictions": aggregate[
                    "correct_unknown_predictions"
                ],
                "unscored_null_annotations": aggregate["skipped_annotations"],
            }
        field_metrics[field] = metrics

    def combined_micro(fields: tuple[str, ...]) -> dict[str, float | int]:
        tp = sum(field_metrics[field]["counts"]["tp"] for field in fields)
        fp = sum(field_metrics[field]["counts"]["fp"] for field in fields)
        fn = sum(field_metrics[field]["counts"]["fn"] for field in fields)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    all_fields_micro = combined_micro(ENTITY_FIELDS)
    searchable_micro = combined_micro(SET_FIELDS)
    overall = {
        # Keep all-fields micro for a complete schema view, but expose the set-only
        # metric separately so frequent scalar "unknown" labels cannot hide weak
        # searchable metadata.
        "micro": all_fields_micro,
        "searchable_entities_micro": searchable_micro,
        "scalar_classification_micro": combined_micro(SCALAR_FIELDS),
        "macro_by_field": {
            metric: _mean(
                [field_metrics[field]["micro"][metric] for field in ENTITY_FIELDS]
            )
            for metric in ("precision", "recall", "f1")
        },
    }

    checks = {
        "annotated_frame_coverage": {
            "value": len(ground_truth["frames"]),
            "minimum": min_annotated_frames,
            "passed": len(ground_truth["frames"]) >= min_annotated_frames,
        },
        "overall_micro_f1": {
            "value": overall["micro"]["f1"],
            "minimum": min_overall_micro_f1,
            "passed": overall["micro"]["f1"] >= min_overall_micro_f1,
        },
        "searchable_entity_micro_f1": {
            "value": overall["searchable_entities_micro"]["f1"],
            "minimum": min_searchable_micro_f1,
            "passed": overall["searchable_entities_micro"]["f1"]
            >= min_searchable_micro_f1,
        },
        "object_recall": {
            "value": field_metrics["objects"]["micro"]["recall"],
            "minimum": min_object_recall,
            "passed": field_metrics["objects"]["micro"]["recall"]
            >= min_object_recall,
        },
    }
    warnings = []
    weak_fields = [
        field
        for field in ENTITY_FIELDS
        if field_metrics[field]["micro"]["f1"] < 0.5
    ]
    if weak_fields:
        warnings.append(
            "Entity fields below 0.50 visual-truth F1: " + ", ".join(weak_fields)
        )
    if overall["searchable_entities_micro"]["f1"] < 0.5:
        warnings.append(
            "Searchable set fields are below 0.50 micro-F1; all-fields micro-F1 "
            "is higher because scalar unknown classifications are easy negatives."
        )
    negative_fp = sum(
        field_metrics[field]
        .get("negative_handling", {})
        .get("false_positive_negative_frames", 0)
        for field in SET_FIELDS
    )
    if negative_fp:
        warnings.append(
            f"Predictions added entities to {negative_fp} known-negative frame/field cases."
        )

    error_analysis: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for field in ENTITY_FIELDS:
        false_positives: Counter[str] = Counter()
        false_negatives: Counter[str] = Counter()
        for frame_report in frame_reports:
            field_report = frame_report["fields"][field]
            if field_report["skipped"]:
                continue
            false_positives.update(
                _normalise(value) for value in field_report["false_positives"]
            )
            false_negatives.update(
                _normalise(value) for value in field_report["false_negatives"]
            )
        error_analysis[field] = {
            "top_false_positives": [
                {"value": value, "count": count}
                for value, count in false_positives.most_common(10)
            ],
            "top_false_negatives": [
                {"value": value, "count": count}
                for value, count in false_negatives.most_common(10)
            ],
        }

    return {
        "ground_truth_version": ground_truth["version"],
        "annotation_method": ground_truth.get("annotation_method"),
        "annotated_frames": len(ground_truth["frames"]),
        "metrics": {"fields": field_metrics, "overall": overall},
        "error_analysis": error_analysis,
        "acceptance": {
            "passed": all(check["passed"] for check in checks.values()),
            "checks": checks,
            "warnings": warnings,
        },
        "frames": frame_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark structured entities against visual ground truth."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-annotated-frames", type=int, default=10)
    parser.add_argument("--min-overall-micro-f1", type=float, default=0.5)
    parser.add_argument("--min-searchable-micro-f1", type=float, default=0.4)
    parser.add_argument("--min-object-recall", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        ground_truth = load_entity_ground_truth(args.ground_truth)
        records = validate_metadata_file(args.metadata)
        report = evaluate_entity_records(
            records,
            ground_truth,
            min_annotated_frames=args.min_annotated_frames,
            min_overall_micro_f1=args.min_overall_micro_f1,
            min_searchable_micro_f1=args.min_searchable_micro_f1,
            min_object_recall=args.min_object_recall,
        )
        report.update(
            {
                "metadata_path": str(args.metadata.resolve()),
                "ground_truth_path": str(args.ground_truth.resolve()),
            }
        )
        write_json_atomically(args.output, report, overwrite=True)
    except Exception as exc:
        parser.exit(status=1, message=f"Entity benchmark failed: {exc}\n")

    metrics = report["metrics"]
    print(
        "Entity visual benchmark | "
        f"frames={report['annotated_frames']} | "
        f"micro_f1={metrics['overall']['micro']['f1']:.4f} | "
        f"searchable_micro_f1="
        f"{metrics['overall']['searchable_entities_micro']['f1']:.4f} | "
        f"macro_field_f1={metrics['overall']['macro_by_field']['f1']:.4f} | "
        f"object_precision={metrics['fields']['objects']['micro']['precision']:.4f} | "
        f"object_recall={metrics['fields']['objects']['micro']['recall']:.4f}"
    )
    for warning in report["acceptance"]["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved report to {args.output.resolve()}")
    if not report["acceptance"]["passed"]:
        parser.exit(status=2, message="Entity benchmark acceptance failed.\n")


if __name__ == "__main__":
    main()
