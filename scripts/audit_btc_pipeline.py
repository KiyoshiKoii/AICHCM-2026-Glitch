"""Audit attached BTC source data and generated semantic metadata.

The audit deliberately separates required object-pipeline inputs from optional
keyframe images.  It can scan every raw BTC object JSON, then cross-check the
processed Parquet/metadata artifacts for the groups currently generated.

Examples:
    python scripts/audit_btc_pipeline.py
    python scripts/audit_btc_pipeline.py --raw-videos L21,L22
    python scripts/audit_btc_pipeline.py --processed-videos L21,L22 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import paths  # noqa: E402
from common.frame_ref import (  # noqa: E402
    canonical_keyframe_map,
    frame_id,
    load_keyframe_map,
    resolve,
)
from semantic_pipeline.btc_objects import (  # noqa: E402
    MAX_DETECTIONS,
    MIN_DETECTION_SCORE,
    load_video_objects_index,
)
from semantic_pipeline.elasticsearch_backend import (  # noqa: E402
    collapse_spatial_relations_for_index,
)
from semantic_pipeline.metadata_builder import load_media_info  # noqa: E402
from semantic_pipeline.object_ontology import ObjectOntology  # noqa: E402
from semantic_pipeline.schemas import validate_metadata_file  # noqa: E402


DEFAULT_QUERY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "semantic_pipeline"
    / "evaluation_queries_btc.json"
)
MAX_REPORTED_ERRORS = 100


def select_videos(value: str | None, available: list[str]) -> list[str]:
    if value is None or value.strip().casefold() == "all":
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    selected: list[str] = []
    for item in requested:
        if "_V" in item:
            selected.append(item)
        else:
            selected.extend(
                video for video in available if video.startswith(f"{item}_")
            )
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown BTC videos: {unknown}")
    return selected


def _audit_detection_payload(payload: Any, source: Path) -> tuple[Counter, list[str]]:
    counts: Counter = Counter(files=1)
    errors: list[str] = []
    if not isinstance(payload, dict):
        return counts, [f"{source}: root must be a JSON object"]

    names = (
        "detection_scores",
        "detection_class_names",
        "detection_class_entities",
        "detection_boxes",
    )
    arrays = [payload.get(name) for name in names]
    if not all(isinstance(value, list) for value in arrays):
        return counts, [f"{source}: missing detection arrays"]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        return counts, [f"{source}: parallel arrays have different lengths"]

    scores, mids, labels, boxes = arrays
    counts["detections"] += len(scores)
    if len(scores) != 100:
        counts["non_100_frames"] += 1
        errors.append(
            f"{source}: expected 100 padded detections, found {len(scores)}"
        )
    for position, (raw_score, mid, label, box) in enumerate(
        zip(scores, mids, labels, boxes)
    ):
        try:
            score = float(raw_score)
            if not 0 <= score <= 1:
                raise ValueError("score outside [0,1]")
            if not isinstance(mid, str) or not mid.startswith("/m/"):
                raise ValueError("invalid Open Images MID")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("empty class label")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("box must have four values")
            ymin, xmin, ymax, xmax = (float(value) for value in box)
            if not (0 <= ymin < ymax <= 1 and 0 <= xmin < xmax <= 1):
                raise ValueError("invalid normalized YXYX box")
        except (TypeError, ValueError) as exc:
            errors.append(f"{source}[{position}]: {exc}")
            if len(errors) >= 5:
                break
            continue
        if score >= MIN_DETECTION_SCORE:
            counts["above_threshold"] += 1
    return counts, errors


def audit_raw_video(video_id: str) -> dict[str, Any]:
    counts: Counter = Counter(videos=1)
    errors: list[str] = []
    try:
        refs = load_keyframe_map(video_id)
    except Exception as exc:
        return {"video_id": video_id, "counts": dict(counts), "errors": [str(exc)]}

    counts["map_rows"] = len(refs)
    counts["duplicate_frame_indices"] = len(refs) - len(
        {ref.frame_idx for ref in refs.values()}
    )
    object_dir = paths.objects_dir(video_id)
    try:
        object_files = sorted(object_dir.glob("*.json"), key=lambda item: int(item.stem))
    except ValueError:
        object_files = []
        errors.append(f"{object_dir}: object filenames must be numeric")
    counts["object_files"] = len(object_files)
    expected_names = [f"{number:03d}.json" for number in range(1, len(refs) + 1)]
    actual_names = [source.name for source in object_files]
    if actual_names != expected_names:
        errors.append(
            f"{video_id}: map/object join mismatch "
            f"({len(refs)} rows vs {len(object_files)} files)"
        )

    counts["has_media_info"] = int(paths.media_info(video_id).is_file())
    counts["has_clip_features"] = int(paths.clip_features(video_id).is_file())
    if not counts["has_media_info"]:
        errors.append(f"{video_id}: missing media-info")
    if not counts["has_clip_features"]:
        errors.append(f"{video_id}: missing clip features")

    keyframe_dir = paths.keyframe_dir(video_id)
    keyframe_count = (
        sum(1 for source in keyframe_dir.glob("*.jpg"))
        if keyframe_dir.is_dir()
        else 0
    )
    counts["keyframe_images"] = keyframe_count
    counts["videos_with_keyframes"] = int(keyframe_count > 0)
    if keyframe_count and keyframe_count != len(refs):
        errors.append(
            f"{video_id}: partial keyframes ({keyframe_count}/{len(refs)})"
        )

    for source in object_files:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{source}: cannot parse JSON: {exc}")
            continue
        payload_counts, payload_errors = _audit_detection_payload(payload, source)
        counts.update(payload_counts)
        errors.extend(payload_errors)
    return {"video_id": video_id, "counts": dict(counts), "errors": errors}


def audit_raw(videos: list[str], workers: int) -> dict[str, Any]:
    totals: Counter = Counter()
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for position, result in enumerate(executor.map(audit_raw_video, videos), start=1):
            totals.update(result["counts"])
            errors.extend(result["errors"])
            if position % 100 == 0 or position == len(videos):
                print(f"raw audit: {position}/{len(videos)} videos")
    frame_count = totals["object_files"]
    return {
        **dict(totals),
        "raw_detections_per_frame": (
            totals["detections"] / frame_count if frame_count else 0.0
        ),
        "above_threshold_per_frame": (
            totals["above_threshold"] / frame_count if frame_count else 0.0
        ),
        "errors": errors[:MAX_REPORTED_ERRORS],
        "error_count": len(errors),
        "passed": not errors and bool(videos),
    }


def _same_detection(source: Any, generated: Any) -> bool:
    return (
        source.object_id == generated.object_id
        and source.mid == generated.mid
        and abs(source.confidence - generated.confidence) <= 1e-5
        and all(
            abs(left - right) <= 1e-5
            for left, right in zip(source.bbox, generated.bbox)
        )
    )


def audit_processed(
    videos: list[str],
    ontology: ObjectOntology,
    wanted_frame_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    totals: Counter = Counter()
    errors: list[str] = []
    selected_records: dict[str, Any] = {}
    distinct_mids: set[str] = set()
    lexicon_occurrences = 0

    for position, video_id in enumerate(videos, start=1):
        metadata_path = paths.processed_metadata_dir() / f"{video_id}.json"
        parquet_path = paths.objects_index(video_id)
        if not metadata_path.is_file():
            errors.append(f"{video_id}: missing generated metadata")
            continue
        if not parquet_path.is_file():
            errors.append(f"{video_id}: missing generated Parquet object index")
            continue
        try:
            records = validate_metadata_file(metadata_path)
            indexed = load_video_objects_index(video_id)
            canonical = canonical_keyframe_map(video_id)
        except Exception as exc:
            errors.append(f"{video_id}: cannot load processed artifacts: {exc}")
            continue
        if len(indexed) != len(load_keyframe_map(video_id)):
            errors.append(f"{video_id}: Parquet does not cover every map row")
        if len(records) != len(canonical):
            errors.append(
                f"{video_id}: metadata has {len(records)} records, "
                f"expected {len(canonical)} canonical records"
            )

        media = load_media_info(video_id)
        seen_ids: set[str] = set()
        for record in records:
            totals["records"] += 1
            if record.frame_id in seen_ids:
                errors.append(f"{video_id}: duplicate metadata ID {record.frame_id}")
            seen_ids.add(record.frame_id)
            try:
                ref = resolve(video_id, record.keyframe_n or -1)
            except Exception as exc:
                errors.append(f"{record.frame_id}: invalid keyframe_n: {exc}")
                continue
            if (
                record.frame_id != frame_id(ref)
                or record.frame_index != ref.frame_idx
                or record.timestamp_ms != ref.timestamp_ms
            ):
                errors.append(f"{record.frame_id}: BTC identity mismatch")
            expected = indexed.get(ref.keyframe_n)
            if expected is None or len(expected) != len(record.detections):
                errors.append(f"{record.frame_id}: Parquet/metadata detection count mismatch")
            elif not all(
                _same_detection(left, right)
                for left, right in zip(expected, record.detections)
            ):
                errors.append(f"{record.frame_id}: Parquet/metadata detection mismatch")

            labels = {item.label.casefold() for item in record.detections}
            if labels != {item.casefold() for item in record.entities.objects}:
                errors.append(f"{record.frame_id}: entities.objects is not detector-derived")
            if sum(record.object_counts.values()) != len(record.detections):
                errors.append(f"{record.frame_id}: object_counts mismatch")
            if record.detections and not record.object_text.strip():
                errors.append(f"{record.frame_id}: object_text is empty")
            if len(record.detections) > MAX_DETECTIONS or any(
                item.confidence < MIN_DETECTION_SCORE
                or not item.mid
                or item.label_source not in {"btc_detector", "context_grounding"}
                for item in record.detections
            ):
                errors.append(f"{record.frame_id}: invalid filtered detection")
            if record.has_visual_text != bool(
                record.caption.strip()
                or record.ocr_text.strip()
                or record.ocr_text_raw.strip()
            ):
                errors.append(f"{record.frame_id}: has_visual_text mismatch")
            if (
                record.video_title != media["title"]
                or record.video_description != media["description"]
                or record.video_keywords
                != list(
                    dict.fromkeys(
                        value.strip().casefold() for value in media["keywords"]
                    )
                )
            ):
                errors.append(f"{record.frame_id}: media-info projection mismatch")

            for detection in record.detections:
                totals["detections"] += 1
                totals["lexicon_detection_occurrences"] += int(
                    detection.mid in ontology.lexicon
                )
                distinct_mids.add(detection.mid)
                lexicon_occurrences += int(detection.mid in ontology.lexicon)
                expected_labels = ontology.labels_for_index(detection.mid)
                if not expected_labels.issubset(set(record.object_labels)):
                    errors.append(f"{record.frame_id}: missing ontology object label")
            totals["raw_relations"] += len(record.spatial_relations)
            totals["collapsed_relations"] += len(
                collapse_spatial_relations_for_index(record.spatial_relations)
            )
            if record.frame_id in wanted_frame_ids:
                selected_records[record.frame_id] = record
        totals["videos"] += 1
        totals["deduplicated_keyframes"] += len(load_keyframe_map(video_id)) - len(
            canonical
        )
        if position % 10 == 0 or position == len(videos):
            print(f"processed audit: {position}/{len(videos)} videos")

    records = totals["records"]
    raw_relations = totals["raw_relations"]
    report = {
        **dict(totals),
        "detections_per_frame": totals["detections"] / records if records else 0.0,
        "raw_relations_per_frame": raw_relations / records if records else 0.0,
        "collapsed_relations_per_frame": (
            totals["collapsed_relations"] / records if records else 0.0
        ),
        "relation_reduction_percent": (
            100 * (raw_relations - totals["collapsed_relations"]) / raw_relations
            if raw_relations
            else 0.0
        ),
        "distinct_mids": len(distinct_mids),
        "lexicon_distinct_mid_coverage": (
            len(distinct_mids & set(ontology.lexicon)) / len(distinct_mids)
            if distinct_mids
            else 0.0
        ),
        "lexicon_detection_occurrence_coverage": (
            lexicon_occurrences / totals["detections"]
            if totals["detections"]
            else 0.0
        ),
        "errors": errors[:MAX_REPORTED_ERRORS],
        "error_count": len(errors),
        "passed": not errors and bool(videos),
    }
    return report, selected_records


def audit_queries(
    query_path: Path,
    records: dict[str, Any],
    ontology: ObjectOntology,
) -> dict[str, Any]:
    cases = json.loads(query_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    visually_checked = 0
    evaluated = 0
    skipped = 0
    for case in cases:
        relevant = case.get("relevant_frame_ids", [])
        if not relevant:
            errors.append(f"{case.get('query_id')}: no relevant frames")
            continue
        available_relevant = [
            relevant_id for relevant_id in relevant if relevant_id in records
        ]
        if not available_relevant:
            skipped += 1
            continue
        evaluated += 1
        case_passed = False
        for relevant_id in available_relevant:
            record = records.get(relevant_id)
            detections = record.detections
            object_terms_ok = True
            for term in case.get("filters", {}).get("objects", []):
                mids, unmapped = ontology.map_query_terms([term])
                if mids:
                    matched = any(
                        bool({item.mid, *ontology.ancestors(item.mid)} & mids)
                        for item in detections
                        if item.mid
                    )
                elif unmapped:
                    matched = any(
                        item.label.casefold() == unmapped[0].casefold()
                        for item in detections
                    )
                else:
                    matched = False
                object_terms_ok = object_terms_ok and matched
            case_passed = case_passed or object_terms_ok
        if not case_passed:
            errors.append(
                f"{case.get('query_id')}: relevant frame does not satisfy filters"
            )
        if "visually_checked" in case.get("tags", []):
            visually_checked += 1
    return {
        "cases": len(cases),
        "evaluated_cases": evaluated,
        "skipped_out_of_scope_cases": skipped,
        "visually_checked_cases": visually_checked,
        "errors": errors[:MAX_REPORTED_ERRORS],
        "error_count": len(errors),
        "passed": not errors and bool(cases),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-videos",
        help="Raw video IDs/groups; default is every attached map (use 'all' explicitly too).",
    )
    parser.add_argument(
        "--processed-videos",
        help="Processed video IDs/groups; default is every generated metadata JSON.",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.reports_dir() / "btc_pipeline_audit.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    available = [source.stem for source in sorted(paths.MAP_KEYFRAMES.glob("*.csv"))]
    if not available:
        raise SystemExit(f"No BTC map-keyframes found under {paths.MAP_KEYFRAMES}")
    raw_videos = select_videos(args.raw_videos, available)
    generated = [
        source.stem
        for source in sorted(paths.processed_metadata_dir().glob("*.json"))
    ]
    processed_videos = (
        select_videos(args.processed_videos, available)
        if args.processed_videos
        else generated
    )
    unknown_generated = sorted(set(processed_videos) - set(generated))
    if unknown_generated:
        raise SystemExit(f"Missing generated metadata for {unknown_generated}")

    query_payload = json.loads(args.queries.read_text(encoding="utf-8"))
    wanted_frame_ids = {
        frame_id_
        for case in query_payload
        for frame_id_ in case.get("relevant_frame_ids", [])
    }
    ontology = ObjectOntology(require_external=True)
    raw_report = audit_raw(raw_videos, args.workers)
    processed_report, selected_records = audit_processed(
        processed_videos, ontology, wanted_frame_ids
    )
    query_report = audit_queries(args.queries, selected_records, ontology)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(paths.DATA_ROOT.resolve()),
        "raw": raw_report,
        "processed": processed_report,
        "queries": query_report,
        "passed": all(
            section["passed"]
            for section in (raw_report, processed_report, query_report)
        ),
    }
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "raw_videos": raw_report.get("videos", 0),
                "raw_files": raw_report.get("object_files", 0),
                "processed_videos": processed_report.get("videos", 0),
                "records": processed_report.get("records", 0),
                "raw_errors": raw_report["error_count"],
                "processed_errors": processed_report["error_count"],
                "query_errors": query_report["error_count"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
