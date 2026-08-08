"""Fuse BTC identity, objects, media info, and optional visual text into schema v1.1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from common import paths
    from common.frame_ref import canonical_keyframe_map, frame_id, load_keyframe_map
except ImportError:  # pragma: no cover
    from src.common import paths
    from src.common.frame_ref import canonical_keyframe_map, frame_id, load_keyframe_map

try:
    from btc_objects import BTC_DETECTION_MODEL, load_frame_detections, object_counts
    from migrate_metadata import write_json_atomically
    from object_ontology import ObjectOntology, get_default_ontology
    from schemas import FrameEntities, FrameMetadata, validate_metadata_records
    from spatial_extractor import (
        LABEL_GROUNDING_VERSION,
        SPATIAL_RULE_VERSION,
        ground_detection_labels,
        infer_spatial_relations,
    )
except ImportError:
    from .btc_objects import BTC_DETECTION_MODEL, load_frame_detections, object_counts
    from .migrate_metadata import write_json_atomically
    from .object_ontology import ObjectOntology, get_default_ontology
    from .schemas import FrameEntities, FrameMetadata, validate_metadata_records
    from .spatial_extractor import (
        LABEL_GROUNDING_VERSION,
        SPATIAL_RULE_VERSION,
        ground_detection_labels,
        infer_spatial_relations,
    )


DESCRIPTION_LIMIT = 500


def _compact_description(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split("►", 1)[0].split())[:DESCRIPTION_LIMIT]


def load_media_info(video_id: str) -> dict:
    source = paths.media_info(video_id)
    if not source.is_file():
        return {"title": "", "description": "", "keywords": []}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read BTC media-info for {video_id}: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BTC media-info must contain an object: {source}")
    keywords = payload.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "title": str(payload.get("title") or "").strip(),
        "description": _compact_description(payload.get("description")),
        "keywords": [
            str(value).strip() for value in keywords if str(value).strip()
        ],
    }


def object_projection(detections, ontology: ObjectOntology) -> tuple[str, list[str], dict[str, int]]:
    counts = object_counts(detections)
    labels = set(counts)
    ancestor_labels: set[str] = set()
    expanded_terms: set[str] = set()
    by_label_cells: dict[str, list[str]] = {}
    for detection in detections:
        by_label_cells.setdefault(detection.label, []).append(detection.grid_cell or "center")
        if detection.mid:
            expanded = ontology.labels_for_index(detection.mid)
            labels.update(expanded)
            ancestor_labels.update(expanded - {detection.label})
            expanded_terms.update(ontology.expand_for_index(detection.mid))

    phrases: list[str] = []
    for label, count in counts.items():
        phrase = f"{count} {label}"
        cells = sorted(set(by_label_cells[label]))
        if count == 1 and len(cells) == 1:
            phrase += f" at {cells[0]}"
        phrases.append(phrase)
    if ancestor_labels:
        phrases.append("categories " + ", ".join(sorted(ancestor_labels)))
    compact_terms = expanded_terms - labels - {
        "entity",
        "physical entity",
        "thing",
        "object",
    }
    if compact_terms:
        phrases.append("aliases " + ", ".join(sorted(compact_terms)))
    return ", ".join(phrases), sorted(labels), counts


def _visual_index(records: Iterable[FrameMetadata]) -> dict[tuple[str, int], FrameMetadata]:
    result: dict[tuple[str, int], FrameMetadata] = {}
    for record in records:
        keyframe_number = record.keyframe_n or record.frame_index
        result[(Path(record.video_name).stem, keyframe_number)] = record
    return result


def build_video_records(
    video_id: str,
    *,
    visual_records: Iterable[FrameMetadata] = (),
    ontology: ObjectOntology | None = None,
    prefer_parquet: bool = True,
    with_spatial: bool = True,
) -> list[FrameMetadata]:
    ontology = ontology or get_default_ontology()
    refs = load_keyframe_map(video_id)
    source_files = sorted(paths.objects_dir(video_id).glob("*.json"))
    if not paths.objects_index(video_id).is_file() and len(source_files) != len(refs):
        raise ValueError(
            f"BTC join mismatch for {video_id}: {len(refs)} map rows but "
            f"{len(source_files)} object files"
        )
    media = load_media_info(video_id)
    visual_by_key = _visual_index(visual_records)
    records: list[FrameMetadata] = []

    for keyframe_n, ref in canonical_keyframe_map(video_id).items():
        visual = visual_by_key.get((video_id, keyframe_n))
        detections = load_frame_detections(
            video_id, keyframe_n, prefer_parquet=prefer_parquet
        )
        object_text, object_labels, counts = object_projection(detections, ontology)
        base_entities = visual.entities if visual is not None else FrameEntities()
        primary_labels = sorted({item.label for item in detections})
        entities = base_entities.model_copy(update={"objects": primary_labels})
        caption = visual.caption if visual is not None else ""
        ocr_text = visual.ocr_text if visual is not None else ""
        ocr_text_raw = visual.ocr_text_raw if visual is not None else ""
        has_visual_text = bool(caption.strip() or ocr_text.strip() or ocr_text_raw.strip())
        processing = (
            visual.processing.model_copy(deep=True)
            if visual is not None
            else FrameMetadata.model_construct(
                frame_id="unused", video_name="unused", frame_index=0
            ).processing
        )
        processing = processing.model_copy(
            update={
                "detection_model": BTC_DETECTION_MODEL,
                "label_grounding_version": LABEL_GROUNDING_VERSION,
                "spatial_rule_version": SPATIAL_RULE_VERSION if with_spatial else None,
                "spatial_processed_at": (
                    datetime.now(timezone.utc) if with_spatial else None
                ),
            }
        )
        raw_record = {
            "schema_version": "1.1",
            "frame_id": frame_id(ref),
            "video_name": video_id,
            "frame_index": ref.frame_idx,
            "keyframe_n": keyframe_n,
            "timestamp_ms": ref.timestamp_ms,
            "caption": caption,
            "ocr_text": ocr_text,
            "ocr_text_raw": ocr_text_raw,
            "video_title": media["title"],
            "video_description": media["description"],
            "video_keywords": media["keywords"],
            "object_text": object_text,
            "object_labels": object_labels,
            "object_counts": counts,
            "has_visual_text": has_visual_text,
            "entities": entities.model_dump(mode="json"),
            "code": (
                visual.code.model_dump(mode="json") if visual is not None else {}
            ),
            "detections": [item.model_dump(mode="json") for item in detections],
            "spatial_relations": [],
            "processing": processing.model_dump(mode="json"),
        }
        record = FrameMetadata.model_validate(raw_record)
        if with_spatial:
            # LLM/visual entities are independent evidence. Do not ground a BTC
            # label using ``entities.objects`` that was itself copied from BTC.
            grounding_record = record.model_copy(update={"entities": base_entities})
            grounded, _ = ground_detection_labels(
                record.detections, grounding_record
            )
            relations = infer_spatial_relations(grounded)
            grounded_text, grounded_labels, grounded_counts = object_projection(
                grounded, ontology
            )
            grounded_entities = base_entities.model_copy(
                update={"objects": sorted({item.label for item in grounded})}
            )
            record = FrameMetadata.model_validate(
                {
                    **record.model_dump(mode="json"),
                    "entities": grounded_entities.model_dump(mode="json"),
                    "object_text": grounded_text,
                    "object_labels": grounded_labels,
                    "object_counts": grounded_counts,
                    "detections": [item.model_dump(mode="json") for item in grounded],
                    "spatial_relations": [
                        item.model_dump(mode="json") for item in relations
                    ],
                }
            )
        records.append(record)
    return records


def _load_visual_records(source: Path | None) -> list[FrameMetadata]:
    if source is None:
        return []
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    records: list[FrameMetadata] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(validate_metadata_records(payload))
    return records


def _select_videos(value: str) -> list[str]:
    available = paths.video_ids()
    requested = [item.strip() for item in value.split(",") if item.strip()]
    selected: list[str] = []
    for item in requested:
        if "_V" in item:
            selected.append(item)
        else:
            selected.extend(video for video in available if video.startswith(f"{item}_"))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown BTC videos: {unknown}")
    return list(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        required=True,
        help="Comma-separated BTC video IDs or groups, e.g. L21,L22",
    )
    parser.add_argument("--visual-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, default=paths.processed_metadata_dir())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        videos = _select_videos(args.videos)
        visual_records = _load_visual_records(args.visual_metadata)
        ontology = get_default_ontology()
    except Exception as exc:
        parser.exit(2, f"Metadata setup failed: {exc}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    for position, video_id in enumerate(videos, start=1):
        output = args.output_dir / f"{video_id}.json"
        if args.resume and output.is_file():
            summary["skipped_videos"] += 1
            continue
        records = build_video_records(
            video_id,
            visual_records=visual_records,
            ontology=ontology,
            prefer_parquet=not args.no_parquet,
            with_spatial=not args.no_spatial,
        )
        validated = validate_metadata_records(
            [record.model_dump(mode="json") for record in records]
        )
        write_json_atomically(
            output,
            [record.model_dump(mode="json") for record in validated],
            overwrite=True,
        )
        summary["videos"] += 1
        summary["frames"] += len(records)
        summary["deduplicated_keyframes"] += (
            len(load_keyframe_map(video_id)) - len(records)
        )
        summary["detections"] += sum(len(record.detections) for record in records)
        summary["relations"] += sum(
            len(record.spatial_relations) for record in records
        )
        print(f"[{position}/{len(videos)}] wrote {len(records)} frames -> {output}")
    print(json.dumps(dict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
