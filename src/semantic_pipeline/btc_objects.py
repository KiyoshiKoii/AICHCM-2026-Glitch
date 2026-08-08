"""Adapter for the BTC Open Images V4 object-detection files.

The source JSON contains 100 padded predictions per frame.  This module applies
the production filter in the required order (score, area, class-wise NMS, top
K), converts BTC's YXYX boxes to normalized XYXY, and preserves the Open Images
MID so labels can be joined to the ontology without relying on display names.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from common import paths
except ImportError:  # pragma: no cover - direct module execution fallback
    from src.common import paths

try:
    from schemas import Detection
    from spatial_extractor import RawDetection, bbox_iou, normalise_detections
except ImportError:
    from .schemas import Detection
    from .spatial_extractor import RawDetection, bbox_iou, normalise_detections


MIN_DETECTION_SCORE = 0.20
MIN_BOX_AREA = 0.0001
NMS_IOU_THRESHOLD = 0.60
MAX_DETECTIONS = 15
BTC_DETECTION_MODEL = "btc-openimages-v4"


def _grid_cell(bbox: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bbox
    column = min(2, int(((x1 + x2) / 2) * 3))
    row = min(2, int(((y1 + y2) / 2) * 3))
    columns = ("left", "center", "right")
    rows = ("top", "center", "bottom")
    if row == 1 and column == 1:
        return "center"
    return f"{rows[row]}-{columns[column]}"


def _parse_parallel_arrays(payload: dict[str, Any], source: Path) -> list[RawDetection]:
    field_names = (
        "detection_scores",
        "detection_class_names",
        "detection_class_entities",
        "detection_boxes",
    )
    arrays = [payload.get(name) for name in field_names]
    if not all(isinstance(value, list) for value in arrays):
        raise ValueError(f"{source} is missing BTC detection arrays")
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"BTC detection arrays have different lengths in {source}: "
            f"{dict(zip(field_names, map(len, arrays)))}"
        )

    raw: list[RawDetection] = []
    scores, mids, labels, boxes = arrays
    for position, (score_value, mid, label, box) in enumerate(
        zip(scores, mids, labels, boxes)
    ):
        try:
            score = float(score_value)
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("box must have four coordinates")
            ymin, xmin, ymax, xmax = (float(value) for value in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid BTC detection at {source} position {position}"
            ) from exc
        raw.append(
            RawDetection(
                label=str(label),
                bbox=(xmin, ymin, xmax, ymax),
                confidence=score,
                mid=str(mid),
                label_source="btc_detector",
            )
        )
    return raw


def filter_btc_detections(
    raw: list[RawDetection],
    *,
    min_score: float = MIN_DETECTION_SCORE,
    min_box_area: float = MIN_BOX_AREA,
    nms_iou: float = NMS_IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> list[Detection]:
    """Filter already-normalized BTC detections before alphabetical normalization."""
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    if not 0 <= min_box_area < 1:
        raise ValueError("min_box_area must be in [0, 1)")
    if not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou must be between 0 and 1")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")

    candidates: list[RawDetection] = []
    for item in raw:
        if item.confidence is None or item.confidence < min_score:
            continue
        x1, y1, x2, y2 = item.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            continue
        if (x2 - x1) * (y2 - y1) < min_box_area:
            continue
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            -(item.confidence or 0.0),
            item.mid or item.label.casefold(),
            item.bbox,
        )
    )

    kept: list[RawDetection] = []
    for candidate in candidates:
        class_key = candidate.mid or candidate.label.casefold()
        if any(
            (previous.mid or previous.label.casefold()) == class_key
            and bbox_iou(previous.bbox, candidate.bbox) >= nms_iou
            for previous in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break

    normalized = normalise_detections(
        kept,
        image_width=1,
        image_height=1,
        min_box_area=min_box_area,
        duplicate_iou=1.0,
        max_detections=max_detections,
    )
    return [
        detection.model_copy(
            update={
                "area": round(
                    (detection.bbox[2] - detection.bbox[0])
                    * (detection.bbox[3] - detection.bbox[1]),
                    8,
                ),
                "grid_cell": _grid_cell(detection.bbox),
            }
        )
        for detection in normalized
    ]


def load_btc_object_file(
    source: str | Path,
    **filter_options: Any,
) -> tuple[list[Detection], dict[str, int]]:
    source = Path(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read BTC object JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BTC object JSON must contain an object: {source}")
    raw = _parse_parallel_arrays(payload, source)
    above_threshold = sum(
        item.confidence is not None
        and item.confidence >= filter_options.get("min_score", MIN_DETECTION_SCORE)
        for item in raw
    )
    detections = filter_btc_detections(raw, **filter_options)
    return detections, {
        "raw": len(raw),
        "above_threshold": above_threshold,
        "filtered": len(detections),
    }


def object_counts(detections: list[Detection]) -> dict[str, int]:
    return dict(sorted(Counter(item.label for item in detections).items()))


def _detection_to_parquet(item: Detection) -> dict[str, Any]:
    return {
        "object_id": item.object_id,
        "label": item.label,
        "mid": item.mid,
        "bbox": list(item.bbox),
        "confidence": item.confidence,
        "grid_cell": item.grid_cell,
        "area": item.area,
        "label_source": item.label_source,
    }


def _detection_from_parquet(item: dict[str, Any]) -> Detection:
    return Detection.model_validate(
        {
            "object_id": item["object_id"],
            "label": item["label"],
            "mid": item.get("mid"),
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "grid_cell": item.get("grid_cell"),
            "area": item.get("area"),
            "label_source": item.get("label_source") or "btc_detector",
        }
    )


def build_video_objects_index(
    video_id: str,
    output_path: str | Path | None = None,
    **filter_options: Any,
) -> dict[str, Any]:
    """Merge one video's small JSON files into one native nested Parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pyarrow is required to build the BTC objects index") from exc

    source_dir = paths.objects_dir(video_id)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing BTC objects directory: {source_dir}")
    files = sorted(source_dir.glob("*.json"), key=lambda item: int(item.stem))
    if not files:
        raise ValueError(f"BTC objects directory is empty: {source_dir}")

    rows: list[dict[str, Any]] = []
    totals = Counter()
    for position, source in enumerate(files, start=1):
        keyframe_n = int(source.stem)
        if keyframe_n != position:
            raise ValueError(
                f"BTC object files must be contiguous from 001 in {source_dir}; "
                f"expected {position:03d}.json, found {source.name}"
            )
        detections, stats = load_btc_object_file(source, **filter_options)
        totals.update(stats)
        rows.append(
            {
                "video_id": video_id,
                "keyframe_n": keyframe_n,
                "raw_detection_count": stats["raw"],
                "above_threshold_count": stats["above_threshold"],
                "detections": [_detection_to_parquet(item) for item in detections],
            }
        )

    detection_struct = pa.struct(
        [
            ("object_id", pa.string()),
            ("label", pa.string()),
            ("mid", pa.string()),
            ("bbox", pa.list_(pa.float32(), 4)),
            ("confidence", pa.float32()),
            ("grid_cell", pa.string()),
            ("area", pa.float32()),
            ("label_source", pa.string()),
        ]
    )
    schema = pa.schema(
        [
            ("video_id", pa.string()),
            ("keyframe_n", pa.int32()),
            ("raw_detection_count", pa.int16()),
            ("above_threshold_count", pa.int16()),
            ("detections", pa.list_(detection_struct)),
        ]
    )
    output = Path(output_path) if output_path is not None else paths.objects_index(video_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    load_video_objects_index.cache_clear()
    frames = len(rows)
    return {
        "video_id": video_id,
        "frames": frames,
        "raw_per_frame": totals["raw"] / frames,
        "above_threshold_per_frame": totals["above_threshold"] / frames,
        "filtered_per_frame": totals["filtered"] / frames,
        "output": str(output),
    }


@lru_cache(maxsize=64)
def load_video_objects_index(video_id: str) -> dict[int, list[Detection]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the BTC objects index") from exc
    source = paths.objects_index(video_id)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing BTC objects Parquet index for {video_id}: {source}; "
            "run scripts/build_objects_index.py"
        )
    rows = pq.read_table(source, columns=["keyframe_n", "detections"]).to_pylist()
    return {
        int(row["keyframe_n"]): [
            _detection_from_parquet(item) for item in row["detections"]
        ]
        for row in rows
    }


def load_frame_detections(
    video_id: str,
    keyframe_n: int,
    *,
    prefer_parquet: bool = True,
) -> list[Detection]:
    if prefer_parquet and paths.objects_index(video_id).is_file():
        try:
            return load_video_objects_index(video_id)[keyframe_n]
        except KeyError as exc:
            raise KeyError(
                f"Missing keyframe n={keyframe_n} in objects index for {video_id}"
            ) from exc
    source = paths.objects(video_id, keyframe_n)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing BTC object file for {video_id} keyframe {keyframe_n}: {source}"
        )
    return load_btc_object_file(source)[0]
