"""Merge Gemini keyframe metadata with timestamped ASR without duplication.

Gemini metadata is keyed by the ordinal keyframe number (``..._f0001``),
whereas BTC submissions use the original video frame index.  The CSV in
``data/raw/map-keyframes`` is the authoritative bridge between those two
identities.  This module keeps the ordinal id as the internal document id so
it remains compatible with keyframe files and CLIP results, and persists the
native frame index separately for submission.

Each ASR segment is assigned exactly once: to the keyframe nearest to its
temporal midpoint.  This is equivalent to Voronoi windows between adjacent
keyframe timestamps, and avoids repeating a complete transcript in every
frame document.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:  # Support ``python src/semantic_pipeline/...``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_pipeline.core.compact_metadata import CompactVisualRecord
from semantic_pipeline.core.frame_id import parse_frame_id


DEFAULT_CAPTION_DIR = Path("data/metadata/caption")
DEFAULT_ASR_DIR = Path("data/metadata/metadata_asr")
DEFAULT_MAP_DIR = Path("data/raw/map-keyframes")
DEFAULT_OUTPUT_DIR = Path("data/processed/multimodal")
_UNK_TOKEN = re.compile(r"\bunk\b", re.IGNORECASE)
_ASR_REQUIRED_KEYS = frozenset({"video_name", "full_transcript", "segments"})


@dataclass(frozen=True)
class KeyframeMapRow:
    n: int
    pts_time: float
    fps: float
    frame_idx: int


@dataclass(frozen=True)
class ASRSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class MergeReport:
    video_id: str
    caption_frames: int
    mapped_frames: int
    asr_available: bool
    asr_segments: int
    assigned_asr_segments: int
    frames_with_asr: int
    asr_segments_with_unk: int


def _normalise_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _number(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} must be finite")
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def load_keyframe_map(path: str | Path) -> list[KeyframeMapRow]:
    """Read and validate one BTC keyframe map in chronological order."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Keyframe map does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"n", "pts_time", "fps", "frame_idx"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path}: expected CSV columns {sorted(required)}")
        rows: list[KeyframeMapRow] = []
        seen_n: set[int] = set()
        previous_time = -1.0
        previous_frame_index = -1
        for line_number, raw in enumerate(reader, 2):
            try:
                n = int(str(raw["n"]).strip())
                frame_idx = int(str(raw["frame_idx"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_number}: n/frame_idx must be integers") from exc
            pts_time = _number(raw["pts_time"], field="pts_time", source=csv_path)
            fps = _number(raw["fps"], field="fps", source=csv_path)
            if n < 1 or frame_idx < 0 or pts_time < 0 or fps <= 0:
                raise ValueError(f"{csv_path}:{line_number}: invalid keyframe map values")
            if n in seen_n:
                raise ValueError(f"{csv_path}:{line_number}: duplicate keyframe n={n}")
            if pts_time <= previous_time:
                raise ValueError(f"{csv_path}:{line_number}: pts_time must be strictly increasing")
            if frame_idx < previous_frame_index:
                raise ValueError(f"{csv_path}:{line_number}: frame_idx must not decrease")
            seen_n.add(n)
            previous_time = pts_time
            previous_frame_index = frame_idx
            rows.append(KeyframeMapRow(n, pts_time, fps, frame_idx))
    if not rows:
        raise ValueError(f"{csv_path}: keyframe map is empty")
    return rows


def load_asr_segments(path: str | Path, *, expected_video_id: str) -> list[ASRSegment]:
    """Read timestamped ASR segments, preserving their original indices."""

    json_path = Path(path)
    payload = _read_json(json_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{json_path}: ASR payload must be an object")
    missing = _ASR_REQUIRED_KEYS - set(payload)
    if missing:
        raise ValueError(f"{json_path}: ASR payload is missing {sorted(missing)}")
    if payload["video_name"] != expected_video_id:
        raise ValueError(
            f"{json_path}: video_name {payload['video_name']!r} does not match {expected_video_id!r}"
        )
    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError(f"{json_path}: segments must be an array")

    segments: list[ASRSegment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{json_path}: segment {index} must be an object")
        start = _number(raw.get("start"), field=f"segments[{index}].start", source=json_path)
        end = _number(raw.get("end"), field=f"segments[{index}].end", source=json_path)
        text = _normalise_text(raw.get("text"))
        if start < 0 or end < start:
            raise ValueError(f"{json_path}: segment {index} has invalid time range")
        if not text:
            raise ValueError(f"{json_path}: segment {index} has blank text")
        segments.append(ASRSegment(index=index, start=start, end=end, text=text))
    return sorted(segments, key=lambda item: (item.start, item.end, item.index))


def _caption_keyframe_number(raw_record: Mapping[str, Any], *, video_id: str) -> int:
    frame_id = raw_record.get("frame_id")
    if not isinstance(frame_id, str):
        raise ValueError(f"{video_id}: caption record has no valid frame_id")
    reference = parse_frame_id(frame_id)
    if reference.video_name != video_id:
        raise ValueError(f"{video_id}: caption frame_id belongs to {reference.video_name}")
    if reference.frame_index < 1:
        raise ValueError(f"{video_id}: caption frame_id must encode keyframe n >= 1: {frame_id}")
    return reference.frame_index


def _nearest_keyframe_position(timestamps: Sequence[float], midpoint: float) -> int:
    right = bisect.bisect_left(timestamps, midpoint)
    if right == 0:
        return 0
    if right == len(timestamps):
        return len(timestamps) - 1
    left = right - 1
    return left if midpoint - timestamps[left] <= timestamps[right] - midpoint else right


def merge_video_records(
    caption_records: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    keyframe_map: Sequence[KeyframeMapRow],
    asr_segments: Sequence[ASRSegment] = (),
    asr_available: bool,
) -> tuple[list[dict[str, Any]], MergeReport]:
    """Build compact per-keyframe records with one-to-one ASR assignment."""

    if not caption_records:
        raise ValueError(f"{video_id}: caption artifact is empty")
    map_by_n = {row.n: row for row in keyframe_map}
    normalised: list[tuple[int, dict[str, Any], KeyframeMapRow]] = []
    seen_n: set[int] = set()
    for position, raw in enumerate(caption_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{video_id}: caption record {position} must be an object")
        # Validation here guarantees the merged artifact stays ingestible by
        # the semantic pipeline and does not silently discard Gemini objects.
        visual_record = CompactVisualRecord.model_validate(raw)
        record = visual_record.model_dump(mode="json")
        n = _caption_keyframe_number(record, video_id=video_id)
        if n in seen_n:
            raise ValueError(f"{video_id}: duplicate caption keyframe n={n}")
        if n not in map_by_n:
            raise ValueError(f"{video_id}: map-keyframes has no row for caption keyframe n={n}")
        seen_n.add(n)
        normalised.append((n, record, map_by_n[n]))
    normalised.sort(key=lambda item: item[0])

    timestamps = [row.pts_time for _, _, row in normalised]
    assignments: dict[int, list[ASRSegment]] = defaultdict(list)
    for segment in asr_segments:
        assignments[normalised[_nearest_keyframe_position(timestamps, segment.midpoint)][0]].append(segment)

    merged: list[dict[str, Any]] = []
    frames_with_asr = 0
    for n, record, map_row in normalised:
        assigned = assignments[n]
        has_asr = bool(assigned)
        if has_asr:
            frames_with_asr += 1
        quality_flags = ["contains_unk"] if any(_UNK_TOKEN.search(item.text) for item in assigned) else []
        record.update(
            {
                "keyframe_n": n,
                "timestamp_ms": round(map_row.pts_time * 1000),
                "native_frame_index": map_row.frame_idx,
                "asr_available": asr_available,
                "has_asr": has_asr,
                "asr_text": " ".join(item.text for item in assigned),
                "asr_segment_indices": [item.index for item in assigned],
                "asr_start_ms": round(min(item.start for item in assigned) * 1000)
                if assigned
                else None,
                "asr_end_ms": round(max(item.end for item in assigned) * 1000)
                if assigned
                else None,
                "asr_quality_flags": quality_flags,
            }
        )
        # The visual portion was validated above.  The ASR/map fields are the
        # canonical merged extension and intentionally are not passed back
        # through the visual-only strict model (which forbids extra fields).
        merged.append(record)

    report = MergeReport(
        video_id=video_id,
        caption_frames=len(merged),
        mapped_frames=len(merged),
        asr_available=asr_available,
        asr_segments=len(asr_segments),
        assigned_asr_segments=sum(len(items) for items in assignments.values()),
        frames_with_asr=frames_with_asr,
        asr_segments_with_unk=sum(bool(_UNK_TOKEN.search(item.text)) for item in asr_segments),
    )
    if report.assigned_asr_segments != report.asr_segments:
        raise AssertionError("every ASR segment must be assigned exactly once")
    return merged, report


def merge_video_files(
    caption_path: str | Path,
    *,
    asr_dir: str | Path = DEFAULT_ASR_DIR,
    map_dir: str | Path = DEFAULT_MAP_DIR,
) -> tuple[list[dict[str, Any]], MergeReport]:
    """Merge one per-video Gemini checkpoint with its BTC map and ASR file."""

    source_path = Path(caption_path)
    video_id = source_path.stem
    payload = _read_json(source_path)
    if not isinstance(payload, list):
        raise ValueError(f"{source_path}: caption artifact must be a JSON array")
    keyframe_map = load_keyframe_map(Path(map_dir) / f"{video_id}.csv")
    asr_path = Path(asr_dir) / f"{video_id}.json"
    if asr_path.is_file():
        asr_segments = load_asr_segments(asr_path, expected_video_id=video_id)
        return merge_video_records(
            payload,
            video_id=video_id,
            keyframe_map=keyframe_map,
            asr_segments=asr_segments,
            asr_available=True,
        )
    return merge_video_records(
        payload,
        video_id=video_id,
        keyframe_map=keyframe_map,
        asr_available=False,
    )


def _caption_paths(root: Path, video_ids: Iterable[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Caption directory does not exist: {root}")
    paths = sorted(root.glob("L*/L*_V*.json"))
    if not paths:
        raise FileNotFoundError(f"No per-video caption JSON files found under {root}")
    requested = {video_id.upper() for video_id in video_ids}
    if not requested:
        return paths
    selected = [path for path in paths if path.stem.upper() in requested]
    missing = sorted(requested - {path.stem.upper() for path in selected})
    if missing:
        raise FileNotFoundError(f"No caption artifact found for: {', '.join(missing)}")
    return selected


def write_merged_artifact(
    records: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Write atomically, refusing to replace an existing artifact by default."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Gemini keyframe metadata with ASR by time.")
    parser.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTION_DIR)
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_DIR)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", action="append", default=[], help="Video id to merge; repeatable.")
    parser.add_argument("--check", action="store_true", help="Validate and report without writing output.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of a merged artifact.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reports: list[MergeReport] = []
        for caption_path in _caption_paths(args.caption_dir, args.video_id):
            records, report = merge_video_files(
                caption_path,
                asr_dir=args.asr_dir,
                map_dir=args.map_dir,
            )
            reports.append(report)
            if not args.check:
                output_path = args.output_dir / caption_path.parent.name / caption_path.name
                write_merged_artifact(records, output_path=output_path, overwrite=args.overwrite)
        print(json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Multimodal merge failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
