"""Read and validate Gemini, ASR and BTC keyframe-map sources."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .models import ASRSegment, KeyframeMapRow, RuntimeFrame, SourceInfo


FRAME_ID_RE = re.compile(r"^(?P<video>.+)_f(?P<n>\d+)$")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_info(path: Path) -> SourceInfo:
    return SourceInfo(path=str(path), sha256=sha256_file(path))


def _finite_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: {field} must be finite")
    return result


def load_keyframe_map(path: Path) -> list[KeyframeMapRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Map-keyframes file does not exist: {path}")
    rows: list[KeyframeMapRow] = []
    seen_n: set[int] = set()
    last_time = -1.0
    last_frame_idx = -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"n", "pts_time", "fps", "frame_idx"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                n = int(str(raw["n"]).strip())
                frame_idx = int(str(raw["frame_idx"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: n/frame_idx must be integers") from exc
            pts_time = _finite_float(raw["pts_time"], "pts_time", path)
            fps = _finite_float(raw["fps"], "fps", path)
            if n < 1 or frame_idx < 0 or pts_time < 0 or fps <= 0:
                raise ValueError(f"{path}:{line_number}: invalid map values")
            if n in seen_n:
                raise ValueError(f"{path}:{line_number}: duplicate n={n}")
            # BTC map generation can round two distinct PTS values to the same
            # native frame index.  ``pts_time`` remains the authoritative temporal
            # coordinate, so accept equal frame indices while rejecting a reversal.
            if pts_time <= last_time or frame_idx < last_frame_idx:
                raise ValueError(f"{path}:{line_number}: map must be chronological")
            rows.append(KeyframeMapRow(n, pts_time, fps, frame_idx))
            seen_n.add(n)
            last_time = pts_time
            last_frame_idx = frame_idx
    if not rows:
        raise ValueError(f"{path}: map is empty")
    return rows


def load_caption_records(path: Path, video_id: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: caption artifact must be a non-empty JSON array")
    records: list[dict[str, Any]] = []
    seen_n: set[int] = set()
    for position, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}[{position}]: caption record must be an object")
        frame_id = raw.get("frame_id")
        if not isinstance(frame_id, str):
            raise ValueError(f"{path}[{position}]: missing frame_id")
        match = FRAME_ID_RE.fullmatch(frame_id.strip())
        if not match or match.group("video") != video_id:
            raise ValueError(f"{path}[{position}]: invalid frame_id {frame_id!r}")
        n = int(match.group("n"))
        if n < 1 or n in seen_n:
            raise ValueError(f"{path}[{position}]: duplicate/invalid keyframe n={n}")
        record = dict(raw)
        record["_keyframe_n"] = n
        records.append(record)
        seen_n.add(n)
    return sorted(records, key=lambda item: item["_keyframe_n"])


def load_asr_segments(path: Path, video_id: str) -> tuple[list[ASRSegment], bool]:
    if not path.is_file():
        return [], False
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: ASR payload must be an object")
    if payload.get("video_name") != video_id:
        raise ValueError(f"{path}: video_name does not match {video_id}")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError(f"{path}: segments must be an array")
    segments: list[ASRSegment] = []
    last_start = -1.0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: segment {index} must be an object")
        start = _finite_float(raw.get("start"), f"segments[{index}].start", path)
        end = _finite_float(raw.get("end"), f"segments[{index}].end", path)
        text = " ".join(str(raw.get("text", "")).split())
        if start < 0 or end < start or not text:
            raise ValueError(f"{path}: segment {index} has invalid time/text")
        if start < last_start:
            raise ValueError(f"{path}: ASR segments are not chronological")
        segments.append(ASRSegment(index, start, end, text))
        last_start = start
    return segments, True


def build_runtime_frames(
    caption_records: list[dict[str, Any]],
    keyframe_map: list[KeyframeMapRow],
    video_id: str,
    keyframe_dir: Path | None = None,
) -> list[RuntimeFrame]:
    map_by_n = {row.n: row for row in keyframe_map}
    frames: list[RuntimeFrame] = []
    for record in caption_records:
        n = int(record["_keyframe_n"])
        if n not in map_by_n:
            raise ValueError(f"{video_id}: caption keyframe n={n} has no map row")
        if keyframe_dir is not None:
            image = keyframe_dir / f"{n:03d}.jpg"
            if not image.is_file():
                raise FileNotFoundError(f"{video_id}: keyframe image missing: {image}")
        row = map_by_n[n]
        clean_record = dict(record)
        clean_record.pop("_keyframe_n", None)
        frames.append(
            RuntimeFrame(
                video_id=video_id,
                keyframe_n=n,
                frame_id=str(record["frame_id"]),
                timestamp_ms=row.timestamp_ms,
                fps=row.fps,
                native_frame_idx=row.frame_idx,
                raw_metadata=clean_record,
            )
        )
    if len(frames) != len(keyframe_map):
        raise ValueError(
            f"{video_id}: caption/map count mismatch ({len(frames)} vs {len(keyframe_map)})"
        )
    return frames
