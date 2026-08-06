"""Validate the generated stress corpus without heavyweight image libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def validate_image(path: Path, expected_sha256: str) -> None:
    data = path.read_bytes()
    if not (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise ValueError(f"Unsupported or corrupt image signature: {path}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repo_root / "data" / "stress_test_1200",
    )
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    summary = json.loads((corpus / "summary.json").read_text(encoding="utf-8"))
    manifest = load_jsonl(corpus / "manifest.jsonl")
    if len(manifest) != summary["frames"]:
        raise ValueError("Manifest count does not match summary")

    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    frames_by_dataset: Counter[str] = Counter()
    frames_by_video: Counter[str] = Counter()
    frames_with_objects = 0
    detections = 0
    for sample in manifest:
        sample_id = sample["sample_id"]
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        sample_ids.add(sample_id)
        relative_image = sample["relative_image_path"]
        if relative_image in image_paths:
            raise ValueError(f"Duplicate image path: {relative_image}")
        image_paths.add(relative_image)
        validate_image(corpus / relative_image, sample["image_sha256"])

        object_path = sample.get("relative_object_path")
        if object_path:
            raw = json.loads((corpus / object_path).read_text(encoding="utf-8-sig"))
            if not isinstance(raw, (dict, list)):
                raise ValueError(f"Unexpected object JSON: {object_path}")
        if sample.get("object_keywords"):
            frames_with_objects += 1
        detections += len(sample.get("detections", []))
        if int(sample["source_frame_index"]) < 0:
            raise ValueError(f"Negative source frame: {sample_id}")
        frames_by_dataset[sample["dataset"]] += 1
        frames_by_video[sample["video_id"]] += 1

    all_queries = load_jsonl(corpus / "queries" / "all_queries.jsonl")
    query_ids: set[str] = set()
    query_types: Counter[str] = Counter()
    samples_by_id = {sample["sample_id"]: sample for sample in manifest}
    for query in all_queries:
        query_id = query["query_id"]
        if query_id in query_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")
        query_ids.add(query_id)
        query_types[query["type"]] += 1
        target = query["target"]
        if query["type"] in {"textual_kis", "qa"}:
            sample = samples_by_id[query["source_sample_id"]]
            if target["video_id"] != sample["video_id"]:
                raise ValueError(f"Target video mismatch: {query_id}")
            start, end = target["valid_frame_range"]
            if not start <= target["frame_index"] <= end:
                raise ValueError(f"Target range mismatch: {query_id}")
        else:
            frames = target["frame_indices"]
            if frames != sorted(frames):
                raise ValueError(f"TRAKE frames are not ordered: {query_id}")
            if len(frames) != len(target["valid_frame_ranges"]):
                raise ValueError(f"TRAKE range count mismatch: {query_id}")

    expected_query_counts = summary["queries"]
    for query_type in ("textual_kis", "qa", "trake"):
        if query_types[query_type] != expected_query_counts[query_type]:
            raise ValueError(f"Query count mismatch for {query_type}")
    if len(query_ids) != expected_query_counts["total"]:
        raise ValueError("Total query count mismatch")

    report = {
        "status": "valid",
        "frames": len(manifest),
        "videos": len(frames_by_video),
        "frames_by_dataset": dict(sorted(frames_by_dataset.items())),
        "frames_per_video_min": min(frames_by_video.values()),
        "frames_per_video_max": max(frames_by_video.values()),
        "frames_with_object_keywords": frames_with_objects,
        "object_keyword_coverage": round(frames_with_objects / len(manifest), 4),
        "retained_detections": detections,
        "queries": dict(sorted(query_types.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
