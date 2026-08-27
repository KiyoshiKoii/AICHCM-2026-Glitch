"""Offline ranking metrics for the Qwen-vs-CLIP pilot.

The evaluator consumes saved JSON/JSONL files, so running it never loads a
model or changes Qdrant.  This keeps annotation and scoring repeatable after
the embedding/search commands have been run separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    loaded = json.loads(path.read_text(encoding="utf-8"))
    yield from (loaded if isinstance(loaded, list) else loaded["queries"])


def _hit_key(hit: dict[str, Any]) -> tuple[str, str]:
    return str(hit.get("video_id", "")), str(hit.get("frame_id", ""))


def _relevant_keys(record: dict[str, Any]) -> set[tuple[str, str]]:
    values = record.get("relevant", record.get("relevant_frames", []))
    keys: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, str):
            keys.add((str(record.get("video_id", "L21_V001")), value))
        else:
            keys.add((str(value["video_id"]), str(value["frame_id"])))
    return keys


def score(records: Iterable[dict[str, Any]], cutoffs: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    rows = list(records)
    if not rows:
        raise ValueError("No evaluation records found")

    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = 0
        for row in rows:
            relevant = _relevant_keys(row)
            ranked = [_hit_key(hit) for hit in row.get("hits", [])[:cutoff]]
            hits += int(bool(set(ranked) & relevant))
        metrics[f"recall@{cutoff}"] = hits / len(rows)

    reciprocal_ranks = []
    for row in rows:
        relevant = _relevant_keys(row)
        rank = next(
            (position for position, hit in enumerate(row.get("hits", []), start=1) if _hit_key(hit) in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    metrics["mrr@all"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSON/JSONL ranked results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(score(_records(args.results)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
