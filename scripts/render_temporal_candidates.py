"""Render top temporal retrieval candidates as a labeled contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-event", type=int, default=5)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    rows = []
    for event in results["results"]:
        thumbnails = []
        candidates = event.get("candidates", event.get("regions_by_score", []))
        if not candidates and event.get("refined_timestamp_seconds") is not None:
            candidates = [event]
        for rank, candidate in enumerate(candidates[: args.per_event], start=1):
            timestamp = candidate.get(
                "timestamp_seconds",
                candidate.get(
                    "leading_edge_seconds", candidate.get("refined_timestamp_seconds")
                ),
            )
            if timestamp is None:
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1_000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (384, 216))
            score = candidate.get(
                "score", candidate.get("peak_score", candidate.get("confidence", 0.0))
            )
            label = (
                f"{event['event_id']} R{candidate.get('rank', rank)} "
                f"{timestamp:.1f}s {score:.3f}"
            )
            cv2.rectangle(frame, (0, 0), (384, 28), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(frame)
        if thumbnails:
            rows.append(cv2.hconcat(thumbnails))
    capture.release()
    if not rows:
        raise RuntimeError("No candidate frames could be decoded")

    sheet = cv2.vconcat(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Cannot write contact sheet: {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
