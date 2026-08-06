"""Run generated stress queries against the current FastAPI backend.

Textual KIS and Q&A queries use the backend's ranked results directly. The
current backend does not generate Q&A answers, so those predictions retain a
null answer and intentionally expose that capability gap. TRAKE sends each
event description separately, votes for a video, then selects one frame per
event from that video.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def post_search(base_url: str, query: str, top_k: int, timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/search/text",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"query": query, "top_k": top_k}).encode("utf-8"),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body.get("data", {}).get("results", [])


def video_id(hit: dict[str, Any]) -> str:
    value = hit.get("video_name")
    if value:
        return str(value).removesuffix(".mp4")
    frame_id = str(hit.get("frame_id", ""))
    return frame_id.rsplit("_f", 1)[0] if "_f" in frame_id else ""


def normalise_hit(hit: dict[str, Any], answer: Any = None) -> dict[str, Any]:
    return {
        "video_id": video_id(hit),
        "frame_id": hit.get("frame_id"),
        "frame_index": hit.get("frame_index"),
        "score": hit.get("score"),
        "answer": answer,
    }


def run_query(base_url: str, query: dict[str, Any], top_k: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    if query["type"] in {"textual_kis", "qa"}:
        hits = post_search(base_url, query["query"], top_k, timeout)
        answers = [normalise_hit(hit) for hit in hits]
    else:
        event_hits = [
            post_search(base_url, event["description"], top_k, timeout)
            for event in query["events"]
        ]
        votes = Counter(
            video_id(hit)
            for hits in event_hits
            for hit in hits[:10]
            if video_id(hit)
        )
        chosen_video = votes.most_common(1)[0][0] if votes else ""
        chosen_hits = []
        for hits in event_hits:
            hit = next((item for item in hits if video_id(item) == chosen_video), None)
            chosen_hits.append(hit or {})
        answers = [
            {
                "video_id": chosen_video,
                "frame_indices": [hit.get("frame_index") for hit in chosen_hits],
                "frame_ids": [hit.get("frame_id") for hit in chosen_hits],
            }
        ]
    return {
        "query_id": query["query_id"],
        "type": query["type"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "answers": answers,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--queries",
        type=Path,
        default=repo_root / "data" / "stress_test_1200" / "queries" / "all_queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "stress_test_1200" / "predictions.jsonl",
    )
    parser.add_argument("--type", choices=("all", "textual_kis", "qa", "trake"), default="all")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    queries = load_jsonl(args.queries)
    if args.type != "all":
        queries = [query for query in queries if query["type"] == args.type]
    if args.limit > 0:
        queries = queries[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for index, query in enumerate(queries, start=1):
            try:
                prediction = run_query(args.base_url, query, args.top_k, args.timeout)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                prediction = {
                    "query_id": query["query_id"],
                    "type": query["type"],
                    "error": str(exc),
                    "answers": [],
                }
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(queries)}] {query['query_id']}")


if __name__ == "__main__":
    main()
