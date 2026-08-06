"""Score AIC stress-test predictions with the official R@k cutoffs.

Prediction JSONL format:
  {"query_id":"tkis-0001","answers":[{"video_id":"L21_V001","frame_index":90}]}

Q&A answers additionally contain ``answer``. TRAKE answers contain one video
and an ordered ``frame_indices`` list. At most the first 100 answers are used.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CUTOFFS = (1, 5, 20, 50, 100)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            output.append(value)
    return output


def normalise_video(value: Any) -> str:
    text = str(value or "").strip()
    return text[:-4] if text.casefold().endswith(".mp4") else text


def normalise_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def extract_frame_index(answer: dict[str, Any]) -> int | None:
    value = answer.get("frame_index", answer.get("submission_frame_id"))
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    frame_id = str(answer.get("frame_id", ""))
    if "_f" in frame_id:
        try:
            return int(frame_id.rsplit("_f", 1)[1].removesuffix(".jpg"))
        except ValueError:
            return None
    return None


def in_range(value: int | None, bounds: Iterable[int]) -> bool:
    if value is None:
        return False
    start, end = [int(item) for item in bounds]
    return start <= value <= end


def answer_score(query: dict[str, Any], answer: dict[str, Any]) -> float:
    query_type = query["type"]
    target = query["target"]
    video_id = normalise_video(answer.get("video_id", answer.get("video_name")))
    if video_id != normalise_video(target["video_id"]):
        return 0.0
    if query_type in {"textual_kis", "qa"}:
        if not in_range(extract_frame_index(answer), target["valid_frame_range"]):
            return 0.0
        if query_type == "qa" and normalise_answer(answer.get("answer")) != normalise_answer(query["answer"]):
            return 0.0
        return 1.0
    if query_type == "trake":
        predicted = answer.get("frame_indices")
        if not isinstance(predicted, list):
            return 0.0
        ranges = target["valid_frame_ranges"]
        matches = sum(
            in_range(predicted[index] if index < len(predicted) else None, bounds)
            for index, bounds in enumerate(ranges)
        )
        return matches / len(ranges) if ranges else 0.0
    raise ValueError(f"Unsupported query type: {query_type}")


def evaluate_query(query: dict[str, Any], answers: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [answer_score(query, answer) for answer in answers[:100]]
    r_at_k = {
        str(k): max(scores[:k], default=0.0)
        for k in CUTOFFS
    }
    return {
        "query_id": query["query_id"],
        "type": query["type"],
        "answer_count": min(len(answers), 100),
        "r_at_k": r_at_k,
        "final_score": statistics.fmean(r_at_k.values()),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=repo_root / "data" / "stress_test_1200" / "queries" / "all_queries.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repo_root / "data" / "stress_test_1200" / "predictions.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--only-predicted",
        action="store_true",
        help="Score only query IDs present in the prediction file (useful for bounded smoke runs).",
    )
    args = parser.parse_args()

    queries = {item["query_id"]: item for item in load_jsonl(args.queries)}
    predictions = {item["query_id"]: item for item in load_jsonl(args.predictions)}
    rows = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for query_id, query in queries.items():
        if args.only_predicted and query_id not in predictions:
            continue
        prediction = predictions.get(query_id, {})
        answers = prediction.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError(f"Prediction {query_id} answers must be a list")
        row = evaluate_query(query, answers)
        rows.append(row)
        by_type[row["type"]].append(row["final_score"])

    report = {
        "query_count": len(rows),
        "prediction_count": len(predictions),
        "only_predicted": args.only_predicted,
        "mean_final_score": statistics.fmean(row["final_score"] for row in rows) if rows else 0.0,
        "by_type": {
            query_type: {
                "queries": len(values),
                "mean_final_score": statistics.fmean(values),
            }
            for query_type, values in sorted(by_type.items())
        },
        "queries": rows,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
