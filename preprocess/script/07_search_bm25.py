from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from common import DEFAULT_OUTPUT_DIR, strip_accents, tokenize

SUPPORTED_INDEX_VERSION = 3
SUPPORTED_TOKENIZER_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the stage-6 metadata BM25 index. Supports lexical search, "
            "metadata filters, human-readable output, and JSON export."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Natural-language metadata query. May be omitted for filter-only search.",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "06_metadata_bm25.pkl",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--episode-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--slot", type=str, default=None)
    parser.add_argument("--source-network", type=str, default=None)
    parser.add_argument("--min-duration", type=int, default=None, help="Seconds")
    parser.add_argument("--max-duration", type=int, default=None, help="Seconds")
    parser.add_argument(
        "--show-zero-score",
        action="store_true",
        help="Include zero-score results when a lexical query is supplied.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optionally save ranked results as UTF-8 JSON.",
    )
    return parser.parse_args()


def normalize_filter_text(value: Any) -> str:
    return " ".join(tokenize(str(value or "")))


def validate_iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("--episode-date must use YYYY-MM-DD") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.month is not None and not 1 <= args.month <= 12:
        raise ValueError("--month must be between 1 and 12")
    if args.min_duration is not None and args.min_duration < 0:
        raise ValueError("--min-duration must be non-negative")
    if args.max_duration is not None and args.max_duration < 0:
        raise ValueError("--max-duration must be non-negative")
    if (
        args.min_duration is not None
        and args.max_duration is not None
        and args.min_duration > args.max_duration
    ):
        raise ValueError("--min-duration cannot exceed --max-duration")
    args.episode_date = validate_iso_date(args.episode_date)


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path.resolve()}\n"
            "Run preprocess/script/06_build_bm25_index.py first."
        )

    with path.open("rb") as file:
        index = pickle.load(file)

    if not isinstance(index, dict):
        raise RuntimeError("Invalid BM25 index: root object is not a dictionary")

    required_keys = {
        "version",
        "tokenizer_version",
        "documents",
        "term_frequencies",
        "doc_lengths",
        "avg_doc_length",
        "idf",
        "k1",
        "b",
    }
    missing = sorted(required_keys - set(index))
    if missing:
        raise RuntimeError("Invalid BM25 index; missing keys: " + ", ".join(missing))

    if index.get("version") != SUPPORTED_INDEX_VERSION:
        raise RuntimeError(
            f"Unsupported index version {index.get('version')!r}; "
            f"expected {SUPPORTED_INDEX_VERSION}. Rebuild stage 6."
        )
    if index.get("tokenizer_version") != SUPPORTED_TOKENIZER_VERSION:
        raise RuntimeError(
            f"Unsupported tokenizer version {index.get('tokenizer_version')!r}; "
            f"expected {SUPPORTED_TOKENIZER_VERSION}. Rebuild stage 6."
        )

    document_count = len(index["documents"])
    if len(index["term_frequencies"]) != document_count:
        raise RuntimeError("Index corruption: term-frequency count does not match documents")
    if len(index["doc_lengths"]) != document_count:
        raise RuntimeError("Index corruption: document-length count does not match documents")

    return index


def prepare_query(query: str, index: dict[str, Any]) -> dict[str, Any]:
    raw_tokens = tokenize(query)
    pruned_terms = set(index.get("pruned_terms") or [])
    idf = index["idf"]

    removed_high_df = [token for token in raw_tokens if token in pruned_terms]
    remaining = [token for token in raw_tokens if token not in pruned_terms]
    out_of_vocabulary = [token for token in remaining if token not in idf]
    searchable = [token for token in remaining if token in idf]

    # Preserve query repetition but cap it to avoid accidental extreme weighting.
    query_tf = Counter(searchable)
    query_tf = Counter({term: min(freq, 3) for term, freq in query_tf.items()})

    return {
        "raw_tokens": raw_tokens,
        "query_tf": query_tf,
        "removed_high_df": removed_high_df,
        "out_of_vocabulary": out_of_vocabulary,
    }


def score_document(
    query_tf: Counter[str],
    doc_index: int,
    index: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    tf: Counter[str] = index["term_frequencies"][doc_index]
    doc_length = index["doc_lengths"][doc_index]
    avg_doc_length = float(index["avg_doc_length"] or 1.0)
    k1 = float(index["k1"])
    b = float(index["b"])
    idf: dict[str, float] = index["idf"]

    score = 0.0
    contributions: list[dict[str, Any]] = []

    for token, query_frequency in query_tf.items():
        frequency = tf.get(token, 0)
        if frequency <= 0:
            continue

        denominator = frequency + k1 * (
            1.0 - b + b * doc_length / avg_doc_length
        )
        term_score = (
            idf[token]
            * frequency
            * (k1 + 1.0)
            / denominator
            * query_frequency
        )
        score += term_score
        contributions.append(
            {
                "term": token,
                "document_tf": frequency,
                "query_tf": query_frequency,
                "idf": round(float(idf[token]), 6),
                "score": round(term_score, 6),
            }
        )

    contributions.sort(key=lambda item: (-item["score"], item["term"]))
    return score, contributions


def filter_matches(document: dict[str, Any], args: argparse.Namespace) -> bool:
    filters = document.get("filters") or {}

    if args.episode_date and str(filters.get("episode_date") or "") != args.episode_date:
        return False
    if args.year is not None and filters.get("year") != args.year:
        return False
    if args.month is not None and filters.get("month") != args.month:
        return False

    text_filters = (
        ("channel_name", args.channel),
        ("series_name", args.series),
        ("broadcast_slot", args.slot),
        ("source_network", args.source_network),
    )
    for field, expected in text_filters:
        if expected is None:
            continue
        actual_normalized = normalize_filter_text(filters.get(field))
        expected_normalized = normalize_filter_text(expected)
        if not expected_normalized or expected_normalized not in actual_normalized:
            return False

    duration = filters.get("duration_seconds")
    try:
        duration_value = int(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None

    if args.min_duration is not None:
        if duration_value is None or duration_value < args.min_duration:
            return False
    if args.max_duration is not None:
        if duration_value is None or duration_value > args.max_duration:
            return False

    return True


def build_result(
    rank: int,
    score: float,
    document: dict[str, Any],
    contributions: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = document.get("payload") or {}
    filters = document.get("filters") or {}
    search_fields = document.get("search_fields") or {}

    return {
        "rank": rank,
        "video_id": document.get("video_id"),
        "score": round(score, 6),
        "matched_terms": [item["term"] for item in contributions],
        "term_contributions": contributions,
        "title": payload.get("title") or search_fields.get("title"),
        "episode_date": filters.get("episode_date"),
        "publish_date": filters.get("publish_date"),
        "series_name": filters.get("series_name"),
        "broadcast_slot": filters.get("broadcast_slot"),
        "channel_name": filters.get("channel_name"),
        "duration_seconds": filters.get("duration_seconds"),
        "duration_hms": payload.get("duration_hms"),
        "watch_url": payload.get("watch_url"),
        "thumbnail_url": payload.get("thumbnail_url"),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    index = load_index(args.index_file)

    query_info = prepare_query(args.query, index)
    query_tf: Counter[str] = query_info["query_tf"]
    lexical_query_supplied = bool(args.query.strip())

    has_filters = any(
        value is not None
        for value in (
            args.episode_date,
            args.year,
            args.month,
            args.channel,
            args.series,
            args.slot,
            args.source_network,
            args.min_duration,
            args.max_duration,
        )
    )
    if not lexical_query_supplied and not has_filters:
        raise ValueError("Provide a query, at least one metadata filter, or both")

    candidates: list[tuple[float, str, dict[str, Any], list[dict[str, Any]]]] = []
    for doc_index, document in enumerate(index["documents"]):
        if not filter_matches(document, args):
            continue

        if query_tf:
            score, contributions = score_document(query_tf, doc_index, index)
        else:
            score, contributions = 0.0, []

        if lexical_query_supplied and not args.show_zero_score and score <= 0.0:
            continue

        video_id = str(document.get("video_id") or "")
        candidates.append((score, video_id, document, contributions))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[: args.top_k]
    results = [
        build_result(rank, score, document, contributions)
        for rank, (score, _video_id, document, contributions) in enumerate(
            selected, start=1
        )
    ]

    print(f"Index version: {index['version']}")
    print(f"Indexed documents: {len(index['documents'])}")
    print(f"Query: {args.query or '<filter-only>'}")
    print("Tokenized query:", query_info["raw_tokens"])
    print("Searchable terms:", list(query_tf.elements()))
    if query_info["removed_high_df"]:
        print("Ignored high-DF terms:", query_info["removed_high_df"])
    if query_info["out_of_vocabulary"]:
        print("Out-of-vocabulary terms:", query_info["out_of_vocabulary"])
    print(f"Matching candidates: {len(candidates)}")

    if lexical_query_supplied and not query_tf:
        print("No searchable query terms remain after pruning/OOV filtering.")
    if not results:
        print("No results matched the query and filters.")
    else:
        for result in results:
            matched = ", ".join(result["matched_terms"]) or "<filter-only>"
            print(
                f"{result['rank']:02d}. {result['video_id']} | "
                f"score={result['score']:.6f} | matched={matched}"
            )
            print(f"    title:   {result['title']}")
            print(
                "    meta:    "
                f"date={result['episode_date']} | "
                f"channel={result['channel_name']} | "
                f"duration={result['duration_hms']}"
            )
            print(f"    url:     {result['watch_url']}")

    if args.json_output is not None:
        output = {
            "query": args.query,
            "query_tokens": query_info["raw_tokens"],
            "searchable_terms": list(query_tf.elements()),
            "ignored_high_df_terms": query_info["removed_high_df"],
            "out_of_vocabulary_terms": query_info["out_of_vocabulary"],
            "candidate_count": len(candidates),
            "result_count": len(results),
            "results": results,
        }
        save_json(args.json_output, output)
        print(f"JSON output: {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
