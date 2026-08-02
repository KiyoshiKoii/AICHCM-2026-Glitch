from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from common import (
    DEFAULT_OUTPUT_DIR,
    normalize_text,
    read_jsonl,
    strip_accents,
    tokenize,
    write_json,
    write_jsonl,
)

SEARCH_TEXT_VERSION = 3

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one clean metadata search document per video. "
            "Search text contains values only (no repeated field labels), removes URLs, "
            "and keeps structured fields separately for filtering."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "04_enriched_metadata.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sanitize_search_value(value: Any) -> str:
    """Normalize text and remove URL/email artifacts before lexical indexing."""
    if value is None:
        return ""
    text = normalize_text(str(value))
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" ,;|.-")
    return text


def comparison_key(text: str) -> str:
    return WHITESPACE_RE.sub(" ", strip_accents(text).casefold()).strip()


def clean_scalar(value: Any) -> str | None:
    text = sanitize_search_value(value)
    return text or None


def safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = sanitize_search_value(item)
        key = comparison_key(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def date_variants(iso_date: Any) -> list[str]:
    """Variants useful for Vietnamese date queries, without field-label boilerplate."""
    if not iso_date:
        return []

    text = str(iso_date).strip()
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        clean = sanitize_search_value(text)
        return [clean] if clean else []

    return [
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%d/%m/%Y"),
        dt.strftime("%d%m%Y"),
        f"{dt.day} tháng {dt.month} năm {dt.year}",
        f"ngày {dt.day} tháng {dt.month} năm {dt.year}",
    ]


def extend_weighted(parts: list[str], values: Any, repeat: int = 1) -> None:
    if values in (None, "", []):
        return

    if isinstance(values, (list, tuple, set)):
        clean_values = [sanitize_search_value(item) for item in values]
    else:
        clean_values = [sanitize_search_value(values)]

    clean_values = [value for value in clean_values if value]
    if not clean_values:
        return

    fragment = " ".join(clean_values)
    parts.extend([fragment] * max(1, repeat))


def build_search_text(
    *,
    title: str | None,
    description: str | None,
    description_context: list[str],
    keywords: list[str],
    keywords_common: list[str],
    derived: dict[str, Any],
) -> str:
    """
    Create a compact weighted lexical document.

    Repetition approximates field weights for the simple BM25 implementation, while
    omitting labels such as 'Tiêu đề' or 'Ngày tập' that otherwise occur in every doc.
    Duration and upload URL are kept as filters/payload only, not lexical text.
    """
    parts: list[str] = []

    # Video-specific content: strongest signals.
    extend_weighted(parts, title, repeat=4)
    extend_weighted(parts, description, repeat=2)
    extend_weighted(parts, keywords, repeat=3)

    # Structured program metadata.
    extend_weighted(parts, derived.get("series_name"), repeat=2)
    extend_weighted(parts, derived.get("broadcast_slot"))
    extend_weighted(parts, date_variants(derived.get("episode_date")))
    extend_weighted(parts, derived.get("weekday"))
    extend_weighted(parts, derived.get("channel_name"))
    extend_weighted(parts, derived.get("source_network"))

    # Lower-weight reusable channel/program context.
    extend_weighted(parts, description_context)
    extend_weighted(parts, keywords_common)

    return WHITESPACE_RE.sub(" ", " ".join(parts)).strip()


def contains_url_artifact(text: str) -> bool:
    lowered = text.casefold()
    return bool(URL_RE.search(text)) or any(
        token in tokenize(lowered) for token in ("http", "https", "www")
    )


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_file)
    if not records:
        raise RuntimeError(f"No enriched metadata found: {args.input_file.resolve()}")

    documents: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    seen_video_ids: set[str] = set()
    duplicate_video_ids: list[str] = []
    token_counts: list[int] = []

    for record in records:
        video_id = sanitize_search_value(record.get("video_id"))
        clean = safe_dict(record.get("clean_metadata"))
        derived = safe_dict(record.get("derived_metadata"))

        if not video_id:
            issue_counts["missing_video_id"] += 1
            continue

        if video_id in seen_video_ids:
            issue_counts["duplicate_video_id"] += 1
            duplicate_video_ids.append(video_id)
            continue
        seen_video_ids.add(video_id)

        title = clean_scalar(clean.get("title"))
        description = clean_scalar(clean.get("description"))
        description_context = safe_list(clean.get("description_context"))
        keywords = safe_list(clean.get("keywords"))
        keywords_common = safe_list(clean.get("keywords_common"))

        text = build_search_text(
            title=title,
            description=description,
            description_context=description_context,
            keywords=keywords,
            keywords_common=keywords_common,
            derived=derived,
        )
        text_tokens = tokenize(text)
        token_counts.append(len(text_tokens))

        if not title:
            issue_counts["missing_title"] += 1
        if not text:
            issue_counts["empty_search_text"] += 1
        if not text_tokens:
            issue_counts["empty_tokenized_search_text"] += 1
        if not derived:
            issue_counts["missing_derived_metadata"] += 1
        if contains_url_artifact(text):
            issue_counts["url_artifact_in_search_text"] += 1

        coverage_counts[
            "has_specific_description" if description else "missing_specific_description"
        ] += 1
        coverage_counts[
            "has_description_context"
            if description_context
            else "missing_description_context"
        ] += 1
        coverage_counts[
            "has_specific_keywords" if keywords else "missing_specific_keywords"
        ] += 1
        coverage_counts[
            "has_common_keywords" if keywords_common else "missing_common_keywords"
        ] += 1
        coverage_counts[
            "has_episode_date" if derived.get("episode_date") else "missing_episode_date"
        ] += 1

        documents.append(
            {
                "video_id": video_id,
                "search_text_version": SEARCH_TEXT_VERSION,
                "text": text,
                "search_fields": {
                    "title": title,
                    "description": description,
                    "description_context": description_context,
                    "keywords": keywords,
                    "keywords_common": keywords_common,
                    "series_name": derived.get("series_name"),
                    "broadcast_slot": derived.get("broadcast_slot"),
                    "episode_date": derived.get("episode_date"),
                    "publish_date": derived.get("publish_date_normalized"),
                    "weekday": derived.get("weekday"),
                    "channel_name": derived.get("channel_name"),
                    "source_network": derived.get("source_network"),
                },
                "filters": {
                    "episode_date": derived.get("episode_date"),
                    "publish_date": derived.get("publish_date_normalized"),
                    "date_source": derived.get("date_source"),
                    "date_relation": derived.get("date_relation"),
                    "publish_date_offset_days": derived.get("publish_date_offset_days"),
                    "year": derived.get("year"),
                    "month": derived.get("month"),
                    "day": derived.get("day"),
                    "weekday": derived.get("weekday"),
                    "series_name": derived.get("series_name"),
                    "broadcast_slot": derived.get("broadcast_slot"),
                    "channel_name": derived.get("channel_name"),
                    "channel_id": derived.get("channel_id"),
                    "source_network": derived.get("source_network"),
                    "duration_seconds": derived.get("duration_seconds"),
                },
                "payload": {
                    "title": title,
                    "description": description,
                    "watch_url": clean.get("watch_url"),
                    "thumbnail_url": clean.get("thumbnail_url"),
                    "youtube_id": derived.get("youtube_id"),
                    "duration_seconds": derived.get("duration_seconds"),
                    "duration_hms": derived.get("duration_hms"),
                    "channel_name": derived.get("channel_name"),
                },
            }
        )

    output_jsonl = args.output_dir / "05_metadata_search_documents.jsonl"
    output_summary = args.output_dir / "05_search_documents_summary.json"

    write_jsonl(output_jsonl, documents)
    write_json(
        output_summary,
        {
            "input_record_count": len(records),
            "output_document_count": len(documents),
            "search_text_version": SEARCH_TEXT_VERSION,
            "issue_counts": dict(sorted(issue_counts.items())),
            "coverage_counts": dict(sorted(coverage_counts.items())),
            "duplicate_video_ids": duplicate_video_ids,
            "search_text_token_statistics": {
                "total": sum(token_counts),
                "average": round(sum(token_counts) / len(token_counts), 4)
                if token_counts
                else 0,
                "minimum": min(token_counts) if token_counts else 0,
                "maximum": max(token_counts) if token_counts else 0,
            },
            "output_jsonl": str(output_jsonl),
        },
    )

    print(f"Built {len(documents)} search documents from {len(records)} records")
    print(f"Search text version: {SEARCH_TEXT_VERSION}")
    if token_counts:
        print(
            "Average search-document length: "
            f"{sum(token_counts) / len(token_counts):.2f} tokens"
        )
    print(f"Output: {output_jsonl.resolve()}")
    print(f"Summary: {output_summary.resolve()}")

    if issue_counts:
        print("Issues:")
        for key, count in sorted(issue_counts.items()):
            print(f"- {key}: {count}")


if __name__ == "__main__":
    main()
