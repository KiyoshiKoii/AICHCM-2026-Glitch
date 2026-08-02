from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    list_json_files,
    normalize_for_compare,
    normalize_text,
    read_json,
    write_json,
    write_jsonl,
)

# A line that is essentially only one URL, optionally preceded by a bullet/icon.
URL_ONLY_RE = re.compile(
    r"^(?:[►✅\-•.*📱👉]\s*)?(?:https?://|www\.)\S+$",
    re.IGNORECASE,
)

# Lines that begin with a social/platform label are almost always channel promotion.
SOCIAL_PREFIX_RE = re.compile(
    r"^(?:[►✅\-•.*📱👉]\s*)?"
    r"(?:facebook|fanpage|instagram|lotus|tiktok|youtube(?:\s+channel)?|"
    r"ios|android|web|wap|smart\s*tv|app(?:\s+mobile)?|website)\b",
    re.IGNORECASE,
)

# Hard noise is removed entirely rather than retained as channel/program context.
# Keep this conservative: repeated topic/program phrases should go to context, not here.
HARD_NOISE_RE = re.compile(
    r"(?:"
    r"đăng\s*ký|đăng\s*kí|subscribe|"
    r"bản\s*quyền|copyright|"
    r"theo\s*dõi|"
    r"website|fanpage|facebook|instagram|lotus|tiktok|"
    r"youtube\s+channel|"
    r"email|e-mail|fax|"
    r"điện\s*thoại|đường\s*dây\s*nóng|hotline|"
    r"địa\s*chỉ|tòa\s*soạn|"
    r"xem\s*tv\s*online|app\s*mobile|web\s*/\s*wap|smart\s*tv"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean metadata per channel. Hard promotional noise is removed, while "
            "frequent channel/program context and common keywords are preserved in "
            "separate fields."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--boilerplate-threshold",
        type=float,
        default=0.80,
        help=(
            "Inside each channel, a description line or keyword appearing in at "
            "least this fraction of videos is considered common context."
        ),
    )
    parser.add_argument(
        "--min-channel-size",
        type=int,
        default=5,
        help=(
            "Do not infer repeated common context for channels with fewer than this "
            "number of videos. Obvious hard-noise lines are still removed."
        ),
    )
    return parser.parse_args()


def get_channel_key(record: dict[str, Any]) -> str:
    """Return a stable grouping key for per-channel analysis."""
    channel_id = normalize_text(str(record.get("channel_id") or ""))
    if channel_id:
        return channel_id

    author = normalize_text(str(record.get("author") or ""))
    if author:
        return f"author::{author.casefold()}"

    return "__unknown_channel__"


def collect_document_frequencies(
    records: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    """Count each normalized line/keyword at most once per video."""
    line_df: Counter[str] = Counter()
    keyword_df: Counter[str] = Counter()

    for record in records:
        seen_lines: set[str] = set()
        description = str(record.get("description") or "")
        for raw_line in description.splitlines():
            normalized = normalize_for_compare(raw_line)
            if normalized:
                seen_lines.add(normalized)
        line_df.update(seen_lines)

        seen_keywords: set[str] = set()
        keywords = record.get("keywords") or []
        if isinstance(keywords, list):
            for keyword in keywords:
                normalized = normalize_for_compare(str(keyword))
                if normalized:
                    seen_keywords.add(normalized)
        keyword_df.update(seen_keywords)

    return line_df, keyword_df


def build_channel_rules(
    records: list[dict[str, Any]],
    threshold: float,
    min_channel_size: int,
) -> dict[str, dict[str, Any]]:
    """Build independent common-context rules for every channel."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[get_channel_key(record)].append(record)

    rules: dict[str, dict[str, Any]] = {}

    for channel_key, channel_records in grouped.items():
        channel_size = len(channel_records)
        line_df, keyword_df = collect_document_frequencies(channel_records)

        if channel_size < min_channel_size:
            minimum_count: int | None = None
            common_lines: set[str] = set()
            common_keywords: set[str] = set()
        else:
            minimum_count = max(1, math.ceil(channel_size * threshold))
            common_lines = {
                line for line, count in line_df.items() if count >= minimum_count
            }
            common_keywords = {
                keyword
                for keyword, count in keyword_df.items()
                if count >= minimum_count
            }

        sample = channel_records[0]
        rules[channel_key] = {
            "channel_id": sample.get("channel_id"),
            "author": sample.get("author"),
            "record_count": channel_size,
            "minimum_document_count": minimum_count,
            # Keep these names for backward compatibility with earlier outputs.
            "boilerplate_description_lines": common_lines,
            "boilerplate_keywords": common_keywords,
        }

    return rules


def is_separator_line(text: str) -> bool:
    """Return True when a line contains no alphanumeric character."""
    return bool(text) and not any(char.isalnum() for char in text)


def is_hashtag_only_line(text: str) -> bool:
    """Return True for lines made only of hashtags and whitespace."""
    tokens = text.split()
    return bool(tokens) and all(token.startswith("#") for token in tokens)


def is_hard_noise_line(text: str) -> bool:
    """Detect promotional/contact/formatting noise that can be dropped safely."""
    return (
        bool(URL_ONLY_RE.match(text))
        or bool(SOCIAL_PREFIX_RE.match(text))
        or bool(HARD_NOISE_RE.search(text))
        or is_separator_line(text)
        or is_hashtag_only_line(text)
    )


def clean_description(
    description: str,
    common_lines: set[str],
) -> tuple[str, list[str], list[str]]:
    """
    Split a description into three groups.

    Returns:
        specific_text:
            Video-specific lines retained as the main description.
        context_lines:
            Repeated channel/program lines. They are preserved as weak context.
        removed_lines:
            Hard promotional/contact/formatting noise.
    """
    specific_lines: list[str] = []
    context_lines: list[str] = []
    removed_lines: list[str] = []

    seen_specific: set[str] = set()
    seen_context: set[str] = set()
    seen_removed: set[str] = set()

    for raw_line in description.splitlines():
        line = normalize_text(raw_line)
        normalized = normalize_for_compare(line)

        if not normalized:
            continue

        if is_hard_noise_line(line):
            if normalized not in seen_removed:
                removed_lines.append(line)
                seen_removed.add(normalized)
            continue

        if normalized in common_lines:
            if normalized not in seen_context:
                context_lines.append(line)
                seen_context.add(normalized)
            continue

        if normalized not in seen_specific:
            specific_lines.append(line)
            seen_specific.add(normalized)

    return "\n".join(specific_lines), context_lines, removed_lines


def split_keywords(
    keywords: Any,
    common_keyword_set: set[str],
) -> tuple[list[str], list[str]]:
    """
    Split keywords without deleting frequent topical/program signals.

    Returns:
        specific_keywords: keywords not common across the channel.
        common_keywords: frequent channel/program/topic keywords.
    """
    if not isinstance(keywords, list):
        return [], []

    specific_keywords: list[str] = []
    common_keywords: list[str] = []
    seen: set[str] = set()

    for raw_keyword in keywords:
        keyword = normalize_text(str(raw_keyword))
        normalized = normalize_for_compare(keyword)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)

        if normalized in common_keyword_set:
            common_keywords.append(keyword)
        else:
            specific_keywords.append(keyword)

    return specific_keywords, common_keywords


def serialize_rules(
    rules: dict[str, dict[str, Any]],
    threshold: float,
    min_channel_size: int,
    total_records: int,
) -> dict[str, Any]:
    serializable_channels: dict[str, Any] = {}

    for channel_key, rule in sorted(rules.items()):
        serializable_channels[channel_key] = {
            "channel_id": rule["channel_id"],
            "author": rule["author"],
            "record_count": rule["record_count"],
            "minimum_document_count": rule["minimum_document_count"],
            "boilerplate_description_lines": sorted(
                rule["boilerplate_description_lines"]
            ),
            "boilerplate_keywords": sorted(rule["boilerplate_keywords"]),
        }

    return {
        "record_count": total_records,
        "channel_count": len(rules),
        "boilerplate_threshold": threshold,
        "min_channel_size": min_channel_size,
        "note": (
            "Document frequency is calculated separately inside each channel. "
            "Frequent lines and keywords are preserved as context/common fields; "
            "only hard promotional, contact, URL-only, hashtag-only, and separator "
            "lines are removed. Channels smaller than min_channel_size do not "
            "receive inferred common-context rules."
        ),
        "channels": serializable_channels,
    }


def main() -> None:
    args = parse_args()

    if not 0 < args.boilerplate_threshold <= 1:
        raise ValueError("--boilerplate-threshold must be in the interval (0, 1]")
    if args.min_channel_size < 1:
        raise ValueError("--min-channel-size must be at least 1")

    paths = list_json_files(args.input_dir)
    records = [
        {
            "video_id": path.stem,
            "source_file": str(path),
            **read_json(path),
        }
        for path in paths
    ]

    if not records:
        raise RuntimeError("No metadata records found")

    channel_rules = build_channel_rules(
        records=records,
        threshold=args.boilerplate_threshold,
        min_channel_size=args.min_channel_size,
    )

    cleaned_rows: list[dict[str, Any]] = []
    total_specific_description_lines = 0
    total_context_description_lines = 0
    total_removed_description_lines = 0
    total_specific_keywords = 0
    total_common_keywords = 0

    for record in records:
        channel_key = get_channel_key(record)
        rule = channel_rules[channel_key]

        clean_desc, description_context, removed_lines = clean_description(
            str(record.get("description") or ""),
            rule["boilerplate_description_lines"],
        )
        specific_keywords, common_keywords = split_keywords(
            record.get("keywords"),
            rule["boilerplate_keywords"],
        )

        total_specific_description_lines += len(clean_desc.splitlines()) if clean_desc else 0
        total_context_description_lines += len(description_context)
        total_removed_description_lines += len(removed_lines)
        total_specific_keywords += len(specific_keywords)
        total_common_keywords += len(common_keywords)

        cleaned_rows.append(
            {
                "video_id": record["video_id"],
                "source_file": record["source_file"],
                "channel_group_key": channel_key,
                "raw_metadata": {
                    key: value
                    for key, value in record.items()
                    if key not in {"video_id", "source_file"}
                },
                "clean_metadata": {
                    "author": normalize_text(str(record.get("author") or "")),
                    "channel_id": record.get("channel_id"),
                    "channel_url": record.get("channel_url"),
                    "title": normalize_text(str(record.get("title") or "")),
                    "publish_date": record.get("publish_date"),
                    "length": record.get("length"),
                    "description": clean_desc,
                    "description_context": description_context,
                    "keywords": specific_keywords,
                    "keywords_common": common_keywords,
                    "thumbnail_url": record.get("thumbnail_url"),
                    "watch_url": record.get("watch_url"),
                },
                "cleaning_report": {
                    "channel_record_count": rule["record_count"],
                    "channel_minimum_document_count": rule[
                        "minimum_document_count"
                    ],
                    "removed_description_lines": removed_lines,
                    "moved_description_context": description_context,
                    "moved_common_keywords": common_keywords,
                },
            }
        )

    output_jsonl = args.output_dir / "03_clean_metadata.jsonl"
    output_rules = args.output_dir / "03_cleaning_rules.json"
    output_summary = args.output_dir / "03_cleaning_summary.json"

    write_jsonl(output_jsonl, cleaned_rows)
    write_json(
        output_rules,
        serialize_rules(
            rules=channel_rules,
            threshold=args.boilerplate_threshold,
            min_channel_size=args.min_channel_size,
            total_records=len(records),
        ),
    )
    write_json(
        output_summary,
        {
            "record_count": len(cleaned_rows),
            "channel_count": len(channel_rules),
            "specific_description_line_count": total_specific_description_lines,
            "context_description_line_count": total_context_description_lines,
            "removed_description_line_count": total_removed_description_lines,
            "specific_keyword_count": total_specific_keywords,
            "common_keyword_count": total_common_keywords,
            "output_jsonl": str(output_jsonl),
            "output_rules": str(output_rules),
        },
    )

    print(f"Cleaned {len(cleaned_rows)} records")
    print(f"Built rules for {len(channel_rules)} channels")
    for channel_key, rule in sorted(channel_rules.items()):
        minimum = rule["minimum_document_count"]
        threshold_text = (
            f"{minimum}/{rule['record_count']}"
            if minimum is not None
            else "skipped: channel too small"
        )
        print(
            f"- {channel_key}: {rule['record_count']} videos, "
            f"threshold={threshold_text}, "
            f"common_lines={len(rule['boilerplate_description_lines'])}, "
            f"common_keywords={len(rule['boilerplate_keywords'])}"
        )

    print(
        "Summary: "
        f"specific_description_lines={total_specific_description_lines}, "
        f"context_description_lines={total_context_description_lines}, "
        f"removed_description_lines={total_removed_description_lines}, "
        f"specific_keywords={total_specific_keywords}, "
        f"common_keywords={total_common_keywords}"
    )
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
