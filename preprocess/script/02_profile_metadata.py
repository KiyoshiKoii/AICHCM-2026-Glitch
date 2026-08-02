from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
import json
import re
import unicodedata

from common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    list_json_files,
    normalize_for_compare,
    normalize_text,
    read_json,
    safe_len,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile metadata fields and repeated boilerplate.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_field_value(value: Any) -> str:
    """
    Chuẩn hóa để so sánh giá trị field nhưng vẫn giữ nguyên URL.

    Khác normalize_for_compare():
    - Không thay URL thành <url>
    - Chỉ chuẩn hóa Unicode, chữ hoa/thường và khoảng trắng
    """
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def canonical_value(value: Any) -> str:
    if value is None:
        return "<null>"

    if isinstance(value, list):
        # Keyword không phụ thuộc thứ tự nên sort trước khi so sánh.
        normalized_items = sorted(
            normalize_field_value(item)
            for item in value
        )
        return json.dumps(
            normalized_items,
            ensure_ascii=False,
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    return normalize_field_value(value)


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for path in list_json_files(args.input_dir):
        data = read_json(path)
        records.append({"video_id": path.stem, **data})

    if not records:
        raise RuntimeError("No metadata records found")

    all_fields = sorted(set().union(*(record.keys() for record in records)) - {"video_id"})
    field_rows: list[dict[str, Any]] = []

    for field in all_fields:
        values = [record.get(field) for record in records]
        non_missing = [value for value in values if value not in (None, "", [])]
        canonical = [canonical_value(value) for value in non_missing]
        unique_count = len(set(canonical))
        duplicate_rate = 0.0 if not canonical else 1.0 - unique_count / len(canonical)

        field_rows.append(
            {
                "field": field,
                "record_count": len(records),
                "missing_count": len(records) - len(non_missing),
                "missing_rate": round((len(records) - len(non_missing)) / len(records), 6),
                "unique_count": unique_count,
                "unique_rate_non_missing": round(unique_count / len(non_missing), 6) if non_missing else None,
                "duplicate_rate_non_missing": round(duplicate_rate, 6) if non_missing else None,
                "avg_size": round(mean(safe_len(value) for value in non_missing), 3) if non_missing else None,
                "sample_value": str(non_missing[0])[:300] if non_missing else None,
            }
        )

    line_counter: Counter[str] = Counter()
    line_example: dict[str, str] = {}
    keyword_counter: Counter[str] = Counter()
    keyword_example: dict[str, str] = {}

    for record in records:
        description = str(record.get("description") or "")
        seen_lines: set[str] = set()
        for raw_line in description.splitlines():
            line = normalize_text(raw_line)
            normalized = normalize_for_compare(line)
            if not normalized or normalized in seen_lines:
                continue
            seen_lines.add(normalized)
            line_counter[normalized] += 1
            line_example.setdefault(normalized, line)

        seen_keywords: set[str] = set()
        keywords = record.get("keywords") or []
        if isinstance(keywords, list):
            for raw_keyword in keywords:
                keyword = normalize_text(str(raw_keyword))
                normalized = normalize_for_compare(keyword)
                if not normalized or normalized in seen_keywords:
                    continue
                seen_keywords.add(normalized)
                keyword_counter[normalized] += 1
                keyword_example.setdefault(normalized, keyword)

    description_rows = [
        {
            "normalized_line": normalized,
            "example_line": line_example[normalized],
            "document_frequency": count,
            "document_rate": round(count / len(records), 6),
        }
        for normalized, count in line_counter.most_common()
    ]
    keyword_rows = [
        {
            "normalized_keyword": normalized,
            "example_keyword": keyword_example[normalized],
            "document_frequency": count,
            "document_rate": round(count / len(records), 6),
        }
        for normalized, count in keyword_counter.most_common()
    ]

    write_csv(args.output_dir / "02_field_profile.csv", field_rows)
    write_csv(args.output_dir / "02_description_line_frequency.csv", description_rows)
    write_csv(args.output_dir / "02_keyword_frequency.csv", keyword_rows)
    write_json(
        args.output_dir / "02_profile_summary.json",
        {
            "record_count": len(records),
            "field_count": len(all_fields),
            "description_unique_line_count": len(line_counter),
            "keyword_unique_count": len(keyword_counter),
        },
    )

    print(f"Profiled {len(records)} records")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
