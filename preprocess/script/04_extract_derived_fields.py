from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_OUTPUT_DIR,
    extract_youtube_id,
    normalize_text,
    parse_date,
    parse_date_from_title,
    read_jsonl,
    seconds_to_hms,
    write_csv,
    write_json,
    write_jsonl,
)

WEEKDAY_VI = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}

# Chỉ ánh xạ khi nguồn là đủ rõ ràng. Không suy diễn Tuổi Trẻ/Thanh Niên là “đài”.
AUTHOR_TO_NETWORK = {
    "60 giây official": "HTV",
    "htv sports": "HTV",
    "htv entertainment": "HTV",
    "htv giải trí": "HTV",
}

SLOT_RE = re.compile(r"\b(Sáng|Trưa|Chiều|Tối|Đêm)\b", flags=re.IGNORECASE)
DATE_MARKER_RE = re.compile(
    r"\s*[-–—|:]\s*(?:ngày\s*)?\d{2}[/-]?\d{2}[/-]?(?:20)?\d{2,4}\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract conservative structured fields from cleaned metadata."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "03_clean_metadata.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = normalize_text(str(item))
        if text:
            output.append(text)
    return output


def extract_series_and_slot(title: str) -> tuple[str | None, str | None, str | None]:
    """
    Trích xuất bảo thủ để tránh coi tiêu đề sự kiện là tên chương trình.

    Chỉ suy ra series khi title có cấu trúc rõ ràng kiểu:
        "60 Giây Sáng - Ngày 01082024 - ..."
        "Tên chương trình - 01/08/2024 - ..."

    Trả về: (series_name, broadcast_slot, series_name_source)
    """
    title = normalize_text(title)
    if not title:
        return None, None, None

    slot_match = SLOT_RE.search(title)
    slot = slot_match.group(1).capitalize() if slot_match else None

    marker = DATE_MARKER_RE.search(title)
    if not marker:
        # Không có marker ngày rõ ràng: chỉ giữ slot, không đoán series.
        return None, slot, None

    head = title[: marker.start()].strip(" -–—|:")
    if not head:
        return None, slot, None

    if slot_match and slot_match.start() < marker.start():
        # Bỏ từ chỉ buổi khỏi tên series, ví dụ "60 Giây Sáng" -> "60 Giây".
        head_slot_match = SLOT_RE.search(head)
        if head_slot_match:
            head = (
                head[: head_slot_match.start()] + head[head_slot_match.end() :]
            ).strip(" -–—|:")

    return (head or None), slot, "title_date_pattern"


def infer_source_network(clean: dict[str, Any]) -> tuple[str | None, str | None]:
    author = normalize_text(str(clean.get("author") or ""))
    mapped = AUTHOR_TO_NETWORK.get(author.casefold())
    if mapped:
        return mapped, "author_mapping"

    # Chỉ dùng tín hiệu rất rõ từ tên tác giả; không quét description để tránh suy diễn sai.
    if re.search(r"\bhtv\b", author, flags=re.IGNORECASE):
        return "HTV", "author_token"

    return None, None


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_file)

    output_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    mismatch_video_ids: list[str] = []

    for record in records:
        video_id = str(record.get("video_id") or "")
        clean_raw = record.get("clean_metadata")
        clean = clean_raw if isinstance(clean_raw, dict) else {}

        if not clean:
            issue_counts["missing_or_invalid_clean_metadata"] += 1

        title = normalize_text(str(clean.get("title") or ""))
        publish_dt = parse_date(clean.get("publish_date"))
        title_dt = parse_date_from_title(title)
        episode_dt = title_dt or publish_dt

        series_name, broadcast_slot, series_name_source = extract_series_and_slot(title)
        source_network, source_network_source = infer_source_network(clean)
        youtube_id = extract_youtube_id(clean.get("watch_url"))

        title_publish_date_match: bool | None = None
        if title_dt and publish_dt:
            title_publish_date_match = title_dt.date() == publish_dt.date()
            if not title_publish_date_match:
                issue_counts["title_publish_date_mismatch"] += 1
                mismatch_video_ids.append(video_id)

        if not title:
            issue_counts["missing_title"] += 1
        if not episode_dt:
            issue_counts["missing_episode_date"] += 1
        if not youtube_id:
            issue_counts["missing_youtube_id"] += 1
        if series_name is None:
            issue_counts["series_name_not_inferred"] += 1
        if source_network is None:
            issue_counts["source_network_not_inferred"] += 1

        description_context = safe_list(clean.get("description_context"))
        keywords = safe_list(clean.get("keywords"))
        keywords_common = safe_list(clean.get("keywords_common"))

        derived = {
            "series_name": series_name,
            "series_name_source": series_name_source,
            "broadcast_slot": broadcast_slot,
            "episode_date": episode_dt.strftime("%Y-%m-%d") if episode_dt else None,
            "year": episode_dt.year if episode_dt else None,
            "month": episode_dt.month if episode_dt else None,
            "day": episode_dt.day if episode_dt else None,
            "weekday": WEEKDAY_VI.get(episode_dt.weekday()) if episode_dt else None,
            "channel_name": clean.get("author") or None,
            "channel_id": clean.get("channel_id") or None,
            "source_network": source_network,
            "source_network_source": source_network_source,
            "duration_seconds": clean.get("length"),
            "duration_hms": seconds_to_hms(clean.get("length")),
            "youtube_id": youtube_id,
            "date_source": "title" if title_dt else ("publish_date" if publish_dt else None),
            "title_publish_date_match": title_publish_date_match,
            "description_context_count": len(description_context),
            "specific_keyword_count": len(keywords),
            "common_keyword_count": len(keywords_common),
        }

        # Giữ nguyên toàn bộ record đầu vào, bao gồm description_context/keywords_common.
        enriched = {**record, "derived_metadata": derived}
        output_rows.append(enriched)

        csv_rows.append(
            {
                "video_id": video_id,
                **derived,
                "title": title,
                "description": clean.get("description") or "",
                "description_context": " | ".join(description_context),
                "keywords": " | ".join(keywords),
                "keywords_common": " | ".join(keywords_common),
                "watch_url": clean.get("watch_url"),
                "thumbnail_url": clean.get("thumbnail_url"),
            }
        )

    output_jsonl = args.output_dir / "04_enriched_metadata.jsonl"
    output_csv = args.output_dir / "04_enriched_metadata.csv"
    output_summary = args.output_dir / "04_derived_summary.json"

    write_jsonl(output_jsonl, output_rows)
    write_csv(output_csv, csv_rows)
    write_json(
        output_summary,
        {
            "record_count": len(records),
            "issue_counts": dict(sorted(issue_counts.items())),
            "title_publish_date_mismatch_video_ids": mismatch_video_ids,
            "output_jsonl": str(output_jsonl),
            "output_csv": str(output_csv),
        },
    )

    print(f"Extracted derived fields for {len(records)} records")
    print(f"Output: {args.output_dir.resolve()}")
    if issue_counts:
        print("Diagnostics:")
        for key, count in sorted(issue_counts.items()):
            print(f"- {key}: {count}")


if __name__ == "__main__":
    main()
