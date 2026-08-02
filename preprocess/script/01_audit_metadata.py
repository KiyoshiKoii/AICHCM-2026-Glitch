from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    REQUIRED_FIELDS,
    extract_youtube_id,
    list_json_files,
    parse_date,
    parse_date_from_title,
    read_json,
    safe_len,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw YouTube metadata JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def detect_missing_sequence(video_ids: list[str]) -> list[str]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for video_id in video_ids:
        match = re.fullmatch(r"(.+_V)(\d+)", video_id)
        if match:
            number_text = match.group(2)
            groups[match.group(1)].append((int(number_text), len(number_text)))

    missing: list[str] = []
    for prefix, number_items in groups.items():
        if not number_items:
            continue
        numbers = [number for number, _ in number_items]
        width = max(width for _, width in number_items)
        for number in range(min(numbers), max(numbers) + 1):
            if number not in numbers:
                missing.append(f"{prefix}{number:0{width}d}")
    return sorted(missing)


def main() -> None:
    args = parse_args()
    files = list_json_files(args.input_dir)

    rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, str]] = []
    watch_urls: list[str] = []
    youtube_ids: list[str] = []
    video_ids: list[str] = []

    for path in files:
        video_id = path.stem
        video_ids.append(video_id)
        try:
            data = read_json(path)
        except Exception as exc:  # Keep auditing remaining files.
            load_errors.append({"file": str(path), "error": str(exc)})
            continue

        missing_fields = sorted(REQUIRED_FIELDS - set(data))
        null_fields = sorted(key for key in REQUIRED_FIELDS if key in data and data[key] is None)

        publish_dt = parse_date(data.get("publish_date"))
        title_dt = parse_date_from_title(str(data.get("title", "")))
        date_match = None
        if publish_dt and title_dt:
            date_match = publish_dt.date() == title_dt.date()

        watch_url = str(data.get("watch_url") or "").strip()
        youtube_id = extract_youtube_id(watch_url)
        if watch_url:
            watch_urls.append(watch_url)
        if youtube_id:
            youtube_ids.append(youtube_id)

        length_value = data.get("length")
        try:
            length_seconds = int(length_value)
            length_valid = length_seconds >= 0
        except (TypeError, ValueError):
            length_seconds = None
            length_valid = False

        rows.append(
            {
                "video_id": video_id,
                "file": str(path),
                "load_ok": True,
                "missing_fields": "|".join(missing_fields),
                "null_fields": "|".join(null_fields),
                "schema_ok": not missing_fields and not null_fields,
                "author": data.get("author"),
                "channel_id": data.get("channel_id"),
                "title": data.get("title"),
                "publish_date": data.get("publish_date"),
                "title_date": title_dt.strftime("%Y-%m-%d") if title_dt else None,
                "publish_date_parsed": publish_dt.strftime("%Y-%m-%d") if publish_dt else None,
                "date_match": date_match,
                "length_seconds": length_seconds,
                "length_valid": length_valid,
                "description_chars": safe_len(data.get("description")),
                "keyword_count": len(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else None,
                "watch_url": watch_url or None,
                "youtube_id": youtube_id,
            }
        )

    watch_counts = Counter(watch_urls)
    youtube_counts = Counter(youtube_ids)
    for row in rows:
        row["duplicate_watch_url"] = bool(row["watch_url"] and watch_counts[row["watch_url"]] > 1)
        row["duplicate_youtube_id"] = bool(row["youtube_id"] and youtube_counts[row["youtube_id"]] > 1)

    missing_sequence = detect_missing_sequence(video_ids)
    summary = {
        "input_dir": str(args.input_dir),
        "json_file_count": len(files),
        "loaded_count": len(rows),
        "load_error_count": len(load_errors),
        "schema_error_count": sum(not bool(row["schema_ok"]) for row in rows),
        "date_mismatch_count": sum(row["date_match"] is False for row in rows),
        "invalid_length_count": sum(not bool(row["length_valid"]) for row in rows),
        "duplicate_watch_url_count": sum(bool(row["duplicate_watch_url"]) for row in rows),
        "duplicate_youtube_id_count": sum(bool(row["duplicate_youtube_id"]) for row in rows),
        "missing_video_ids_in_detected_ranges": missing_sequence,
    }

    write_csv(args.output_dir / "01_metadata_audit.csv", rows)
    write_csv(args.output_dir / "01_load_errors.csv", load_errors, ["file", "error"])
    write_json(args.output_dir / "01_audit_summary.json", summary)

    print(f"Audited {len(files)} JSON files")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Missing sequence IDs: {missing_sequence or 'none'}")


if __name__ == "__main__":
    main()
