from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_INPUT_DIR = Path("media-info-aic25-b1") / "media-info"
DEFAULT_OUTPUT_DIR = Path("preprocess") / "output" / "metadata"

REQUIRED_FIELDS = {
    "author",
    "channel_id",
    "channel_url",
    "description",
    "keywords",
    "length",
    "publish_date",
    "thumbnail_url",
    "title",
    "watch_url",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object, got {type(data).__name__}")
    return data


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL item at {path}:{line_no} is not an object")
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def list_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir.resolve()}\n"
            "Run the command from project root or pass --input-dir explicitly."
        )
    return sorted(p for p in input_dir.rglob("*.json") if p.is_file())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return normalize_space(text)


def normalize_for_compare(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics while preserving đ/Đ as d/D."""
    text = str(text).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    text = strip_accents(unicodedata.normalize("NFC", text).casefold())
    return re.findall(r"[a-z0-9]+", text)


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = normalize_space(str(value))
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def parse_date_from_title(title: str) -> datetime | None:
    title = normalize_text(title)

    # Examples: 01082024, 01/08/2024, 01-08-2024
    match = re.search(r"(?<!\d)(\d{2})[/-]?(\d{2})[/-]?(20\d{2})(?!\d)", title)
    if not match:
        return None

    day, month, year = match.groups()
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def extract_youtube_id(watch_url: Any) -> str | None:
    if not watch_url:
        return None
    text = str(watch_url).strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return None

    host = parsed.netloc.casefold()
    if "youtu.be" in host:
        return parsed.path.strip("/") or None
    if "youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v")
        if query_id:
            return query_id[0]
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return parts[1]
    return None


def seconds_to_hms(value: Any) -> str | None:
    try:
        total = int(value)
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value)
    return len(str(value))
