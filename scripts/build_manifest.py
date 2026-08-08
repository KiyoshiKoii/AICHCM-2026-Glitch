"""Sinh data/raw/MANIFEST.csv - bản kiểm kê dataset BTC.

MANIFEST.csv là file DUY NHẤT trong data/raw/ được commit lên git. Nó nhỏ (vài chục
KB) nhưng trả lời được những câu hỏi mà nếu không có nó thì phải quét cả trăm nghìn
file mới biết:

  - máy này đang có những video nào, thuộc đợt phát hành nào
  - video nào thiếu clip-features / map-keyframes / objects (BTC giao thiếu, hoặc
    tải lỗi, hoặc copy dở)
  - fps và số keyframe của từng video, để tính lại timestamp lúc nộp bài

Chạy lại mỗi khi BTC thả thêm data:

    python scripts/build_manifest.py
    python scripts/build_manifest.py --check    # chỉ báo cáo core data, không ghi file
    python scripts/build_manifest.py --check --require-keyframes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import paths  # noqa: E402

# Mã đợt phát hành BTC nhúng trong tên thư mục: "objects-aic25-b1" -> "aic25-b1".
BATCH_PATTERN = re.compile(r"aic\d+-b\d+")

FIELDNAMES = [
    "video_id",
    "group",
    "batch",
    "n_keyframes",
    "fps",
    "n_objects",
    "has_clip_features",
    "has_map_keyframes",
    "has_media_info",
    "duration_sec",
]


def load_batches() -> dict[str, str]:
    """Đọc log organize_data.ps1 ghi lại, suy ra video thuộc đợt phát hành nào.

    Mỗi video xuất hiện nhiều lần trong log — một lần cho mỗi pack nó góp mặt
    (keyframes, objects, clip-features, ...). Nên KHÔNG được dựng dict thẳng từ
    tên thư mục: video nào cũng sẽ mang tên pack nào tình cờ được xử lý sau cùng.
    Thứ thực sự cần là mã đợt (``aic25-b1``), vốn giống nhau ở mọi pack cùng đợt.

    Các pack không mang mã đợt trong tên (``Keyframes_L21``) thì bỏ qua, để đợt
    suy được từ pack khác của cùng video.
    """
    path = paths.RAW / "_provenance.csv"
    if not path.is_file():
        return {}

    batches: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = row.get("video_id")
            if not video_id:
                continue
            match = BATCH_PATTERN.search(row.get("source_dir") or "")
            if match:
                batches[video_id] = match.group(0)
    return batches


def read_fps(video_id: str) -> float | None:
    """Lấy fps từ map-keyframes CSV (cột fps, hằng số trên mọi dòng)."""
    path = paths.map_keyframes(video_id)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("fps")
                if value:
                    return round(float(value), 3)
    except (ValueError, csv.Error):
        return None
    return None


def read_duration(video_id: str) -> float | None:
    """Độ dài video từ media-info, nếu BTC có ghi."""
    path = paths.media_info(video_id)
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(info, dict):
        return None
    for key in ("length", "duration", "duration_sec"):
        value = info.get(key)
        if isinstance(value, (int, float)):
            return round(float(value), 2)
    return None


def count_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for entry in directory.iterdir() if entry.is_file())


def build_rows() -> list[dict[str, object]]:
    batches = load_batches()
    rows: list[dict[str, object]] = []

    for video_id in paths.video_ids():
        rows.append(
            {
                "video_id": video_id,
                "group": video_id.split("_", 1)[0],
                "batch": batches.get(video_id, ""),
                "n_keyframes": count_files(paths.keyframe_dir(video_id)),
                "fps": read_fps(video_id) or "",
                "n_objects": count_files(paths.objects_dir(video_id)),
                "has_clip_features": int(paths.clip_features(video_id).is_file()),
                "has_map_keyframes": int(paths.map_keyframes(video_id).is_file()),
                "has_media_info": int(paths.media_info(video_id).is_file()),
                "duration_sec": read_duration(video_id) or "",
            }
        )
    return rows


def report(rows: list[dict[str, object]], *, require_keyframes: bool = False) -> int:
    """In tóm tắt. Trả về số video có vấn đề."""
    if not rows:
        print("Chua co video nao trong data/raw/keyframes/.")
        print("Chay: powershell -File scripts/organize_data.ps1 -Execute")
        return 0

    groups: dict[str, int] = {}
    for row in rows:
        groups[str(row["group"])] = groups.get(str(row["group"]), 0) + 1

    total_keyframes = sum(int(row["n_keyframes"]) for row in rows)
    videos_with_keyframes = sum(bool(row["n_keyframes"]) for row in rows)
    print(f"{len(rows)} video, {total_keyframes} keyframe")
    print("  theo nhom: " + ", ".join(f"{g}={n}" for g, n in sorted(groups.items())))
    print(
        f"  anh keyframe da gan: {videos_with_keyframes}/{len(rows)} video "
        "(khong bat buoc cho object pipeline)"
    )

    problems = []
    for row in rows:
        missing = [
            label
            for label, key in (
                ("clip-features", "has_clip_features"),
                ("map-keyframes", "has_map_keyframes"),
                ("media-info", "has_media_info"),
            )
            if not row[key]
        ]
        if not row["n_objects"]:
            missing.append("objects")
        if require_keyframes and not row["n_keyframes"]:
            missing.append("keyframes")
        if missing:
            problems.append((str(row["video_id"]), missing))

    if not problems:
        print("  khong co video nao thieu du lieu")
        return 0

    # Gom theo "thiếu đúng những gì" — 800 dòng liệt kê từng video không đọc nổi,
    # trong khi cái cần biết là BTC giao thiếu nguyên mảng nào.
    by_pattern: dict[str, list[str]] = {}
    for video_id, missing in problems:
        by_pattern.setdefault(", ".join(missing), []).append(video_id)

    print(f"\n  {len(problems)} video thieu du lieu:")
    for pattern, ids in sorted(by_pattern.items(), key=lambda kv: -len(kv[1])):
        groups = sorted({vid.split("_", 1)[0] for vid in ids})
        print(f"    thieu {pattern}: {len(ids)} video (nhom {', '.join(groups)})")
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="chi bao cao, khong ghi MANIFEST.csv",
    )
    parser.add_argument(
        "--require-keyframes",
        action="store_true",
        help="xem keyframe image la bat buoc (mac dinh object pipeline khong can anh)",
    )
    args = parser.parse_args()

    print(paths.describe())
    print()

    rows = build_rows()
    problems = report(rows, require_keyframes=args.require_keyframes)

    if args.check:
        return 1 if problems else 0

    if rows:
        paths.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with paths.MANIFEST.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDa ghi {paths.MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
