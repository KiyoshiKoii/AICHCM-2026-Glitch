"""
merge_asr.py — Gộp các file ASR lẻ thành 1 file bàn giao cho Dev 2
==================================================================
`extractor.py` ghi mỗi video 1 file trong `data/metadata/metadata_asr/` để
checkpoint nhanh và resume gọn. Nhưng task.md của Dev 4 quy định output bàn giao
phải là **1 mảng JSON** tại `data/metadata/metadata_asr.json` (Dev 2 nạp thẳng
vào BM25 / Elasticsearch). Script này lo phần chuyển đổi đó.

    python merge_asr.py                  # metadata_asr/*.json -> metadata_asr.json
    python merge_asr.py --check          # chỉ kiểm tra, không ghi file
    python merge_asr.py --split          # chiều ngược lại: tách file gộp ra thư mục

`--split` dùng khi bạn đang có sẵn file gộp cũ và muốn chuyển sang cấu trúc mới
mà không phải transcribe lại từ đầu.
"""

import argparse
import json
import sys
from pathlib import Path

from config import OUTPUT_DIR, OUTPUT_PATH

REQUIRED_KEYS = {"video_name", "full_transcript", "segments"}


def load_records(input_dir: Path) -> list[dict]:
    """Đọc toàn bộ file JSON lẻ, sắp xếp theo video_name cho ổn định giữa các lần chạy."""
    records = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ⚠️  Bỏ qua {path.name}: JSON hỏng ({exc})")
            continue

        missing = REQUIRED_KEYS - record.keys()
        if missing:
            print(f"  ⚠️  Bỏ qua {path.name}: thiếu trường {sorted(missing)}")
            continue

        # Tên file phải khớp video_name, nếu lệch là dấu hiệu file bị đổi tên tay.
        if record["video_name"] != path.stem:
            print(f"  ⚠️  {path.name}: video_name='{record['video_name']}' không khớp tên file")

        records.append(record)

    records.sort(key=lambda r: r["video_name"])
    return records


def merge(input_dir: Path, output_path: Path, check_only: bool = False) -> list[dict]:
    if not input_dir.exists():
        print(f"❌ Chưa có thư mục output: {input_dir}")
        print("👉 Chạy `python extractor.py` để sinh dữ liệu trước.")
        sys.exit(1)

    records = load_records(input_dir)
    if not records:
        print(f"❌ Không tìm thấy record hợp lệ nào trong {input_dir}")
        sys.exit(1)

    total_segments = sum(len(r["segments"]) for r in records)
    empty = [r["video_name"] for r in records if not r["full_transcript"].strip()]

    print(f"📊 {len(records)} video, {total_segments} segments.")
    if empty:
        print(f"  ⚠️  {len(empty)} video không có lời thoại: {', '.join(empty[:5])}"
              + (" ..." if len(empty) > 5 else ""))

    if check_only:
        print("✅ Kiểm tra xong (--check nên không ghi file).")
        return records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.part")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Đã ghi {output_path} ({size_mb:.1f} MB)")
    return records


def split(input_path: Path, output_dir: Path) -> int:
    """Tách file gộp cũ thành các file lẻ (dùng khi migrate cấu trúc cũ sang mới)."""
    if not input_path.exists():
        print(f"❌ Không tìm thấy file gộp: {input_path}")
        sys.exit(1)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print(f"❌ {input_path} không phải mảng JSON.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        name = record.get("video_name")
        if not name:
            print("  ⚠️  Bỏ qua 1 record không có video_name")
            continue
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    print(f"✅ Đã tách {written} record vào {output_dir}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gộp metadata_asr/*.json thành metadata_asr.json (và ngược lại)"
    )
    parser.add_argument("--input-dir", default=OUTPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true",
                         help="Chỉ kiểm tra tính hợp lệ, không ghi file gộp")
    parser.add_argument("--split", action="store_true",
                         help="Chiều ngược lại: tách file gộp thành các file lẻ")
    args = parser.parse_args()

    if args.split:
        split(Path(args.output), Path(args.input_dir))
    else:
        merge(Path(args.input_dir), Path(args.output), args.check)
