"""Nguồn chân lý duy nhất cho mọi đường dẫn dữ liệu.

Không hardcode đường dẫn dataset ở bất kỳ chỗ nào khác — luôn đi qua module này.
Nhờ vậy lúc BTC thả thêm batch mới hoặc ai đó dời data sang ổ khác, chỉ phải sửa
đúng một file.

Thứ tự ưu tiên khi xác định gốc dữ liệu:
    1. Biến môi trường ``AIC_DATA_ROOT`` (cho ai để data ở ổ đĩa khác)
    2. ``<gốc repo>/data`` (mặc định, khớp với scripts/organize_data.ps1)

Xem thêm ``data/README.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

# paths.py -> common -> src -> <gốc repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.getenv("AIC_DATA_ROOT") or REPO_ROOT / "data")

# Data nguyên bản của BTC — CHỈ ĐỌC. Không ghi gì vào đây.
RAW = DATA_ROOT / "raw"
# Output do pipeline của team sinh ra — xoá lúc nào cũng được, chạy lại là có.
PROCESSED = DATA_ROOT / "processed"
# Repo/model tham khảo tải từ bên ngoài.
EXTERNAL = DATA_ROOT / "external"
OPENIMAGES = EXTERNAL / "openimages"
OPENIMAGES_HIERARCHY = OPENIMAGES / "bbox_labels_600_hierarchy.json"
OPENIMAGES_CLASS_DESCRIPTIONS = OPENIMAGES / "class-descriptions-boxable.csv"

KEYFRAMES = RAW / "keyframes"
CLIP_FEATURES = RAW / "clip-features-32"
MAP_KEYFRAMES = RAW / "map-keyframes"
MEDIA_INFO = RAW / "media-info"
OBJECTS = RAW / "objects"
MANIFEST = RAW / "MANIFEST.csv"

# Fixture test 24 frame, được commit lên git — KHÔNG phải data thật.
# Test (Task 5) chạy trên đây để clone về là chạy được ngay.
SAMPLE_FRAMES = REPO_ROOT / "src" / "semantic_pipeline" / "sample_frames"


# --- raw: tra cứu theo video_id -------------------------------------------------


def keyframe_dir(video_id: str) -> Path:
    """Thư mục chứa keyframe của một video: ``.../keyframes/L21_V001``."""
    return KEYFRAMES / video_id


def keyframe(video_id: str, index: int) -> Path:
    """Một keyframe cụ thể. ``index`` là số thứ tự BTC đánh, zero-pad 3 chữ số."""
    return KEYFRAMES / video_id / f"{index:03d}.jpg"


def clip_features(video_id: str) -> Path:
    """Ma trận CLIP feature của cả video: ``.../clip-features-32/L21_V001.npy``."""
    return CLIP_FEATURES / f"{video_id}.npy"


def map_keyframes(video_id: str) -> Path:
    """CSV ánh xạ keyframe -> frame gốc (cột: n, pts_time, fps, frame_idx)."""
    return MAP_KEYFRAMES / f"{video_id}.csv"


def media_info(video_id: str) -> Path:
    """Metadata video của BTC (title, url, ...)."""
    return MEDIA_INFO / f"{video_id}.json"


def objects_dir(video_id: str) -> Path:
    """Thư mục kết quả object detection của một video."""
    return OBJECTS / video_id


def objects(video_id: str, index: int) -> Path:
    """Object detection của một keyframe cụ thể."""
    return OBJECTS / video_id / f"{index:03d}.json"


def video_ids() -> list[str]:
    """Toàn bộ video_id xuất hiện ở BẤT KỲ pack nào trên đĩa, đã sắp xếp.

    Cố tình lấy hợp của mọi pack thay vì chỉ ``keyframes/``: BTC giao data lệch
    nhau giữa các pack (đợt b1 có metadata cho cả 873 video nhưng keyframes mới
    chỉ tới L22). Nếu lấy keyframes làm nguồn chuẩn thì hàng trăm video sẽ âm
    thầm biến mất khỏi mọi thống kê. Video nào thiếu gì thì tra MANIFEST.csv.

    Trả về list rỗng nếu chưa setup data — để caller tự báo lỗi cho thân thiện.
    """
    found: set[str] = set()
    for directory in (KEYFRAMES, OBJECTS):
        if directory.is_dir():
            found.update(p.name for p in directory.iterdir() if p.is_dir())
    for directory in (CLIP_FEATURES, MAP_KEYFRAMES, MEDIA_INFO):
        if directory.is_dir():
            found.update(p.stem for p in directory.iterdir() if p.is_file())
    return sorted(found)


def videos_with_keyframes() -> list[str]:
    """Chỉ những video thực sự có ảnh keyframe — dùng cho pipeline cần đọc ảnh."""
    if not KEYFRAMES.is_dir():
        return []
    return sorted(p.name for p in KEYFRAMES.iterdir() if p.is_dir())


# --- processed: output của team --------------------------------------------------


def processed_metadata_dir() -> Path:
    """Metadata OCR/entity/spatial chạy trên dataset thật (dev2)."""
    return PROCESSED / "metadata"


def objects_index(video_id: str) -> Path:
    """Objects của một video đã gộp thành 1 file.

    ``raw/objects/`` là hàng trăm nghìn file JSON nhỏ — NTFS đọc rất chậm. Bước
    tiền xử lý gộp lại theo video giúp giảm mạnh số lần chạm đĩa.
    """
    return PROCESSED / "objects_index" / f"{video_id}.parquet"


def embeddings_dir() -> Path:
    return PROCESSED / "embeddings"


def ensure_processed_dirs() -> None:
    """Tạo sẵn cây thư mục ``processed/``. An toàn khi gọi nhiều lần."""
    for path in (
        processed_metadata_dir(),
        PROCESSED / "objects_index",
        embeddings_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Tóm tắt trạng thái data — tiện in ra đầu mỗi script để debug."""
    lines = [f"DATA_ROOT = {DATA_ROOT}"]
    for label, path in (
        ("keyframes", KEYFRAMES),
        ("clip-features-32", CLIP_FEATURES),
        ("map-keyframes", MAP_KEYFRAMES),
        ("media-info", MEDIA_INFO),
        ("objects", OBJECTS),
    ):
        mark = "OK  " if path.is_dir() else "THIEU"
        count = ""
        if path.is_dir():
            count = f"({sum(1 for _ in path.iterdir())} muc)"
        lines.append(f"  [{mark}] {label:<18} {count:<12} {path}")
    lines.append(
        f"  tong {len(video_ids())} video, "
        f"{len(videos_with_keyframes())} video co keyframe"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
