"""
config.py — Cấu hình đường dẫn tập trung cho Visual Pipeline
=============================================================
Khi thêm data mới hoặc đổi vị trí thư mục dữ liệu,
CHỈ CẦN SỬA FILE NÀY. Không cần động vào extractor.py / database.py.

Cấu trúc thư mục chuẩn (mặc định):
    src/visual_pipeline/
    ├── config.py
    ├── extractor.py
    ├── database.py
    ├── server.py
    ├── data/
    │   ├── keyframes/          ← thả các folder video vào đây (VD: L25_V001/, L25_V002/)
    │   └── npy_features/       ← tự sinh khi chạy extractor.py
    └── local_qdrant_db/        ← tự sinh khi chạy database.py
"""

import os
from pathlib import Path

from qdrant_client import QdrantClient

# ──────────────────────────────────────────────────────────────────────────────
# ROOT — thư mục chứa script này (src/visual_pipeline/)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_env_value(name: str) -> str | None:
    """Read an exported variable first, then the repository `.env` file."""
    if value := os.getenv(name):
        return value

    dotenv_path = Path(BASE_DIR).parents[1] / ".env"
    if not dotenv_path.is_file():
        return None
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None

# ──────────────────────────────────────────────────────────────────────────────
# [CÓ THỂ SỬA] DATA_DIR — thư mục gốc chứa toàn bộ dữ liệu
# Mặc định: src/visual_pipeline/data/
# Nếu muốn trỏ ra ngoài, thay bằng đường dẫn tuyệt đối, ví dụ:
#   DATA_DIR = r"D:\MyDataset\aic2026"
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data")

# ──────────────────────────────────────────────────────────────────────────────
# Các đường dẫn con — không cần sửa nếu bạn giữ nguyên cấu trúc bên trên
# ──────────────────────────────────────────────────────────────────────────────
# Thư mục chứa các folder keyframe của từng video (VD: L25_V001/, L25_V002/...)
KEYFRAME_DIR = os.path.join(DATA_DIR, "keyframes")

# Thư mục lưu file .npy (vector features) — tự tạo nếu chưa có
NPY_DIR = os.path.join(DATA_DIR, "npy_features")

# Thư mục lưu Qdrant local database — chỉ dùng khi không đặt QDRANT_URL.
QDRANT_DB_PATH = os.path.join(BASE_DIR, "local_qdrant_db")

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Model & DB
# ──────────────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID = _get_env_value("VISUAL_CLIP_MODEL") or "openai/clip-vit-base-patch32"
# Khi chạy Qdrant bằng Docker, đặt ví dụ:
#   QDRANT_URL=http://127.0.0.1:6333
# Không đặt biến này sẽ giữ tương thích với local embedded database cũ.
QDRANT_URL = _get_env_value("QDRANT_URL")
QDRANT_TIMEOUT_SECONDS = float(_get_env_value("QDRANT_TIMEOUT_SECONDS") or "10")
COLLECTION_NAME = "kis_images"
VECTOR_SIZE = 512
BATCH_SIZE = 32


def qdrant_target_description() -> str:
    """Return a safe, human-readable description of the active Qdrant target."""
    return QDRANT_URL or QDRANT_DB_PATH


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for Docker/remote Qdrant or the legacy local DB.

    A URL is deliberately opt-in so existing environments keep working until
    they add ``QDRANT_URL``.  Docker mode avoids opening the multi-GB embedded
    RocksDB from every Python process on Windows.
    """
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT_SECONDS)

    os.makedirs(QDRANT_DB_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_DB_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Tiện ích: In tóm tắt cấu hình khi cần debug
# ──────────────────────────────────────────────────────────────────────────────
def print_config():
    """In ra tất cả các đường dẫn đang dùng để dễ debug."""
    print("=" * 60)
    print("📁 VISUAL PIPELINE — CẤU HÌNH ĐƯỜNG DẪN")
    print("=" * 60)
    print(f"  BASE_DIR      : {BASE_DIR}")
    print(f"  DATA_DIR      : {DATA_DIR}")
    print(f"  KEYFRAME_DIR  : {KEYFRAME_DIR}")
    print(f"  NPY_DIR       : {NPY_DIR}")
    print(f"  QDRANT TARGET : {qdrant_target_description()}")
    print(f"  QDRANT MODE   : {'Docker/remote' if QDRANT_URL else 'local embedded'}")
    print(f"  MODEL         : {CLIP_MODEL_ID}")
    print(f"  COLLECTION    : {COLLECTION_NAME}")
    print(f"  VECTOR_SIZE   : {VECTOR_SIZE}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print("=" * 60)

    # Kiểm tra trạng thái các thư mục
    print("\n📊 TRẠNG THÁI THƯ MỤC:")
    dirs = {
        "keyframes": KEYFRAME_DIR,
        "npy_features": NPY_DIR,
        "local_qdrant_db (legacy)": QDRANT_DB_PATH,
    }
    for name, path in dirs.items():
        if os.path.exists(path):
            try:
                count = len(os.listdir(path))
                print(f"  ✅ {name}: tồn tại ({count} mục)")
            except Exception:
                print(f"  ✅ {name}: tồn tại")
        else:
            print(f"  📭 {name}: chưa có (sẽ tự tạo khi cần)")
    print()

if __name__ == "__main__":
    print_config()
