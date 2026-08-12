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
import re
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# ROOT — thư mục chứa script này (src/visual_pipeline/)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# config.py cũng có thể được chạy trực tiếp trên PowerShell Windows.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


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


def _get_positive_int(name: str, default: int) -> int:
    """Read a positive integer setting from the environment/.env file."""
    raw_value = _get_env_value(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} phải là số nguyên dương, nhận: {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} phải lớn hơn 0, nhận: {value}")
    return value


def _safe_name(value: str) -> str:
    """Create a stable directory/collection suffix from an env-provided value."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "default"

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

# Thư mục lưu Qdrant local database — tự tạo nếu chưa có
QDRANT_DB_PATH = os.path.join(BASE_DIR, "local_qdrant_db")

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Model & DB
# ──────────────────────────────────────────────────────────────────────────────
VISUAL_MODEL_ID = _get_env_value("VISUAL_MODEL_ID") or "Qwen/Qwen3-VL-Embedding-8B"
# Khuyến nghị đặt biến này thành Hugging Face commit SHA trên cloud để extractor
# và server luôn dùng đúng cùng checkpoint. "main" được giữ làm mặc định để dễ bắt đầu.
VISUAL_MODEL_REVISION = _get_env_value("VISUAL_MODEL_REVISION") or "main"
VECTOR_SIZE = _get_positive_int("VISUAL_VECTOR_SIZE", 512)
BATCH_SIZE = _get_positive_int("VISUAL_BATCH_SIZE", 1)
VISUAL_DTYPE = (_get_env_value("VISUAL_DTYPE") or "auto").lower()
if VISUAL_DTYPE not in {"auto", "bfloat16", "float16", "float32"}:
    raise ValueError(
        "VISUAL_DTYPE chỉ hỗ trợ auto, bfloat16, float16 hoặc float32; "
        f"nhận: {VISUAL_DTYPE!r}"
    )

# Flash Attention chỉ được bật khi image CUDA cloud đã cài flash-attn tương thích.
VISUAL_ATTN_IMPLEMENTATION = _get_env_value("VISUAL_ATTN_IMPLEMENTATION")
QUERY_INSTRUCTION = (
    _get_env_value("VISUAL_QUERY_INSTRUCTION")
    or "Retrieve images relevant to the user's visual search query."
)

# Tách artifact/index Qwen ra khỏi CLIP cũ. Đổi revision, dimension hoặc
# VISUAL_ARTIFACT_ID sẽ tạo namespace mới, nên resume không thể dùng nhầm .npy.
_default_artifact_id = (
    f"{_safe_name(VISUAL_MODEL_ID)}_{_safe_name(VISUAL_MODEL_REVISION)}_d{VECTOR_SIZE}"
)
VISUAL_ARTIFACT_ID = _safe_name(
    _get_env_value("VISUAL_ARTIFACT_ID") or _default_artifact_id
)

# Thư mục lưu .npy Qwen — tự tạo nếu chưa có.
NPY_DIR = os.path.join(DATA_DIR, "npy_features", VISUAL_ARTIFACT_ID)
EXTRACTION_METADATA_PATH = os.path.join(NPY_DIR, "metadata.json")

# Collection mới tuyệt đối không trộn vector Qwen với vector CLIP cũ.
COLLECTION_NAME = (
    _get_env_value("VISUAL_COLLECTION_NAME")
    or f"kis_images_{VISUAL_ARTIFACT_ID}"
)
INDEX_METADATA_PATH = os.path.join(
    QDRANT_DB_PATH,
    f"{COLLECTION_NAME}.metadata.json",
)

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
    print(f"  QDRANT_DB_PATH: {QDRANT_DB_PATH}")
    print(f"  MODEL         : {VISUAL_MODEL_ID}")
    print(f"  REVISION      : {VISUAL_MODEL_REVISION}")
    print(f"  ARTIFACT      : {VISUAL_ARTIFACT_ID}")
    print(f"  COLLECTION    : {COLLECTION_NAME}")
    print(f"  VECTOR_SIZE   : {VECTOR_SIZE}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print(f"  DTYPE         : {VISUAL_DTYPE}")
    print(f"  ATTN          : {VISUAL_ATTN_IMPLEMENTATION or 'default'}")
    print(f"  QUERY INSTRUCT: {QUERY_INSTRUCTION}")
    print("=" * 60)

    # Kiểm tra trạng thái các thư mục
    print("\n📊 TRẠNG THÁI THƯ MỤC:")
    dirs = {
        "keyframes": KEYFRAME_DIR,
        "npy_features": NPY_DIR,
        "local_qdrant_db": QDRANT_DB_PATH,
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
