"""
config.py — Cấu hình đường dẫn tập trung cho Audio Pipeline (ASR)
=============================================================
Khi thêm data mới hoặc đổi vị trí thư mục dữ liệu,
CHỈ CẦN SỬA FILE NÀY. Không cần động vào extractor.py.

Cấu trúc thư mục chuẩn (mặc định):
    src/audio_pipeline/
    ├── config.py
    ├── extractor.py
    └── tests/
    data/
    ├── videos/                 ← thả các file video gốc vào đây (VD: L21_V001.mp4)
    └── metadata/
        └── metadata_asr.json   ← tự sinh khi chạy extractor.py
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# ROOT — thư mục chứa script này (src/audio_pipeline/)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# [CÓ THỂ SỬA] DATA_DIR — thư mục gốc chứa toàn bộ dữ liệu
# Mặc định: <repo_root>/data/
# Nếu muốn trỏ ra ngoài, thay bằng đường dẫn tuyệt đối, ví dụ:
#   DATA_DIR = r"D:\MyDataset\aic2026"
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data")

# ──────────────────────────────────────────────────────────────────────────────
# Các đường dẫn con — không cần sửa nếu bạn giữ nguyên cấu trúc bên trên
# ──────────────────────────────────────────────────────────────────────────────
# Thư mục chứa video gốc (VD: L21_V001.mp4, L21_V002.mp4...)
VIDEO_DIR = os.path.join(DATA_DIR, "videos")

# Thư mục chứa các file metadata output (chung với các pipeline khác)
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# File output cuối cùng — theo đúng format quy định trong task.md của Dev 4
OUTPUT_PATH = os.path.join(METADATA_DIR, "metadata_asr.json")

# Các đuôi file video được coi là hợp lệ khi quét VIDEO_DIR
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Model faster-whisper
# ──────────────────────────────────────────────────────────────────────────────
# "large-v3" cho độ chính xác tốt nhất (Việt + Anh) nhưng cần GPU/VRAM lớn.
# Máy yếu hơn / chỉ chạy CPU nên đổi sang "medium" hoặc "small" qua --model-size.
WHISPER_MODEL_SIZE = "large-v3"

# device/compute_type mặc định: tự chọn theo GPU có sẵn hay không (xem extractor.py)
DEFAULT_GPU_COMPUTE_TYPE = "float16"
DEFAULT_CPU_COMPUTE_TYPE = "int8"

# beam_size mặc định cho decode (càng cao càng chính xác nhưng càng chậm)
BEAM_SIZE = 5


def print_config():
    """In ra tất cả các đường dẫn đang dùng để dễ debug."""
    print("=" * 60)
    print("🎙️  AUDIO PIPELINE (ASR) — CẤU HÌNH ĐƯỜNG DẪN")
    print("=" * 60)
    print(f"  BASE_DIR      : {BASE_DIR}")
    print(f"  DATA_DIR      : {DATA_DIR}")
    print(f"  VIDEO_DIR     : {VIDEO_DIR}")
    print(f"  OUTPUT_PATH   : {OUTPUT_PATH}")
    print(f"  MODEL SIZE    : {WHISPER_MODEL_SIZE}")
    print(f"  BEAM_SIZE     : {BEAM_SIZE}")
    print("=" * 60)

    print("\n📊 TRẠNG THÁI THƯ MỤC:")
    for name, path in {"videos": VIDEO_DIR, "metadata": METADATA_DIR}.items():
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
