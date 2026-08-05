"""
config.py — Cấu hình đường dẫn tập trung cho Audio Pipeline (ASR)
=============================================================
Khi thêm data mới hoặc đổi vị trí thư mục dữ liệu,
CHỈ CẦN SỬA FILE NÀY. Không cần động vào extractor.py.

Cấu trúc thư mục chuẩn (mặc định):
    src/audio_pipeline/
    ├── config.py
    ├── audio.py            ← tách audio bằng ffmpeg
    ├── extractor.py
    ├── convert_model.py    ← convert PhoWhisper sang CTranslate2 (chạy 1 lần)
    └── tests/
    data/
    ├── videos/                 ← thả các file video gốc vào đây (VD: L21_V001.mp4)
    ├── audio_cache/            ← WAV 16kHz tạm, tự sinh & tự xoá khi chạy
    ├── models/                 ← model PhoWhisper đã convert (tự sinh)
    └── metadata/
        ├── metadata_asr/       ← output chính: mỗi video 1 file (extractor.py)
        │   ├── L21_V001.json
        │   └── L21_V002.json
        └── metadata_asr.json   ← file gộp bàn giao Dev 2 (merge_asr.py)
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

# Thư mục output chính: MỖI VIDEO 1 FILE (VD: metadata_asr/L21_V001.json).
# Tách nhỏ thay vì gom vào 1 file lớn vì:
#   - Checkpoint sau mỗi video chỉ ghi ~20KB thay vì ghi đè lại cả file ~17MB
#     (gom 1 file thì với 873 video sẽ thành O(n²) lượt ghi đĩa).
#   - Resume chỉ cần kiểm tra file tồn tại, không phải parse cả file lớn.
#   - Nhiều người chạy song song trên các máy khác nhau rồi gộp lại không bị đụng file.
OUTPUT_DIR = os.path.join(METADATA_DIR, "metadata_asr")

# File gộp cuối cùng — ĐÚNG format mảng JSON quy định trong task.md của Dev 4,
# sinh ra từ OUTPUT_DIR bằng `python merge_asr.py` để bàn giao cho Dev 2.
OUTPUT_PATH = os.path.join(METADATA_DIR, "metadata_asr.json")

# Thư mục đệm chứa WAV 16kHz tách ra từ video. File được xoá ngay sau khi
# transcribe xong (trừ khi chạy --keep-audio) nên không tốn dung lượng lâu dài.
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "audio_cache")

# Thư mục chứa model đã convert sang CTranslate2 (xem convert_model.py)
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Các đuôi file video được coi là hợp lệ khi quét VIDEO_DIR
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Model
# ──────────────────────────────────────────────────────────────────────────────
# PhoWhisper (VinAI) fine-tune Whisper trên 844 giờ tiếng Việt đa giọng vùng miền,
# đạt SOTA trên các benchmark ASR tiếng Việt — WER thấp hơn hẳn Whisper gốc
# (VD trên CMV-Vi: PhoWhisper-large 8.14 so với Whisper-large-v3 ~2x cao hơn).
# Dataset của BTC là bản tin tiếng Việt nên đây là lựa chọn mặc định.
#
# PhoWhisper phân phối dưới dạng HuggingFace transformers, phải convert sang
# CTranslate2 mới chạy được bằng faster-whisper:
#     python convert_model.py
# Sau khi convert, model nằm ở data/models/PhoWhisper-large-ct2/.
PHOWHISPER_HF_ID = "vinai/PhoWhisper-large"
PHOWHISPER_CT2_DIR = os.path.join(MODEL_DIR, "PhoWhisper-large-ct2")

# Model mặc định. Có thể là:
#   - đường dẫn tới thư mục model CT2 đã convert (VD: PHOWHISPER_CT2_DIR)
#   - tên size Whisper chuẩn ("large-v3", "medium"...) — faster-whisper tự tải
#   - repo id của một model CT2 trên HuggingFace
# Nếu chưa convert PhoWhisper thì tự động fallback về WHISPER_FALLBACK_MODEL.
DEFAULT_MODEL = PHOWHISPER_CT2_DIR
WHISPER_FALLBACK_MODEL = "large-v3"

# PhoWhisper là model chuyên tiếng Việt (fine-tune monolingual) nên phải ép
# language="vi": vừa bỏ được bước detect language (nhanh hơn), vừa tránh model
# đoán nhầm sang ngôn ngữ khác. Đặt None để faster-whisper tự nhận diện
# (chỉ nên dùng khi chạy Whisper gốc đa ngữ trên data lẫn tiếng Anh).
DEFAULT_LANGUAGE = "vi"

# device/compute_type mặc định: tự chọn theo GPU có sẵn hay không (xem extractor.py)
DEFAULT_GPU_COMPUTE_TYPE = "float16"
DEFAULT_CPU_COMPUTE_TYPE = "int8"

# beam_size mặc định cho decode (càng cao càng chính xác nhưng càng chậm)
BEAM_SIZE = 5

# Độ dài (giây) mỗi chunk audio mà VAD cắt ra trước khi đưa vào model.
#
# ⚠️ QUAN TRỌNG với PhoWhisper: bản fine-tune của VinAI KHÔNG sinh được
# timestamp token như Whisper gốc (thử `without_timestamps=False` sẽ ra mốc thời
# gian rác kiểu [5.04-5.18] cho cả câu dài). Vì vậy mốc `start`/`end` của segment
# được lấy từ ranh giới chunk của VAD — tức là CHUNK_LENGTH quyết định luôn độ mịn
# của segment trong metadata_asr.json:
#     30 (mặc định của faster-whisper) -> segment dài ~25s, tìm mốc thời gian rất thô
#     10                               -> segment dài ~8s
#      5                               -> segment dài ~3.5s (tương đương Whisper gốc)
# Task yêu cầu segment để "hỗ trợ tìm kiếm mốc thời gian chính xác" nên chọn 5.
# Chunk càng nhỏ thì càng chậm (ít tận dụng được ngữ cảnh 30s của model).
CHUNK_LENGTH = 5

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình song song hoá (throughput)
# ──────────────────────────────────────────────────────────────────────────────
# Số chunk audio (do VAD cắt ra) được nhồi cùng lúc vào GPU mỗi lượt forward.
# Đây là đòn bẩy tăng tốc lớn nhất trên GPU: batch càng lớn càng tận dụng được
# nhân CUDA, đổi lại tốn thêm VRAM. Với model large float16 (~3GB) trên card
# 8GB thì batch 8 là an toàn; card 12GB+ có thể nâng lên 16.
BATCH_SIZE = 8

# Số luồng ffmpeg chạy nền để tách audio TRƯỚC cho các video kế tiếp, nhờ vậy
# GPU không phải ngồi chờ I/O giữa 2 video. 2 luồng là đủ để che lấp độ trễ.
AUDIO_WORKERS = 2

# Số video đã tách audio sẵn được giữ trong hàng đợi. Giữ nhỏ để không sinh ra
# hàng chục GB WAV cùng lúc (mỗi giờ video ≈ 115MB WAV 16kHz mono).
PREFETCH_QUEUE_SIZE = 2


def resolve_model(model: str | None = None) -> str:
    """Chọn model thực tế sẽ nạp, tự fallback nếu PhoWhisper chưa được convert."""
    if model:
        return model
    if os.path.isdir(DEFAULT_MODEL):
        return DEFAULT_MODEL
    print(f"⚠️  Chưa tìm thấy PhoWhisper đã convert tại: {DEFAULT_MODEL}")
    print(f"👉 Chạy `python convert_model.py` để có model tốt nhất cho tiếng Việt.")
    print(f"   Tạm thời dùng Whisper gốc: {WHISPER_FALLBACK_MODEL}\n")
    return WHISPER_FALLBACK_MODEL


def print_config():
    """In ra tất cả các đường dẫn đang dùng để dễ debug."""
    print("=" * 60)
    print("🎙️  AUDIO PIPELINE (ASR) — CẤU HÌNH ĐƯỜNG DẪN")
    print("=" * 60)
    print(f"  BASE_DIR      : {BASE_DIR}")
    print(f"  DATA_DIR      : {DATA_DIR}")
    print(f"  VIDEO_DIR     : {VIDEO_DIR}")
    print(f"  AUDIO_CACHE   : {AUDIO_CACHE_DIR}")
    print(f"  OUTPUT_DIR    : {OUTPUT_DIR}")
    print(f"  MODEL         : {resolve_model()}")
    print(f"  LANGUAGE      : {DEFAULT_LANGUAGE or 'auto-detect'}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print(f"  AUDIO_WORKERS : {AUDIO_WORKERS}")
    print(f"  BEAM_SIZE     : {BEAM_SIZE}")
    print("=" * 60)

    print("\n📊 TRẠNG THÁI THƯ MỤC:")
    for name, path in {
        "videos": VIDEO_DIR,
        "metadata": METADATA_DIR,
        "models": MODEL_DIR,
    }.items():
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
