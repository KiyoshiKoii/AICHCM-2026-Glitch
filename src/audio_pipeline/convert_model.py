"""
convert_model.py — Convert PhoWhisper sang định dạng CTranslate2 (chạy 1 lần)
============================================================================
PhoWhisper (VinAI) là model ASR tiếng Việt tốt nhất hiện có: fine-tune Whisper
trên 844 giờ tiếng Việt đa giọng vùng miền, đạt SOTA trên các benchmark ASR
tiếng Việt (WER 8.14 trên CMV-Vi, 4.67 trên VIVOS với bản large).

VinAI phát hành model dưới dạng HuggingFace transformers, trong khi faster-whisper
chạy trên CTranslate2 — nên phải convert 1 lần trước khi dùng.

    python convert_model.py                 # convert PhoWhisper-large (mặc định)
    python convert_model.py --model vinai/PhoWhisper-medium
    python convert_model.py --quantization int8_float16   # nhẹ VRAM hơn

Yêu cầu: `pip install torch transformers` (chỉ cần cho bước convert này, bản CPU
là đủ — convert chỉ đọc/ghi lại trọng số chứ không chạy inference).
Sau khi convert xong có thể gỡ torch/transformers nếu muốn tiết kiệm dung lượng.
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import MODEL_DIR, PHOWHISPER_CT2_DIR, PHOWHISPER_HF_ID

# faster-whisper cần các file tokenizer đi kèm model CT2 mới decode ra chữ được.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
]


def convert(hf_model_id: str, output_dir: Path, quantization: str = "float16",
            force: bool = False) -> Path:
    """Convert model HuggingFace sang định dạng CTranslate2 cho faster-whisper.

    Gọi thẳng API Python của ctranslate2 thay vì lệnh `ct2-transformers-converter`:
    lệnh đó nằm trong thư mục Scripts/ của env, sẽ không tìm thấy nếu người dùng
    chạy bằng đường dẫn python.exe tuyệt đối hay qua `conda run`.
    """
    if output_dir.exists():
        if not force:
            print(f"✅ Model đã có sẵn tại: {output_dir}")
            print("   Dùng --force để convert lại từ đầu.")
            return output_dir
        print(f"🗑️  Xoá model cũ tại: {output_dir}")
        shutil.rmtree(output_dir)

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print("❌ Chưa cài ctranslate2. Cài đặt: pip install ctranslate2")
        sys.exit(1)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"⏳ Đang convert {hf_model_id} -> {output_dir} (quantization={quantization})")
    print("   Bước này tải ~3GB trọng số từ HuggingFace, có thể mất vài phút...\n")

    try:
        converter = TransformersConverter(
            hf_model_id,
            copy_files=TOKENIZER_FILES,
            load_as_float16=quantization.startswith("float16"),
            low_cpu_mem_usage=True,
        )
        converter.convert(str(output_dir), quantization=quantization, force=force)
    except ImportError as exc:
        print(f"\n❌ Thiếu thư viện cho bước convert: {exc}")
        print("👉 Cài đặt: pip install torch transformers")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Convert thất bại: {exc}")
        sys.exit(1)

    print(f"\n✅ Convert xong! Model nằm tại: {output_dir}")
    print("   extractor.py sẽ tự động dùng model này.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PhoWhisper (HuggingFace) sang CTranslate2 cho faster-whisper"
    )
    parser.add_argument("--model", default=PHOWHISPER_HF_ID,
                         help=f"Repo id trên HuggingFace (mặc định: {PHOWHISPER_HF_ID})")
    parser.add_argument("--output-dir", default=None,
                         help=f"Thư mục output (mặc định: {PHOWHISPER_CT2_DIR})")
    parser.add_argument("--quantization", default="float16",
                         choices=["float32", "float16", "int8", "int8_float16"],
                         help="float16 cho GPU, int8 cho CPU / tiết kiệm VRAM")
    parser.add_argument("--force", action="store_true", help="Convert lại dù đã có sẵn")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.model == PHOWHISPER_HF_ID:
        output_dir = Path(PHOWHISPER_CT2_DIR)
    else:
        # Model khác mặc định -> đặt tên thư mục theo tên repo để không đè lên nhau
        output_dir = Path(MODEL_DIR) / f"{args.model.split('/')[-1]}-ct2"

    convert(args.model, output_dir, args.quantization, args.force)
