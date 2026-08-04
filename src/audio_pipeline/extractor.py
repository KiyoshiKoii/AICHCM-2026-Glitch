# Task (Phase 2, Dev 4): Trích xuất Script/Phụ đề (ASR) từ Video
# Pipeline mỗi video:
#   - faster-whisper decode trực tiếp audio track của video (Việt/Anh, tự nhận diện ngôn ngữ)
#   - Gom các segment (start, end, text) + ghép full_transcript cho BM25/Elasticsearch (Dev 2)
# Kết quả gom vào data/metadata/metadata_asr.json theo format quy định trong task.md.

import argparse
import json
import re
import sys
from pathlib import Path

from config import (
    BEAM_SIZE,
    DEFAULT_CPU_COMPUTE_TYPE,
    DEFAULT_GPU_COMPUTE_TYPE,
    OUTPUT_PATH,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
    WHISPER_MODEL_SIZE,
    print_config,
)

# Fix encoding issue when printing Vietnamese characters in Windows Terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

WHITESPACE_RE = re.compile(r"\s+")


def normalize_video_name(video_path: "Path | str") -> str:
    """Chuẩn hóa tên video khớp với thư mục trong data/keyframes/ (VD: L21_V001)."""
    return Path(video_path).stem.strip()


def build_segments(raw_segments) -> list[dict]:
    """Chuyển segment output của faster-whisper thành list dict {start, end, text}."""
    segments = []
    for seg in raw_segments:
        text = WHITESPACE_RE.sub(" ", seg.text).strip()
        if not text:
            continue
        segments.append({
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": text,
        })
    return segments


def build_full_transcript(segments: list[dict]) -> str:
    """Ghép toàn bộ lời thoại thành 1 chuỗi để nạp vào BM25 / Elasticsearch."""
    return WHITESPACE_RE.sub(" ", " ".join(seg["text"] for seg in segments)).strip()


def build_video_record(video_name: str, segments: list[dict]) -> dict:
    return {
        "video_name": video_name,
        "full_transcript": build_full_transcript(segments),
        "segments": segments,
    }


class ASRExtractor:
    """Bọc faster-whisper WhisperModel — load model 1 lần, transcribe nhiều video."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE, device: str = "auto",
                 compute_type: str | None = None, beam_size: int = BEAM_SIZE):
        from faster_whisper import WhisperModel

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type is None:
            compute_type = DEFAULT_GPU_COMPUTE_TYPE if device == "cuda" else DEFAULT_CPU_COMPUTE_TYPE

        self.beam_size = beam_size
        print(f"[init] Đang tải faster-whisper ({model_size}) trên {device.upper()} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[init] Model đã sẵn sàng.\n")

    def transcribe(self, video_path: Path, language: str | None = None) -> dict:
        raw_segments, info = self.model.transcribe(
            str(video_path),
            language=language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        segments = build_segments(raw_segments)
        print(f"  Ngôn ngữ nhận diện: {info.language} (p={info.language_probability:.2f}) — {len(segments)} segments")
        return build_video_record(normalize_video_name(video_path), segments)


def extract_asr(video_dir: str, output_path: str, model_size: str = WHISPER_MODEL_SIZE,
                 device: str = "auto", compute_type: str | None = None,
                 language: str | None = None, beam_size: int = BEAM_SIZE,
                 limit: int | None = None, overwrite: bool = False,
                 single_video: str | None = None) -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume logic: bỏ qua video đã xử lý (trừ khi --overwrite) ---
    existing_map: dict[str, dict] = {}
    if out_file.exists():
        try:
            existing_records = json.loads(out_file.read_text(encoding="utf-8"))
            for r in existing_records:
                if r.get("video_name"):
                    existing_map[r["video_name"]] = r
            print(f"[resume] Loaded {len(existing_map)} existing records.")
        except Exception as e:
            print(f"[resume] Could not load existing {output_path}: {e}")

    if single_video:
        video_paths = [Path(single_video)]
        print(f"[run] Processing single video: {single_video}")
    else:
        video_dir_path = Path(video_dir)
        if not video_dir_path.exists():
            print(f"❌ Chưa có thư mục video tại: {video_dir}")
            print("👉 Hãy tạo thư mục và thả video gốc vào:")
            print(f"   {video_dir}/")
            print(f"   ├── L21_V001.mp4")
            print(f"   ├── L21_V002.mp4")
            print(f"   └── ...")
            return

        all_paths = sorted(
            p for p in video_dir_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if overwrite:
            video_paths = all_paths
        else:
            video_paths = [p for p in all_paths if normalize_video_name(p) not in existing_map]

        if limit is not None:
            video_paths = video_paths[:limit]
        print(f"[run] Found {len(video_paths)} video(s) to process in {video_dir}")

    if not video_paths:
        print("[run] Không có video mới để xử lý. Exiting.")
        return

    extractor = ASRExtractor(model_size=model_size, device=device, compute_type=compute_type,
                              beam_size=beam_size)

    for i, video_path in enumerate(video_paths, start=1):
        print(f"{'=' * 55}")
        print(f"  [{i}/{len(video_paths)}] VIDEO: {video_path.name}")
        try:
            record = extractor.transcribe(video_path, language=language)
            existing_map[record["video_name"]] = record

            # Lưu checkpoint sau mỗi video để không mất tiến độ khi crash giữa chừng
            out_file.write_text(
                json.dumps(list(existing_map.values()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  💾 Đã lưu: {out_file}")
        except Exception as exc:
            print(f"  ⚠️  Lỗi khi xử lý {video_path.name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 55}")
    print(f"✅ Hoàn tất! Tổng cộng {len(existing_map)} video trong {output_path}")


if __name__ == "__main__":
    print_config()

    parser = argparse.ArgumentParser(
        description="Trích xuất transcript (faster-whisper) từ video ra metadata_asr.json"
    )
    parser.add_argument("--video-dir", default=VIDEO_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--model-size", default=WHISPER_MODEL_SIZE,
                         help="tiny/base/small/medium/large-v3 (nhẹ hơn cho máy CPU-only)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default=None,
                         help="VD: float16, int8, int8_float16 (mặc định tự chọn theo device)")
    parser.add_argument("--language", default=None,
                         help="Ép ngôn ngữ ('vi'/'en'); mặc định tự nhận diện mỗi video")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu (test nhanh)")
    parser.add_argument("--overwrite", action="store_true", help="Xử lý lại cả video đã có trong output")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn đến 1 video cụ thể cần test")
    args = parser.parse_args()

    extract_asr(
        video_dir=args.video_dir,
        output_path=args.output,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        limit=args.limit,
        overwrite=args.overwrite,
        single_video=args.video,
    )
