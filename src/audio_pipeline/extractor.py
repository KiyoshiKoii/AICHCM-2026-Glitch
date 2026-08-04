# Task (Phase 2, Dev 4): Trích xuất Script/Phụ đề (ASR) từ Video
#
# Luồng xử lý (2 tầng chạy song song để GPU không bị nghỉ):
#   [luồng nền] ffmpeg tách audio video kế tiếp  ─┐
#                                                 ├─> hàng đợi ─> [luồng chính] GPU transcribe
#   [luồng nền] ffmpeg tách audio video kế tiếp  ─┘
#
#   - Tách audio: ffmpeg -vn -> WAV mono 16kHz (xem audio.py)
#   - Transcribe: PhoWhisper (CTranslate2) qua BatchedInferencePipeline,
#     VAD cắt bỏ khoảng lặng rồi nhồi nhiều chunk vào GPU cùng lúc.
#   - Gom segment (start, end, text) + ghép full_transcript cho BM25/Elasticsearch (Dev 2)
#
# Kết quả gom vào data/metadata/metadata_asr.json theo format quy định trong task.md.

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from audio import audio_cache_path, cleanup_audio, extract_audio
from config import (
    AUDIO_CACHE_DIR,
    AUDIO_WORKERS,
    BATCH_SIZE,
    BEAM_SIZE,
    CHUNK_LENGTH,
    DEFAULT_CPU_COMPUTE_TYPE,
    DEFAULT_GPU_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    OUTPUT_PATH,
    PREFETCH_QUEUE_SIZE,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
    print_config,
    resolve_model,
)

# Fix encoding issue when printing Vietnamese characters in Windows Terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

WHITESPACE_RE = re.compile(r"\s+")

# Đặt ở cuối hàng đợi để luồng chính biết đã hết video.
_QUEUE_DONE = object()


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
    """Bọc faster-whisper — load model 1 lần, transcribe nhiều audio.

    Dùng BatchedInferencePipeline: VAD cắt audio thành các chunk có tiếng nói,
    rồi nhồi `batch_size` chunk vào GPU mỗi lượt forward thay vì chạy tuần tự
    từng chunk. Đây là nguồn tăng tốc chính trên GPU (thường nhanh gấp 3-4 lần).
    """

    def __init__(self, model: str | None = None, device: str = "auto",
                 compute_type: str | None = None, beam_size: int = BEAM_SIZE,
                 batch_size: int = BATCH_SIZE, language: str | None = DEFAULT_LANGUAGE,
                 chunk_length: int = CHUNK_LENGTH):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"

        if compute_type is None:
            compute_type = DEFAULT_GPU_COMPUTE_TYPE if device == "cuda" else DEFAULT_CPU_COMPUTE_TYPE

        self.beam_size = beam_size
        self.batch_size = batch_size
        self.language = language
        self.chunk_length = chunk_length

        model_id = resolve_model(model)
        print(f"[init] Đang tải model ({model_id}) trên {device.upper()} ({compute_type})...")
        whisper_model = WhisperModel(model_id, device=device, compute_type=compute_type)
        self.pipeline = BatchedInferencePipeline(model=whisper_model)
        print(f"[init] Model đã sẵn sàng (batch_size={batch_size}, "
              f"chunk_length={chunk_length}s, language={language or 'auto-detect'}).\n")

    def transcribe(self, audio_path: Path, video_name: str) -> dict:
        raw_segments, info = self.pipeline.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            vad_filter=True,
            # Mốc thời gian của segment lấy theo ranh giới chunk VAD (xem CHUNK_LENGTH
            # trong config.py) — PhoWhisper không sinh được timestamp token.
            chunk_length=self.chunk_length,
        )
        segments = build_segments(raw_segments)
        print(f"  Ngôn ngữ: {info.language} (p={info.language_probability:.2f}) "
              f"— {len(segments)} segments")
        return build_video_record(video_name, segments)


def _cuda_available() -> bool:
    """Kiểm tra GPU mà không cần import torch (env này không bắt buộc có torch)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _audio_producer(video_paths: list[Path], cache_dir: Path, queue: Queue,
                    workers: int, stop_event: threading.Event) -> None:
    """Tách audio ở luồng nền và đẩy vào hàng đợi cho luồng chính transcribe.

    Hàng đợi có giới hạn (PREFETCH_QUEUE_SIZE) nên producer tự động dừng lại khi
    GPU xử lý chậm hơn — tránh sinh hàng chục GB WAV cùng lúc.
    """
    def work(video_path: Path):
        if stop_event.is_set():
            return
        try:
            wav_path = extract_audio(video_path, audio_cache_path(video_path, cache_dir))
            queue.put((video_path, wav_path, None))
        except Exception as exc:
            queue.put((video_path, None, exc))

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ffmpeg") as pool:
            # map() giữ nguyên thứ tự video và tự chặn lại khi queue đầy.
            list(pool.map(work, video_paths))
    finally:
        queue.put(_QUEUE_DONE)


def extract_asr(video_dir: str = VIDEO_DIR, output_path: str = OUTPUT_PATH,
                model: str | None = None, device: str = "auto",
                compute_type: str | None = None, language: str | None = DEFAULT_LANGUAGE,
                beam_size: int = BEAM_SIZE, batch_size: int = BATCH_SIZE,
                chunk_length: int = CHUNK_LENGTH,
                audio_workers: int = AUDIO_WORKERS, cache_dir: str = AUDIO_CACHE_DIR,
                limit: int | None = None, overwrite: bool = False,
                single_video: str | None = None, keep_audio: bool = False) -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir)

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

    extractor = ASRExtractor(model=model, device=device, compute_type=compute_type,
                              beam_size=beam_size, batch_size=batch_size, language=language,
                              chunk_length=chunk_length)

    # Tách audio chạy nền, GPU transcribe ở luồng chính -> 2 việc chồng lên nhau.
    queue: Queue = Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()
    producer = threading.Thread(
        target=_audio_producer,
        args=(video_paths, cache_path, queue, audio_workers, stop_event),
        daemon=True,
    )
    producer.start()

    started = time.perf_counter()
    processed = 0
    failed = 0

    try:
        while True:
            item = queue.get()
            if item is _QUEUE_DONE:
                break

            video_path, wav_path, error = item
            processed += 1
            print(f"{'=' * 55}")
            print(f"  [{processed}/{len(video_paths)}] VIDEO: {video_path.name}")

            if error is not None:
                failed += 1
                print(f"  ⚠️  Lỗi khi tách audio: {error}")
                continue

            try:
                record = extractor.transcribe(wav_path, normalize_video_name(video_path))
                existing_map[record["video_name"]] = record

                # Lưu checkpoint sau mỗi video để không mất tiến độ khi crash giữa chừng
                out_file.write_text(
                    json.dumps(list(existing_map.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  💾 Đã lưu: {out_file}")
            except Exception as exc:
                failed += 1
                print(f"  ⚠️  Lỗi khi transcribe {video_path.name}: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                if not keep_audio:
                    cleanup_audio(wav_path)
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng theo yêu cầu. Tiến độ đã được lưu, chạy lại để tiếp tục.")
    finally:
        stop_event.set()
        producer.join(timeout=5)

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 55}")
    print(f"✅ Hoàn tất! Tổng cộng {len(existing_map)} video trong {output_path}")
    print(f"   Lượt chạy này: {processed - failed} thành công, {failed} lỗi, "
          f"{elapsed / 60:.1f} phút.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất transcript (PhoWhisper/faster-whisper) từ video ra metadata_asr.json"
    )
    parser.add_argument("--video-dir", default=VIDEO_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--model", default=None,
                         help="Thư mục model CT2, repo HuggingFace, hoặc size Whisper "
                              "('large-v3'). Mặc định: PhoWhisper đã convert.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default=None,
                         help="VD: float16, int8, int8_float16 (mặc định tự chọn theo device)")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                         help="Ép ngôn ngữ ('vi'/'en'). Để trống ('') để tự nhận diện.")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help="Số chunk audio nhồi vào GPU cùng lúc. Giảm nếu hết VRAM.")
    parser.add_argument("--chunk-length", type=int, default=CHUNK_LENGTH,
                         help="Độ dài chunk VAD (giây) — quyết định độ mịn timestamp "
                              "của segment. Nhỏ hơn = mịn hơn nhưng chậm hơn.")
    parser.add_argument("--audio-workers", type=int, default=AUDIO_WORKERS,
                         help="Số luồng ffmpeg tách audio chạy nền")
    parser.add_argument("--cache-dir", default=AUDIO_CACHE_DIR)
    parser.add_argument("--keep-audio", action="store_true",
                         help="Giữ lại file WAV sau khi transcribe (để debug)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu (test nhanh)")
    parser.add_argument("--overwrite", action="store_true", help="Xử lý lại cả video đã có trong output")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn đến 1 video cụ thể cần test")
    args = parser.parse_args()

    print_config()

    extract_asr(
        video_dir=args.video_dir,
        output_path=args.output,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language or None,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        audio_workers=args.audio_workers,
        cache_dir=args.cache_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        single_video=args.video,
        keep_audio=args.keep_audio,
    )
