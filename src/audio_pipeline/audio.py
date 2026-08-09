"""
audio.py — Tách audio khỏi video bằng ffmpeg trước khi transcribe
=================================================================
Vì sao phải tách riêng thay vì ném thẳng file .mp4 vào faster-whisper?

1. **Nhanh hơn**: file video của BTC nặng ~130MB/video (1080p). Đưa thẳng vào
   faster-whisper thì nó phải demux cả container qua PyAV trong tiến trình Python.
   Gọi ffmpeg với cờ `-vn` bỏ hẳn luồng video, chỉ đọc ~1% dữ liệu cần thiết.
2. **Song song hoá được**: ffmpeg chạy ở tiến trình riêng nên tách audio cho
   video kế tiếp có thể chạy nền trong lúc GPU đang transcribe video hiện tại
   (xem extractor.py) — GPU không phải ngồi chờ đọc đĩa.
3. **Đúng định dạng model cần**: Whisper/PhoWhisper đều yêu cầu PCM mono 16kHz.
   Ép sẵn ở bước này để việc resample không lặp lại trong vòng lặp inference.
"""

import shutil
import subprocess
from pathlib import Path

# Whisper/PhoWhisper đều được huấn luyện ở 16kHz mono — không đổi được.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg chưa được cài hoặc chưa nằm trong PATH."""


def ensure_ffmpeg() -> str:
    """Trả về đường dẫn ffmpeg, raise lỗi kèm hướng dẫn cài nếu chưa có."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegNotFoundError(
            "Không tìm thấy ffmpeg trong PATH.\n"
            "  Windows : winget install Gyan.FFmpeg\n"
            "  Ubuntu  : sudo apt install ffmpeg\n"
            "  Conda   : conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def audio_cache_path(video_path: "Path | str", cache_dir: "Path | str") -> Path:
    """Đường dẫn file WAV tương ứng với 1 video trong thư mục cache."""
    return Path(cache_dir) / f"{Path(video_path).stem}.wav"


def extract_audio(video_path: "Path | str", output_path: "Path | str",
                   overwrite: bool = False) -> Path:
    """Tách audio của `video_path` ra WAV PCM 16-bit mono 16kHz tại `output_path`.

    Trả về đường dẫn WAV. Nếu file đã tồn tại và `overwrite=False` thì dùng lại
    luôn (hữu ích khi chạy lại pipeline sau lỗi giữa chừng).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return output_path

    ffmpeg = ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ghi ra file tạm rồi mới đổi tên: tránh để lại WAV cụt nếu bị Ctrl+C giữa
    # chừng, lần chạy sau lại tưởng là file hợp lệ và transcribe thiếu nội dung.
    tmp_path = output_path.with_suffix(".wav.part")

    cmd = [
        ffmpeg,
        "-nostdin",                       # không nuốt stdin khi chạy trong thread
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vn",                            # BỎ luồng video — mấu chốt của tốc độ
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        # Ép định dạng wav: file tạm có đuôi .part nên ffmpeg không tự suy ra được.
        "-f", "wav",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg thất bại trên {video_path.name} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )

    tmp_path.replace(output_path)
    return output_path


def cleanup_audio(audio_path: "Path | str") -> None:
    """Xoá file WAV tạm. Bỏ qua lỗi vì đây chỉ là dọn rác, không được phép làm
    hỏng cả pipeline khi file đang bị khoá bởi tiến trình khác (hay gặp trên Windows)."""
    try:
        Path(audio_path).unlink(missing_ok=True)
    except OSError:
        pass
