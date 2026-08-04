"""Tests cho bước tách audio bằng ffmpeg (audio.py)."""

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

import audio as audio_module
from audio import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    FFmpegNotFoundError,
    audio_cache_path,
    cleanup_audio,
    ensure_ffmpeg,
    extract_audio,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class TestAudioCachePath:
    def test_maps_video_to_wav_in_cache_dir(self):
        path = audio_cache_path("data/videos/L21_V001.mp4", "data/audio_cache")
        assert path == Path("data/audio_cache/L21_V001.wav")


class TestEnsureFfmpeg:
    def test_raises_helpful_error_when_missing(self, monkeypatch):
        monkeypatch.setattr(audio_module.shutil, "which", lambda _: None)
        with pytest.raises(FFmpegNotFoundError, match="winget install"):
            ensure_ffmpeg()


class TestCleanupAudio:
    def test_removes_file(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"x")
        cleanup_audio(f)
        assert not f.exists()

    def test_missing_file_is_not_an_error(self, tmp_path):
        cleanup_audio(tmp_path / "nope.wav")  # must not raise


class TestExtractAudioContract:
    """Kiểm tra lệnh ffmpeg dựng đúng, không cần chạy ffmpeg thật."""

    def test_builds_command_that_drops_video_stream(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
        monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg")

        out = extract_audio(tmp_path / "L21_V001.mp4", tmp_path / "out.wav")

        cmd = captured["cmd"]
        assert "-vn" in cmd, "phải bỏ luồng video để không decode thừa"
        assert cmd[cmd.index("-ar") + 1] == str(TARGET_SAMPLE_RATE)
        assert cmd[cmd.index("-ac") + 1] == str(TARGET_CHANNELS)
        assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
        assert out.exists()

    def test_reuses_existing_wav(self, tmp_path, monkeypatch):
        existing = tmp_path / "out.wav"
        existing.write_bytes(b"already here")

        def boom(*a, **k):
            raise AssertionError("không được gọi ffmpeg khi WAV đã tồn tại")

        monkeypatch.setattr(audio_module.subprocess, "run", boom)
        assert extract_audio(tmp_path / "v.mp4", existing) == existing

    def test_failure_leaves_no_partial_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg")
        monkeypatch.setattr(
            audio_module.subprocess, "run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "bad input"),
        )

        with pytest.raises(RuntimeError, match="ffmpeg thất bại"):
            extract_audio(tmp_path / "v.mp4", tmp_path / "out.wav")

        assert not (tmp_path / "out.wav").exists()
        assert not (tmp_path / "out.wav.part").exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg chưa cài trên máy này")
class TestExtractAudioReal:
    """Chạy ffmpeg thật trên 1 file video tí hon tự sinh."""

    def test_produces_16khz_mono_wav(self, tmp_path):
        video = tmp_path / "L99_V001.mp4"
        # Sinh 1 video 1 giây: hình màu + tone 440Hz, để chắc chắn có luồng audio.
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=128x128:d=1",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-shortest", str(video)],
            check=True, capture_output=True,
        )

        out = extract_audio(video, tmp_path / "out.wav")

        with wave.open(str(out), "rb") as w:
            assert w.getframerate() == TARGET_SAMPLE_RATE
            assert w.getnchannels() == TARGET_CHANNELS
            assert w.getsampwidth() == 2  # pcm_s16le
            assert w.getnframes() > 0
