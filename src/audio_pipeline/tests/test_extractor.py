"""Tests for the pure helper functions + resume/checkpoint logic of the ASR extractor.

Không cần cài faster-whisper / tải model để chạy các test này — model thật
chỉ được load bên trong ASRExtractor.__init__, còn ở đây ta monkeypatch nó
bằng một stub để kiểm tra logic scan/resume/checkpoint độc lập với model.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import extractor as extractor_module
from extractor import (
    build_full_transcript,
    build_segments,
    build_video_record,
    extract_asr,
    normalize_video_name,
)


def fake_segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


class TestNormalizeVideoName:
    def test_strips_extension(self):
        assert normalize_video_name("L21_V001.mp4") == "L21_V001"

    def test_works_with_path_object(self):
        assert normalize_video_name(Path("/data/videos/L22_V010.webm")) == "L22_V010"


class TestBuildSegments:
    def test_rounds_timestamps_and_collapses_whitespace(self):
        raw = [fake_segment(0.001, 4.499, "  Chào mừng   các bạn  ")]
        segments = build_segments(raw)
        assert segments == [{"start": 0.0, "end": 4.5, "text": "Chào mừng các bạn"}]

    def test_drops_empty_segments(self):
        raw = [fake_segment(0.0, 1.0, "   "), fake_segment(1.0, 2.0, "hello")]
        segments = build_segments(raw)
        assert len(segments) == 1
        assert segments[0]["text"] == "hello"


class TestBuildFullTranscript:
    def test_joins_segment_text_in_order(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Xin chào"},
                    {"start": 1.0, "end": 2.0, "text": "các bạn"}]
        assert build_full_transcript(segments) == "Xin chào các bạn"

    def test_empty_segments_gives_empty_string(self):
        assert build_full_transcript([]) == ""


class TestBuildVideoRecord:
    def test_matches_task_output_schema(self):
        segments = [{"start": 0.0, "end": 4.5, "text": "Chào mừng các bạn"}]
        record = build_video_record("L21_V001", segments)
        assert record == {
            "video_name": "L21_V001",
            "full_transcript": "Chào mừng các bạn",
            "segments": segments,
        }


class _StubExtractor:
    """Thay cho ASRExtractor thật — trả record giả dựa trên tên file, không load model."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, video_path, language=None):
        name = normalize_video_name(video_path)
        segments = [{"start": 0.0, "end": 1.0, "text": f"transcript of {name}"}]
        return build_video_record(name, segments)


class TestExtractAsrResumeAndCheckpoint:
    def test_skips_videos_already_in_output(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")
        (video_dir / "L21_V002.mp4").write_bytes(b"")

        output_path = tmp_path / "metadata_asr.json"
        output_path.write_text(
            json.dumps([{"video_name": "L21_V001", "full_transcript": "old", "segments": []}]),
            encoding="utf-8",
        )

        monkeypatch.setattr(extractor_module, "ASRExtractor", _StubExtractor)

        extract_asr(video_dir=str(video_dir), output_path=str(output_path))

        records = {r["video_name"]: r for r in json.loads(output_path.read_text(encoding="utf-8"))}
        assert records["L21_V001"]["full_transcript"] == "old"  # unchanged, was skipped
        assert records["L21_V002"]["full_transcript"] == "transcript of L21_V002"

    def test_overwrite_reprocesses_existing_videos(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")

        output_path = tmp_path / "metadata_asr.json"
        output_path.write_text(
            json.dumps([{"video_name": "L21_V001", "full_transcript": "old", "segments": []}]),
            encoding="utf-8",
        )

        monkeypatch.setattr(extractor_module, "ASRExtractor", _StubExtractor)

        extract_asr(video_dir=str(video_dir), output_path=str(output_path), overwrite=True)

        records = json.loads(output_path.read_text(encoding="utf-8"))
        assert records[0]["full_transcript"] == "transcript of L21_V001"

    def test_missing_video_dir_does_not_raise(self, tmp_path, capsys):
        extract_asr(video_dir=str(tmp_path / "does-not-exist"),
                     output_path=str(tmp_path / "metadata_asr.json"))
        assert not (tmp_path / "metadata_asr.json").exists()
