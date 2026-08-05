"""Tests for the pure helper functions + resume/checkpoint logic of the ASR extractor.

Không cần cài faster-whisper / tải model để chạy các test này — model thật
chỉ được load bên trong ASRExtractor.__init__, còn ở đây ta monkeypatch nó
bằng một stub để kiểm tra logic scan/resume/checkpoint độc lập với model.
Bước tách audio (ffmpeg) cũng được stub để test không phụ thuộc ffmpeg.
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
    load_done_videos,
    normalize_video_name,
    record_path,
    write_record,
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


class TestPerVideoFiles:
    def test_record_path_uses_video_name(self):
        assert record_path("L21_V001", "out") == Path("out/L21_V001.json")

    def test_write_record_creates_readable_json(self, tmp_path):
        record = build_video_record("L21_V001", [{"start": 0.0, "end": 1.0, "text": "xin chào"}])
        path = write_record(record, tmp_path)
        assert json.loads(path.read_text(encoding="utf-8")) == record

    def test_write_record_leaves_no_part_file(self, tmp_path):
        write_record(build_video_record("L21_V001", []), tmp_path)
        assert list(tmp_path.glob("*.part")) == []

    def test_write_record_preserves_vietnamese_diacritics(self, tmp_path):
        record = build_video_record("L21_V001", [{"start": 0.0, "end": 1.0,
                                                   "text": "sụt lún sông Cửu Long"}])
        path = write_record(record, tmp_path)
        assert "sụt lún sông Cửu Long" in path.read_text(encoding="utf-8")

    def test_load_done_videos_lists_stems(self, tmp_path):
        for name in ["L21_V001", "L21_V002"]:
            write_record(build_video_record(name, []), tmp_path)
        assert load_done_videos(tmp_path) == {"L21_V001", "L21_V002"}

    def test_load_done_videos_on_missing_dir(self, tmp_path):
        assert load_done_videos(tmp_path / "nope") == set()


class _StubExtractor:
    """Thay cho ASRExtractor thật — trả record giả dựa trên tên file, không load model."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, video_name):
        segments = [{"start": 0.0, "end": 1.0, "text": f"transcript of {video_name}"}]
        return build_video_record(video_name, segments)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub cả model lẫn ffmpeg để test chạy được ở môi trường không có GPU/ffmpeg."""
    monkeypatch.setattr(extractor_module, "ASRExtractor", _StubExtractor)

    extracted = []

    def fake_extract_audio(video_path, output_path, overwrite=False):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-fake-wav")
        extracted.append(Path(video_path).name)
        return output_path

    monkeypatch.setattr(extractor_module, "extract_audio", fake_extract_audio)
    return extracted


def read_record(out_dir: Path, video_name: str) -> dict:
    return json.loads((out_dir / f"{video_name}.json").read_text(encoding="utf-8"))


class TestExtractAsrResumeAndCheckpoint:
    def test_skips_videos_already_in_output(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")
        (video_dir / "L21_V002.mp4").write_bytes(b"")

        out_dir = tmp_path / "metadata_asr"
        write_record({"video_name": "L21_V001", "full_transcript": "old", "segments": []}, out_dir)

        extract_asr(video_dir=str(video_dir), output_dir=str(out_dir),
                    cache_dir=str(tmp_path / "cache"))

        assert read_record(out_dir, "L21_V001")["full_transcript"] == "old"  # skipped
        assert read_record(out_dir, "L21_V002")["full_transcript"] == "transcript of L21_V002"
        # Video đã xử lý rồi thì không tốn công tách audio lại
        assert stub_pipeline == ["L21_V002.mp4"]

    def test_overwrite_reprocesses_existing_videos(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")

        out_dir = tmp_path / "metadata_asr"
        write_record({"video_name": "L21_V001", "full_transcript": "old", "segments": []}, out_dir)

        extract_asr(video_dir=str(video_dir), output_dir=str(out_dir),
                    cache_dir=str(tmp_path / "cache"), overwrite=True)

        assert read_record(out_dir, "L21_V001")["full_transcript"] == "transcript of L21_V001"

    def test_writes_one_file_per_video(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        for name in ["L21_V003.mp4", "L21_V001.mp4", "L21_V002.mp4"]:
            (video_dir / name).write_bytes(b"")

        out_dir = tmp_path / "metadata_asr"
        extract_asr(video_dir=str(video_dir), output_dir=str(out_dir),
                    cache_dir=str(tmp_path / "cache"))

        assert {p.stem for p in out_dir.glob("*.json")} == {"L21_V001", "L21_V002", "L21_V003"}

    def test_limit_only_processes_first_n(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        for name in ["L21_V001.mp4", "L21_V002.mp4", "L21_V003.mp4"]:
            (video_dir / name).write_bytes(b"")

        out_dir = tmp_path / "metadata_asr"
        extract_asr(video_dir=str(video_dir), output_dir=str(out_dir),
                    cache_dir=str(tmp_path / "cache"), limit=2)

        assert len(list(out_dir.glob("*.json"))) == 2

    def test_cleans_up_audio_cache_by_default(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")
        cache_dir = tmp_path / "cache"

        extract_asr(video_dir=str(video_dir), output_dir=str(tmp_path / "out"),
                    cache_dir=str(cache_dir))

        assert not (cache_dir / "L21_V001.wav").exists()

    def test_keep_audio_retains_wav(self, tmp_path, stub_pipeline):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")
        cache_dir = tmp_path / "cache"

        extract_asr(video_dir=str(video_dir), output_dir=str(tmp_path / "out"),
                    cache_dir=str(cache_dir), keep_audio=True)

        assert (cache_dir / "L21_V001.wav").exists()

    def test_audio_failure_does_not_abort_remaining_videos(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "L21_V001.mp4").write_bytes(b"")
        (video_dir / "L21_V002.mp4").write_bytes(b"")

        monkeypatch.setattr(extractor_module, "ASRExtractor", _StubExtractor)

        def flaky_extract(video_path, output_path, overwrite=False):
            if Path(video_path).stem == "L21_V001":
                raise RuntimeError("ffmpeg boom")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF-fake-wav")
            return output_path

        monkeypatch.setattr(extractor_module, "extract_audio", flaky_extract)

        out_dir = tmp_path / "metadata_asr"
        extract_asr(video_dir=str(video_dir), output_dir=str(out_dir),
                    cache_dir=str(tmp_path / "cache"))

        # Video lỗi không được tạo file -> lần chạy sau tự động thử lại
        assert {p.stem for p in out_dir.glob("*.json")} == {"L21_V002"}

    def test_missing_video_dir_does_not_raise(self, tmp_path):
        out_dir = tmp_path / "metadata_asr"
        extract_asr(video_dir=str(tmp_path / "does-not-exist"), output_dir=str(out_dir))
        assert list(out_dir.glob("*.json")) == []
