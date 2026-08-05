"""Tests cho merge_asr.py — gộp/tách giữa file lẻ và file bàn giao Dev 2."""

import json
from pathlib import Path

import pytest

from merge_asr import load_records, merge, split


def make_record(name: str, text: str = "xin chào") -> dict:
    return {
        "video_name": name,
        "full_transcript": text,
        "segments": [{"start": 0.0, "end": 1.0, "text": text}],
    }


def write(dir_path: Path, record: dict) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{record['video_name']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadRecords:
    def test_sorts_by_video_name(self, tmp_path):
        for name in ["L21_V003", "L21_V001", "L21_V002"]:
            write(tmp_path, make_record(name))
        assert [r["video_name"] for r in load_records(tmp_path)] == [
            "L21_V001", "L21_V002", "L21_V003"]

    def test_skips_corrupt_json(self, tmp_path):
        write(tmp_path, make_record("L21_V001"))
        (tmp_path / "L21_V002.json").write_text("{not json", encoding="utf-8")
        records = load_records(tmp_path)
        assert [r["video_name"] for r in records] == ["L21_V001"]

    def test_skips_records_missing_required_keys(self, tmp_path):
        write(tmp_path, make_record("L21_V001"))
        (tmp_path / "L21_V002.json").write_text(json.dumps({"video_name": "L21_V002"}),
                                                 encoding="utf-8")
        assert [r["video_name"] for r in load_records(tmp_path)] == ["L21_V001"]

    def test_ignores_leftover_part_files(self, tmp_path):
        write(tmp_path, make_record("L21_V001"))
        (tmp_path / "L21_V002.json.part").write_text("{partial", encoding="utf-8")
        assert [r["video_name"] for r in load_records(tmp_path)] == ["L21_V001"]


class TestMerge:
    def test_produces_task_spec_json_array(self, tmp_path):
        for name in ["L21_V001", "L21_V002"]:
            write(tmp_path / "in", make_record(name))
        out = tmp_path / "metadata_asr.json"

        merge(tmp_path / "in", out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 2
        assert set(data[0]) == {"video_name", "full_transcript", "segments"}

    def test_preserves_vietnamese_diacritics(self, tmp_path):
        write(tmp_path / "in", make_record("L21_V001", "sụt lún sông Cửu Long"))
        out = tmp_path / "metadata_asr.json"
        merge(tmp_path / "in", out)
        assert "sụt lún sông Cửu Long" in out.read_text(encoding="utf-8")

    def test_check_mode_does_not_write(self, tmp_path):
        write(tmp_path / "in", make_record("L21_V001"))
        out = tmp_path / "metadata_asr.json"
        merge(tmp_path / "in", out, check_only=True)
        assert not out.exists()

    def test_exits_when_input_dir_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            merge(tmp_path / "nope", tmp_path / "out.json")

    def test_exits_when_no_valid_records(self, tmp_path):
        (tmp_path / "in").mkdir()
        with pytest.raises(SystemExit):
            merge(tmp_path / "in", tmp_path / "out.json")


class TestSplit:
    def test_round_trip_merge_split_is_lossless(self, tmp_path):
        original = [make_record("L21_V001", "một hai ba"), make_record("L21_V002", "bốn năm")]
        merged = tmp_path / "metadata_asr.json"
        merged.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

        split(merged, tmp_path / "out")
        assert load_records(tmp_path / "out") == original

    def test_exits_when_input_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            split(tmp_path / "nope.json", tmp_path / "out")

    def test_exits_when_input_not_a_list(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"video_name": "L21_V001"}), encoding="utf-8")
        with pytest.raises(SystemExit):
            split(bad, tmp_path / "out")
