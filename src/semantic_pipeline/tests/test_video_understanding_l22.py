from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_pipeline.video_understanding.cli import discover_batch_video_ids
from semantic_pipeline.video_understanding.pipeline import build_video


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_l22_v001_offline_pipeline_publishes_three_fixed_outputs(tmp_path: Path) -> None:
    source_paths = [
        ROOT / "data/metadata/caption/L22/L22_V001.json",
        ROOT / "data/metadata/metadata_asr/L22_V001.json",
        ROOT / "data/map-keyframes/L22_V001.csv",
    ]
    before = {str(path): _sha256(path) for path in source_paths}

    result = build_video(
        video_id="L22_V001",
        caption_dir=ROOT / "data/metadata/caption",
        asr_dir=ROOT / "data/metadata/metadata_asr",
        map_dir=ROOT / "data/map-keyframes",
        keyframe_dir=ROOT / "data/keyframes",
        output_root=tmp_path / "video_understanding",
        use_llm=False,
    )

    pilot = tmp_path / "video_understanding/L22/L22_V001/pilot"
    assert result["published"] is True
    assert result["frames"] == 298
    assert result["asr_segments"] == 290
    assert result["scenes"] > 0
    assert result["stories"] > 0
    assert {path.name for path in pilot.iterdir()} == {
        "timeline.json",
        "video_summary.json",
        "validation_report.json",
    }
    timeline = json.loads((pilot / "timeline.json").read_text(encoding="utf-8"))
    summary = json.loads((pilot / "video_summary.json").read_text(encoding="utf-8"))
    report = json.loads((pilot / "validation_report.json").read_text(encoding="utf-8"))
    assert timeline["video_id"] == summary["video_id"] == report["video_id"] == "L22_V001"
    assert timeline["generation_id"] == summary["generation_id"] == report["generation_id"]
    assert report["status"] == "ready"
    assert report["validation"]["all_frame_refs_valid"] is True
    assert report["validation"]["all_asr_refs_valid"] is True
    assert report["validation"]["all_primary_asr_segments_assigned"] is True
    assert report["retrieval_coverage"]["segments_in_search_text"] == report["retrieval_coverage"]["timeline_segments"]
    assert not report["retrieval_coverage"]["missing_segment_titles"]
    assert not report["retrieval_coverage"]["missing_segment_summaries"]
    assert all(_sha256(path) == before[str(path)] for path in source_paths)


def test_l22_v001_replaces_only_the_fixed_pilot_and_keeps_three_files(tmp_path: Path) -> None:
    kwargs = {
        "video_id": "L22_V001",
        "caption_dir": ROOT / "data/metadata/caption",
        "asr_dir": ROOT / "data/metadata/metadata_asr",
        "map_dir": ROOT / "data/map-keyframes",
        "keyframe_dir": ROOT / "data/keyframes",
        "output_root": tmp_path / "video_understanding",
        "use_llm": False,
    }
    first = build_video(**kwargs)
    first_report_path = tmp_path / "video_understanding/L22/L22_V001/pilot/validation_report.json"
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    second = build_video(**kwargs)
    second_report = json.loads(first_report_path.read_text(encoding="utf-8"))

    assert first["published"] is True
    assert second["published"] is True
    assert second_report["generation_id"] != first_report["generation_id"]
    assert len(list(first_report_path.parent.iterdir())) == 3
    assert not list((tmp_path / "video_understanding/.staging").glob("*"))


def test_pipeline_derives_batch_and_segment_ids_from_video_id(tmp_path: Path) -> None:
    result = build_video(
        video_id="L21_V001",
        caption_dir=ROOT / "data/metadata/caption",
        asr_dir=ROOT / "data/metadata/metadata_asr",
        map_dir=ROOT / "data/map-keyframes",
        keyframe_dir=ROOT / "data/keyframes",
        output_root=tmp_path / "video_understanding",
        use_llm=False,
    )

    pilot = tmp_path / "video_understanding/L21/L21_V001/pilot"
    timeline = json.loads((pilot / "timeline.json").read_text(encoding="utf-8"))
    assert result["pilot_dir"] == str(pilot)
    assert timeline["video_id"] == "L21_V001"
    assert timeline["segments"]
    assert all(
        segment["segment_id"].startswith("L21_V001_story_")
        for segment in timeline["segments"]
    )


@pytest.mark.parametrize("video_id", ["L2_V001", "L22_001", "../L22_V001", "L22_V1"])
def test_pipeline_rejects_invalid_video_id_before_resolving_sources(
    tmp_path: Path,
    video_id: str,
) -> None:
    with pytest.raises(ValueError, match="Lxx_Vyyy"):
        build_video(video_id=video_id, output_root=tmp_path)


def test_batch_discovery_is_dynamic_and_sorted(tmp_path: Path) -> None:
    batch_dir = tmp_path / "caption/L30"
    batch_dir.mkdir(parents=True)
    for name in ("L30_V010.json", "L30_V002.json", "L30_bad.json", "L29_V001.json"):
        (batch_dir / name).write_text("[]", encoding="utf-8")

    assert discover_batch_video_ids(tmp_path / "caption", "L30") == [
        "L30_V002",
        "L30_V010",
    ]


@pytest.mark.parametrize("batch_id", ["L3", "30", "../L30"])
def test_batch_discovery_rejects_invalid_batch_id(
    tmp_path: Path,
    batch_id: str,
) -> None:
    with pytest.raises(ValueError, match="Lxx"):
        discover_batch_video_ids(tmp_path, batch_id)
