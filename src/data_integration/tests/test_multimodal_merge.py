from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from semantic_pipeline.core.compact_metadata import CompactVisualRecord
from data_integration.multimodal_merge import (
    ASRSegment,
    KeyframeMapRow,
    load_asr_segments,
    merge_video_files,
    merge_video_records,
    write_merged_artifact,
)
from semantic_pipeline.retrieval.elasticsearch_backend import build_frame_document


def _caption(frame_number: int) -> dict:
    return {
        "frame_id": f"L21_V001_f{frame_number:04d}",
        "visual_source_frame_id": f"L21_V001_f{frame_number:04d}",
        "ocr_source_frame_id": f"L21_V001_f{frame_number:04d}",
        "quality_flags": [],
        "caption": f"Caption for keyframe {frame_number}.",
        "detailed_caption": "A presenter is speaking in a studio.",
        "caption_vi": "Người dẫn chương trình đang nói trong trường quay.",
        "detailed_caption_vi": "Một người dẫn chương trình đang nói trong trường quay truyền hình.",
        "ocr_text": "THOI SU",
        "news_ticker_text": "",
        "detections": [
            {
                "object_id": "person_0",
                "label": "person",
                "bbox": [0.1, 0.1, 0.5, 0.9],
                "description": "a news presenter",
                "description_vi": "một người dẫn chương trình",
                "attributes": ["standing"],
                "action": "speaking",
            }
        ],
        "spatial_relations": [],
    }


def _map() -> list[KeyframeMapRow]:
    return [
        KeyframeMapRow(n=1, pts_time=0.0, fps=30.0, frame_idx=0),
        KeyframeMapRow(n=2, pts_time=3.0, fps=30.0, frame_idx=90),
        KeyframeMapRow(n=3, pts_time=8.7, fps=30.0, frame_idx=261),
    ]


def test_merges_asr_once_and_keeps_btc_submission_index() -> None:
    records, report = merge_video_records(
        [_caption(1), _caption(2), _caption(3)],
        video_id="L21_V001",
        keyframe_map=_map(),
        asr_available=True,
        asr_segments=[
            ASRSegment(index=0, start=0.1, end=1.0, text="Bản tin buổi sáng."),
            ASRSegment(index=1, start=2.6, end=3.4, text="Thời tiết hôm nay."),
            ASRSegment(index=2, start=7.8, end=8.2, text="UNK tên riêng."),
        ],
    )

    assert [item["frame_id"] for item in records] == [
        "L21_V001_f0001",
        "L21_V001_f0002",
        "L21_V001_f0003",
    ]
    assert [item["native_frame_index"] for item in records] == [0, 90, 261]
    assert [item["asr_segment_indices"] for item in records] == [[0], [1], [2]]
    assert records[2]["asr_quality_flags"] == ["contains_unk"]
    assert report.assigned_asr_segments == 3
    assert report.frames_with_asr == 3

    document = build_frame_document(CompactVisualRecord.model_validate(records[1]))
    assert document["frame_id"] == "L21_V001_f0002"
    assert document["frame_number"] == 90
    assert document["keyframe_number"] == 2
    assert document["asr_text"] == "Thời tiết hôm nay."


def test_marks_visual_frames_when_asr_artifact_is_missing() -> None:
    records, report = merge_video_records(
        [_caption(1), _caption(2), _caption(3)],
        video_id="L21_V001",
        keyframe_map=_map(),
        asr_available=False,
    )

    assert report.asr_available is False
    assert report.asr_segments == 0
    assert all(not item["asr_available"] and not item["has_asr"] for item in records)
    assert all(item["asr_text"] == "" for item in records)


def test_rejects_caption_keyframe_that_is_not_in_btc_map() -> None:
    with pytest.raises(ValueError, match="no row"):
        merge_video_records(
            [_caption(4)],
            video_id="L21_V001",
            keyframe_map=_map(),
            asr_available=False,
        )


def test_reads_realistic_artifacts_and_writes_atomically(tmp_path: Path) -> None:
    caption_path = tmp_path / "caption" / "L21" / "L21_V001.json"
    caption_path.parent.mkdir(parents=True)
    caption_path.write_text(json.dumps([_caption(1), _caption(2), _caption(3)]), encoding="utf-8")

    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    with (map_dir / "L21_V001.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(
            [
                {"n": 1, "pts_time": 0.0, "fps": 30.0, "frame_idx": 0},
                {"n": 2, "pts_time": 3.0, "fps": 30.0, "frame_idx": 90},
                {"n": 3, "pts_time": 8.7, "fps": 30.0, "frame_idx": 261},
            ]
        )

    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    (asr_dir / "L21_V001.json").write_text(
        json.dumps(
            {
                "video_name": "L21_V001",
                "full_transcript": "Bản tin buổi sáng.",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Bản tin buổi sáng."}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records, report = merge_video_files(caption_path, asr_dir=asr_dir, map_dir=map_dir)
    output_path = tmp_path / "processed" / "L21" / "L21_V001.json"
    write_merged_artifact(records, output_path=output_path)

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.caption_frames == len(persisted) == 3
    assert persisted[0]["asr_segment_indices"] == [0]
    with pytest.raises(FileExistsError):
        write_merged_artifact(records, output_path=output_path)


def test_rejects_empty_asr_text(tmp_path: Path) -> None:
    asr_path = tmp_path / "L21_V001.json"
    asr_path.write_text(
        json.dumps(
            {
                "video_name": "L21_V001",
                "full_transcript": "",
                "segments": [{"start": 0.0, "end": 1.0, "text": "   "}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank text"):
        load_asr_segments(asr_path, expected_video_id="L21_V001")
