import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import semantic_pipeline.gemini.repair as repair
from semantic_pipeline.core.compact_metadata import CompactVisualRecord


def _write_existing_caption(output_dir: Path) -> Path:
    path = output_dir / "L21" / "L21_V001.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                CompactVisualRecord(
                    frame_id="L21_V001_f0001",
                    caption="Old caption.",
                    detailed_caption="Old detailed caption.",
                ).model_dump(mode="json"),
                CompactVisualRecord(
                    frame_id="L21_V001_f0002",
                    caption="Unchanged caption.",
                    detailed_caption="Unchanged detailed caption.",
                ).model_dump(mode="json"),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": "L21_V001_f0001",
                        "flag": True,
                        "claimed_score": 0.14,
                        "best_score": 0.29,
                    },
                    {
                        "frame_id": "L21_V001_f0002",
                        "flag": False,
                        "claimed_score": 0.31,
                        "best_score": 0.32,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_repair_overwrites_only_flagged_record_and_bypasses_dedup(tmp_path, monkeypatch):
    keyframes = tmp_path / "keyframes" / "L21_V001"
    keyframes.mkdir(parents=True)
    for name in ("001.jpg", "002.jpg"):
        Image.new("RGB", (100, 50), "white").save(keyframes / name)
    output_dir = tmp_path / "caption"
    output_path = _write_existing_caption(output_dir)
    report_path = tmp_path / "clip_report.json"
    _write_report(report_path)
    requested = []

    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def batch_paths(paths, **_kwargs):
            return [list(paths)]

        def extract_batch(self, paths, **_kwargs):
            requested.extend(path.name for path in paths)
            return [
                CompactVisualRecord(
                    frame_id="L21_V001_f0001",
                    caption="Re-extracted caption.",
                    detailed_caption="Re-extracted detailed caption.",
                )
            ]

    monkeypatch.setattr(repair, "GeminiVisualExtractor", FakeExtractor)
    summary = repair.run_repair(
        report_paths=[report_path],
        keyframe_dir=keyframes.parent,
        output_dir=output_dir,
        api_key="test-key",
        request_budget_state_path=tmp_path / "budget.json",
    )

    assert requested == ["001.jpg"]
    assert summary["repaired"] == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["caption"] for item in saved] == [
        "Re-extracted caption.",
        "Unchanged caption.",
    ]


def test_severity_filters_keep_only_high_confidence_repair_candidates(tmp_path):
    report_path = tmp_path / "clip_report.json"
    _write_report(report_path)

    candidates = repair.load_repair_candidates(
        [report_path], max_claimed_score=0.20, min_alternate_gap=0.10
    )

    assert [candidate.frame_id for candidate in candidates] == ["L21_V001_f0001"]


def test_copied_frame_filter_intersects_the_clip_flags(tmp_path):
    report_path = tmp_path / "clip_report.json"
    _write_report(report_path)

    candidates = repair.load_repair_candidates(
        [report_path], copied_frame_ids={"L21_V001_f0001"}
    )

    assert [candidate.frame_id for candidate in candidates] == ["L21_V001_f0001"]
