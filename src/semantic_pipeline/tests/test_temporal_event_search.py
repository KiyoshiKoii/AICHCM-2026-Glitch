from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_pipeline.retrieval.temporal_event_search import (
    TemporalCorpus,
    TemporalEventSearch,
    _matches_concept_groups,
    discover_temporal_corpus,
)
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query
from semantic_pipeline.retrieval.qwen_video_verifier import TemporalVerificationResult


ROOT = Path(__file__).resolve().parents[3]


def test_parser_keeps_duplicate_source_labels_in_order() -> None:
    parsed = parse_temporal_query(
        "Video nấu ăn\nE1: khoảnh khắc đầu tiên cắt nấm\nE2: khoảnh khắc cuối cùng đặt lên đĩa\nE2: sau đó chảo được đặt lên bếp"
    )

    assert parsed.shared_context == "Video nấu ăn"
    assert [event.event_index for event in parsed.events] == [1, 2, 3]
    assert [event.source_label for event in parsed.events] == ["E1", "E2", "E2"]
    assert parsed.events[0].required_anchor == "action_start"
    assert parsed.events[1].required_anchor == "last_complete"


def test_summary_only_query_is_kept_as_video_context() -> None:
    parsed = parse_temporal_query("Bản tin về thời tiết nóng tại Barcelona")

    assert parsed.shared_context == "Bản tin về thời tiết nóng tại Barcelona"
    assert len(parsed.events) == 1


def test_parser_routes_boundary_types_without_treating_fully_as_last() -> None:
    parsed = parse_temporal_query(
        "Factory scene\n"
        "E1: first moment the frame touches the car\n"
        "E2: the handle starts rotating\n"
        "E3: first moment the worker is fully bent down\n"
        "E4: first appearance of the landmark"
    )

    assert [event.event_type for event in parsed.events] == [
        "first_contact",
        "action_start",
        "state_attainment",
        "first_appearance",
    ]
    assert [event.selection_rule for event in parsed.events] == ["earliest_true"] * 4
    assert all(event.transition_required for event in parsed.events)


def test_first_visible_action_is_not_misrouted_as_landmark_appearance() -> None:
    parsed = parse_temporal_query("E1: first visible moment a chef cuts mushrooms")

    assert parsed.events[0].event_type == "action_start"


def _temporal_corpus(video_id: str, location: str, event_text: str, frame_number: int) -> TemporalCorpus:
    frame = {
        "frame_id": f"{video_id}_f{frame_number:04d}",
        "keyframe_n": frame_number,
        "native_frame_idx": frame_number * 100,
        "timestamp_ms": frame_number * 1_000,
        "visual_text": event_text,
        "asr_text": "",
        "ocr_text": "",
    }
    return TemporalCorpus(
        video={
            "video_id": video_id,
            "summary_vi": f"Bản tin về thời tiết nóng tại {location}",
            "summary_en": "",
            "search_text": f"Bản tin thời tiết nóng {location}",
            "main_locations": [location],
            "main_entities": [],
        },
        events=[
            {
                "event_id": f"{video_id}_event_1",
                "description_vi": event_text,
                "search_text": event_text,
                "start_ms": frame["timestamp_ms"],
                "temporal_anchors": [],
                "_frames": [frame],
            }
        ],
        frames=[frame],
    )


def test_video_context_exact_location_outranks_generic_hot_weather() -> None:
    search = TemporalEventSearch(
        [
            _temporal_corpus("L22_V001", "Barcelona", "Nắng nóng kỷ lục tại Barcelona", 91),
            _temporal_corpus("L22_V025", "Tây Ban Nha", "Trượt tuyết tránh nóng tại Tây Ban Nha", 20),
        ]
    )

    result = search.search("Bản tin về thời tiết nóng tại Barcelona", top_k_videos=2)

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["selected_video"]["matched_context_entities"] == ["Barcelona"]
    assert result["events"][0]["frame_id"] == "L22_V001_f0091"
    assert [item["video_id"] for item in result["candidates"]] == ["L22_V001"]
    assert [item["rank"] for item in result["candidates"]] == [1]


def test_all_explicit_concepts_outrank_an_olympic_paris_only_story() -> None:
    search = TemporalEventSearch(
        [
            _temporal_corpus(
                "L22_V001",
                "Pháp",
                "Linh vật Olympic Paris 2024 bán chạy",
                160,
            ),
            _temporal_corpus(
                "L22_V002",
                "Paris",
                "Hoạt động văn hóa tại Olympic Paris",
                140,
            ),
        ]
    )

    result = search.search("linh vật Olympic Paris", top_k_videos=2)

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["events"][0]["frame_id"] == "L22_V001_f0160"
    assert result["candidates"][0]["frame_id"] == "L22_V001_f0160"


def test_concept_match_accepts_paraphrase_but_rejects_scattered_terms() -> None:
    groups = (("phố lồng đèn", "khu phố với đèn lồng"),)

    supported, _ = _matches_concept_groups(
        groups,
        "Khu phố được trang hoàng bằng hàng trăm chiếc đèn lồng dịp Trung thu.",
    )
    scattered, _ = _matches_concept_groups(
        groups,
        "Dự án cải tạo đền thờ được triển khai. Thành phố đầu tư ngân sách "
        "cho nhiều công trình văn hóa ở khu vực phường Long Bình.",
    )

    assert supported == 1.0
    assert scattered == 0.0


def test_first_event_requires_all_explicit_concepts_before_earliest_anchor() -> None:
    def event(event_id: str, text: str, timestamp_ms: int) -> dict[str, object]:
        frame = {
            "frame_id": f"L23_V001_f{timestamp_ms // 1_000:04d}",
            "keyframe_n": timestamp_ms // 1_000,
            "native_frame_idx": timestamp_ms // 40,
            "timestamp_ms": timestamp_ms,
            "visual_text": text,
            "asr_text": "",
            "ocr_text": "",
        }
        return {
            "event_id": event_id,
            "description_vi": text,
            "search_text": text,
            "start_ms": timestamp_ms,
            "temporal_anchors": [
                {
                    "anchor_type": "action_start",
                    "frame_id": frame["frame_id"],
                    "keyframe_n": frame["keyframe_n"],
                    "native_frame_idx": frame["native_frame_idx"],
                    "timestamp_ms": timestamp_ms,
                }
            ],
            "_frames": [frame],
        }

    corpus = TemporalCorpus(
        video={
            "video_id": "L23_V001",
            "summary_vi": "Video đua xe đạp trên phố.",
            "summary_en": "",
            "search_text": "Đua xe đạp trên phố.",
            "main_locations": ["phố"],
            "main_entities": ["vận động viên xe đạp"],
        },
        events=[
            event(
                "L23_V001_event_early_generic",
                "Xe mô tô dẫn đường di chuyển trên phố rộng.",
                1_000,
            ),
            event(
                "L23_V001_event_later_exact",
                "Xe mô tô màu vàng dẫn đường di chuyển trên phố rộng.",
                5_000,
            ),
        ],
        frames=[],
    )

    result = TemporalEventSearch([corpus]).search(
        "Video đua xe đạp trên phố\n"
        "E1: Khoảnh khắc đầu tiên xe mô tô dẫn đường màu vàng di chuyển trên phố rộng."
    )

    assert result["events"][0]["event_id"] == "L23_V001_event_later_exact"
    assert result["events"][0]["timestamp_ms"] == 5_000


def test_first_event_anchor_requires_matching_frame_evidence() -> None:
    def frame(number: int, text: str) -> dict[str, object]:
        return {
            "frame_id": f"L23_V001_f{number:04d}",
            "keyframe_n": number,
            "native_frame_idx": number * 100,
            "timestamp_ms": number * 1_000,
            "visual_text": text,
            "asr_text": "",
            "ocr_text": "",
        }

    unrelated = frame(1, "Linh vật và khán giả đứng ven đường.")
    cyclists = frame(2, "Đoàn vận động viên xe đạp đang di chuyển trên đường nhựa.")
    corpus = TemporalCorpus(
        video={
            "video_id": "L23_V001",
            "summary_vi": "Video đua xe đạp trên phố.",
            "summary_en": "",
            "search_text": "Đua xe đạp trên phố.",
            "main_locations": ["phố"],
            "main_entities": ["vận động viên xe đạp"],
        },
        events=[
            {
                "event_id": "L23_V001_story_title_only",
                "description_vi": "Đoàn vận động viên xe đạp di chuyển qua khán giả.",
                "search_text": "Đoàn vận động viên xe đạp di chuyển qua khán giả.",
                "start_ms": 1_000,
                "temporal_anchors": [],
                "_frames": [unrelated],
            },
            {
                "event_id": "L23_V001_first_cyclists",
                "description_vi": "Đoàn vận động viên xe đạp di chuyển trên đường nhựa.",
                "search_text": "Đoàn vận động viên xe đạp di chuyển trên đường nhựa.",
                "start_ms": 2_000,
                "temporal_anchors": [],
                "_frames": [cyclists],
            },
        ],
        frames=[unrelated, cyclists],
    )

    result = TemporalEventSearch([corpus]).search(
        "Video đua xe đạp trên phố\n"
        "E1: Khoảnh khắc đầu tiên thấy đoàn vận động viên xe đạp."
    )

    assert result["events"][0]["event_id"] == "L23_V001_first_cyclists"
    assert result["events"][0]["frame_id"] == "L23_V001_f0002"


def test_local_concept_coherence_outranks_scattered_story_terms() -> None:
    search = TemporalEventSearch(
        [
            _temporal_corpus(
                "L22_V010",
                "TP.HCM",
                "Khu phố được trang hoàng bằng hàng trăm chiếc đèn lồng dịp Trung thu",
                86,
            ),
            _temporal_corpus(
                "L22_V001",
                "Long Bình",
                "Dự án cải tạo đền thờ được triển khai bằng ngân sách. Thành phố "
                "đầu tư nhiều công trình văn hóa ở khu vực phường Long Bình",
                21,
            ),
        ]
    )

    result = search.search("phố lồng đèn", top_k_videos=2)

    assert result["selected_video"]["video_id"] == "L22_V010"
    assert result["candidates"][0]["frame_id"] == "L22_V010_f0086"
    assert [item["video_id"] for item in result["candidates"]] == ["L22_V010"]


def test_real_l22_temporal_search_returns_one_video_for_all_events() -> None:
    pilot = ROOT / "data/processed/video_understanding/L22/L22_V001/pilot"
    if not (pilot / "timeline.json").is_file():
        pytest.skip("requires the locally generated L22_V001 pilot")
    corpora = discover_temporal_corpus(
        output_root=ROOT / "data/processed/video_understanding",
        caption_dir=ROOT / "data/metadata/caption",
        asr_dir=ROOT / "data/metadata/metadata_asr",
        map_dir=ROOT / "data/map-keyframes",
        keyframe_dir=ROOT / "data/keyframes",
        batch_ids=["L22"],
        video_ids=["L22_V001"],
    )
    result = TemporalEventSearch(corpora).search(
        "news about Barcelona temperature record\n"
        "E1: first time Barcelona temperature record is visible\n"
        "E2: first time extreme heat is shown"
    )

    assert result["selected_video"]["video_id"] == "L22_V001"
    assert result["selected_video"]["total_events"] == 2
    assert result["events"]
    assert all(item["frame_id"].startswith("L22_V001_f") for item in result["events"])
    assert all(item["native_frame_idx"] >= 0 for item in result["events"])


def test_timeline_v2_contains_only_temporal_fields_inside_existing_pilot() -> None:
    path = ROOT / "data/processed/video_understanding/L22/L22_V001/pilot/timeline.json"
    if not path.is_file():
        pytest.skip("requires the locally generated L22_V001 pilot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "video-timeline-v2"
    assert isinstance(payload.get("events"), list)
    assert all("temporal_anchors" in event for event in payload["events"])


def test_qwen_verifier_refines_selected_event_without_changing_video_retrieval(tmp_path: Path) -> None:
    class FakeVerifier:
        def __init__(self) -> None:
            self.requests = []

        def verify(self, request):
            self.requests.append(request)
            return TemporalVerificationResult(
                supported=True,
                timestamp_ms=1_250,
                confidence=0.92,
                selected_state="transition",
                verifier="fake-qwen",
                coarse_window_ms=(0, 4_000),
                refined_window_ms=(1_000, 1_500),
                reason="cyclists first enter the view",
            )

    frame = {
        "frame_id": "L23_V001_f0002",
        "keyframe_n": 2,
        "native_frame_idx": 50,
        "timestamp_ms": 2_000,
        "visual_text": "cyclists are visible on the city road",
        "asr_text": "",
        "ocr_text": "",
    }
    corpus = TemporalCorpus(
        video={
            "video_id": "L23_V001",
            "summary_vi": "bike race on a city road",
            "summary_en": "",
            "search_text": "bike race on a city road",
            "main_locations": [],
            "main_entities": ["cyclists"],
        },
        events=[
            {
                "event_id": "L23_V001_cyclists",
                "description_vi": "cyclists are visible on the city road",
                "search_text": "cyclists are visible on the city road",
                "start_ms": 2_000,
                "end_ms": 3_000,
                "temporal_anchors": [],
                "_frames": [frame],
            }
        ],
        frames=[frame],
    )
    (tmp_path / "L23_V001.mp4").touch()
    verifier = FakeVerifier()

    result = TemporalEventSearch(
        [corpus],
        temporal_verifier=verifier,
        video_dir=tmp_path,
    ).search("bike race on a city road\nE1: first cyclists visible")

    assert result["selected_video"]["video_id"] == "L23_V001"
    assert result["events"][0]["timestamp_ms"] == 1_250
    assert result["events"][0]["native_frame_idx"] == 31
    assert result["events"][0]["verification"]["verifier"] == "fake-qwen"
    assert len(verifier.requests) == 1


def test_qwen_cannot_replace_a_verified_first_event_with_a_later_confident_candidate(tmp_path: Path) -> None:
    class FakeVerifier:
        def verify(self, request):
            is_late = request.event_id.endswith("late")
            return TemporalVerificationResult(
                supported=True,
                timestamp_ms=5_000 if is_late else 1_250,
                confidence=0.99 if is_late else 0.72,
                selected_state="transition",
                verifier="fake-qwen",
                coarse_window_ms=(0, 6_000),
                reason="visible cyclists",
            )

    def event(event_id: str, timestamp_ms: int) -> dict[str, object]:
        frame = {
            "frame_id": f"L23_V001_f{timestamp_ms // 1_000:04d}",
            "keyframe_n": timestamp_ms // 1_000,
            "native_frame_idx": timestamp_ms // 40,
            "timestamp_ms": timestamp_ms,
            "visual_text": "cyclists are visible on the city road",
            "asr_text": "",
            "ocr_text": "",
        }
        return {
            "event_id": event_id,
            "description_vi": "cyclists are visible on the city road",
            "search_text": "cyclists are visible on the city road",
            "start_ms": timestamp_ms,
            "end_ms": timestamp_ms + 1_000,
            "temporal_anchors": [],
            "_frames": [frame],
        }

    early = event("L23_V001_early", 2_000)
    late = event("L23_V001_late", 5_000)
    corpus = TemporalCorpus(
        video={
            "video_id": "L23_V001",
            "summary_vi": "bike race on a city road",
            "summary_en": "",
            "search_text": "bike race on a city road",
            "main_locations": [],
            "main_entities": ["cyclists"],
        },
        events=[early, late],
        frames=[early["_frames"][0], late["_frames"][0]],
    )
    (tmp_path / "L23_V001.mp4").touch()

    result = TemporalEventSearch(
        [corpus],
        temporal_verifier=FakeVerifier(),
        video_dir=tmp_path,
        verifier_candidates=2,
    ).search("bike race on a city road\nE1: first cyclists visible")

    assert result["events"][0]["event_id"] == "L23_V001_early"
    assert result["events"][0]["timestamp_ms"] == 1_250
