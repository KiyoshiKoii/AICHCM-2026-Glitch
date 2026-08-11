from __future__ import annotations

from semantic_pipeline.video_understanding.models import StoryCandidate
from semantic_pipeline.video_understanding.pipeline import _merge_duplicate_candidates


def _story(title: str, summary: str, location: str) -> StoryCandidate:
    return StoryCandidate(
        title=title,
        summary=summary,
        topics=[],
        entities=[],
        locations=[location],
        scene_ids=[title],
        asr_segment_indices=[],
    )


def test_overlapping_news_cards_are_merged_by_text_and_anchor() -> None:
    result = _merge_duplicate_candidates(
        [
            _story(
                "Nhiệt độ tại Barcelona đạt mức kỷ lục",
                "Barcelona ghi nhận nhiệt độ 40 độ C, cao nhất trong 110 năm.",
                "Barcelona",
            ),
            _story(
                "Barcelona ghi nhận kỷ lục nhiệt độ cao nhất trong 110 năm",
                "Thành phố Barcelona trải qua đợt nóng kỷ lục.",
                "Barcelona",
            ),
        ]
    )
    assert len(result) == 1
    assert set(result[0].scene_ids) == {
        "Nhiệt độ tại Barcelona đạt mức kỷ lục",
        "Barcelona ghi nhận kỷ lục nhiệt độ cao nhất trong 110 năm",
    }


def test_different_events_with_only_shared_location_are_kept() -> None:
    result = _merge_duplicate_candidates(
        [
            _story("Tai nạn giao thông tại TP.HCM", "Một vụ tai nạn xảy ra.", "TP.HCM"),
            _story("Khởi công công trình tại TP.HCM", "Một dự án được khởi công.", "TP.HCM"),
        ]
    )
    assert len(result) == 2


def test_shared_scene_does_not_merge_unrelated_news_cards() -> None:
    left = _story(
        "Nhiệt độ tại Barcelona đạt mức kỷ lục",
        "Nhiệt độ cao nhất trong 110 năm.",
        "Barcelona",
    )
    right = _story(
        "Đấu giá thư tay của Công nương Diana",
        "Một bức thư được bán đấu giá tại Anh.",
        "Anh",
    )
    left.scene_ids = ["shared-scene"]
    right.scene_ids = ["shared-scene"]

    assert len(_merge_duplicate_candidates([left, right])) == 2
