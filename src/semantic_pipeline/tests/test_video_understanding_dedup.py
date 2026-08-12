from __future__ import annotations

from semantic_pipeline.video_understanding.models import ASRSegment, MicroScene, StoryCandidate
from semantic_pipeline.video_understanding.pipeline import (
    _merge_duplicate_candidates,
    _normalize_story_evidence,
    _sort_story_candidates,
)


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


def _scene(scene_id: str, start_ms: int, end_ms: int, asr_refs: list[int]) -> MicroScene:
    return MicroScene(
        scene_id=scene_id,
        scene_type="broll_or_other",
        frame_indices=[1],
        start_ms=start_ms,
        end_ms=end_ms,
        representative_keyframe_n=1,
        asr_segment_indices=asr_refs,
    )


def test_asr_only_story_is_mapped_to_its_visual_scene() -> None:
    candidate = StoryCandidate(
        title="Bản tin chỉ có lời dẫn",
        summary="Nội dung được nói trong voice-over.",
        topics=[],
        entities=[],
        locations=[],
        scene_ids=[],
        asr_segment_indices=[7],
    )
    scenes = [_scene("scene-1", 0, 5_000, [7]), _scene("scene-2", 6_000, 10_000, [])]
    segments = {7: ASRSegment(index=7, start=1.0, end=2.0, text="nội dung")}

    result, stats = _normalize_story_evidence([candidate], scenes, segments)

    assert [item.scene_ids for item in result] == [["scene-1"]]
    assert stats["asr_only_stories_received"] == 1
    assert stats["asr_only_stories_mapped"] == 1
    assert stats["asr_only_stories_dropped"] == 0


def test_asr_only_story_too_far_from_visual_scene_is_dropped() -> None:
    candidate = StoryCandidate(
        title="Không có hình ảnh gần kề",
        summary="Lời dẫn không thể định vị.",
        topics=[],
        entities=[],
        locations=[],
        scene_ids=[],
        asr_segment_indices=[8],
    )
    scenes = [_scene("scene-1", 0, 1_000, [])]
    segments = {8: ASRSegment(index=8, start=30.0, end=31.0, text="xa")}

    result, stats = _normalize_story_evidence(
        [candidate], scenes, segments, max_asr_scene_gap_ms=1_000
    )

    assert result == []
    assert stats["asr_only_stories_dropped"] == 1
    assert stats["stories_dropped_without_evidence"] == 1


def test_story_candidates_are_sorted_by_scene_timestamp() -> None:
    scenes = [_scene("early", 0, 1_000, []), _scene("late", 5_000, 6_000, [])]
    segments: dict[int, ASRSegment] = {}
    late = _story("late", "late", "")
    late.scene_ids = ["late"]
    early = _story("early", "early", "")
    early.scene_ids = ["early"]

    result = _sort_story_candidates([late, early], scenes, segments)

    assert [item.scene_ids for item in result] == [["early"], ["late"]]
