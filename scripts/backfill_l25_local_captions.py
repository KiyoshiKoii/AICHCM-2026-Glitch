"""Backfill the L25 captions that could not be obtained from Gemini.

The affected images were inspected manually.  Records retain the same schema as
Gemini output and are explicitly marked with ``local_caption_fallback`` so they
can be identified or regenerated later.
"""

from __future__ import annotations

import json
from pathlib import Path


CAPTION_ROOT = Path("data/metadata/caption/L25")


def _record(
    frame_id: str,
    caption: str,
    detailed_caption: str,
    caption_vi: str,
    detailed_caption_vi: str,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "visual_source_frame_id": frame_id,
        "ocr_source_frame_id": "",
        "quality_flags": ["local_caption_fallback"],
        "caption": caption,
        "detailed_caption": detailed_caption,
        "caption_vi": caption_vi,
        "detailed_caption_vi": detailed_caption_vi,
        "ocr_text": "",
        "news_ticker_text": "",
        "detections": [],
        "spatial_relations": [],
    }


def _physics_teacher(frame_id: str) -> dict[str, object]:
    return _record(
        frame_id,
        "A Vietnamese physics teacher gives a lesson beside a multiple-choice question about an LC circuit.",
        "A male teacher in a white shirt sits behind a laptop against a blue studio background. A physics question and answer options are displayed beside him.",
        "Một giáo viên Vật lý Việt Nam giảng bài bên cạnh câu hỏi trắc nghiệm về mạch LC.",
        "Một giáo viên nam mặc áo sơ mi trắng ngồi sau máy tính xách tay trước nền xanh. Bên cạnh là câu hỏi Vật lý và các lựa chọn đáp án.",
    )


def _physics_slide(frame_id: str, topic: str) -> dict[str, object]:
    return _record(
        frame_id,
        f"A Vietnamese physics lesson slide presents a multiple-choice {topic} problem with equations and a worked solution.",
        f"The full-screen educational slide contains a Vietnamese {topic} question, answer choices, and mathematical formulas used in the solution.",
        f"Slide bài giảng Vật lý bằng tiếng Việt trình bày câu hỏi trắc nghiệm về {topic}, kèm công thức và lời giải.",
        f"Slide học tập toàn màn hình có câu hỏi {topic} bằng tiếng Việt, các đáp án lựa chọn và các công thức toán học trong phần lời giải.",
    )


def _scene_record(frame_id: str, scene: str) -> dict[str, object]:
    captions = {
        "university_aerial": (
            "An aerial view shows a modern university building in a dense city neighborhood.",
            "The camera looks down at a large white-roofed university building surrounded by streets, trees, and nearby buildings.",
            "Góc quay trên cao cho thấy một tòa nhà đại học hiện đại giữa khu vực đô thị đông đúc.",
            "Máy quay nhìn từ trên xuống một tòa nhà đại học mái trắng, xung quanh là đường phố, cây xanh và các công trình lân cận.",
        ),
        "university_front": (
            "The exterior entrance of Saigon International University is shown with flagpoles in front.",
            "A bright daytime shot frames the curved facade and sign of Saigon International University, with several flags outside.",
            "Mặt tiền Trường Đại học Quốc tế Sài Gòn xuất hiện cùng các cột cờ phía trước.",
            "Cảnh ban ngày cho thấy mặt tiền cong và biển hiệu của Trường Đại học Quốc tế Sài Gòn, cùng nhiều lá cờ bên ngoài.",
        ),
        "accreditation": (
            "An accreditation certificate from IACBE is displayed on screen.",
            "A framed certificate for the Saigon International University School of Business Administration fills most of the screen.",
            "Một chứng nhận kiểm định của IACBE được hiển thị trên màn hình.",
            "Một chứng nhận đóng khung dành cho Trường Quản trị Kinh doanh của Đại học Quốc tế Sài Gòn chiếm gần hết khung hình.",
        ),
        "lectern_speaker": (
            "A man in a suit speaks at a flower-decorated lectern in an auditorium.",
            "An older bearded man addresses an audience from a podium, with a presentation screen behind him.",
            "Một người đàn ông mặc vest phát biểu tại bục có trang trí hoa trong hội trường.",
            "Một người đàn ông lớn tuổi có râu nói chuyện với khán giả từ bục phát biểu, phía sau là màn hình trình chiếu.",
        ),
        "woman_interview": (
            "A woman speaks while seated in a meeting room.",
            "A close-up shows a woman with braided hair in a patterned blouse speaking indoors, with chairs in the background.",
            "Một phụ nữ đang nói chuyện khi ngồi trong phòng họp.",
            "Cận cảnh một phụ nữ tóc tết mặc áo họa tiết đang nói trong nhà, phía sau có các ghế ngồi.",
        ),
        "robot_demo": (
            "People interact with a white-and-orange humanoid robot in a technology lab.",
            "In a bright laboratory, staff members stand beside a humanoid service robot and look at its display.",
            "Mọi người tương tác với một robot hình người màu trắng cam trong phòng thí nghiệm công nghệ.",
            "Trong phòng thí nghiệm sáng sủa, các nhân viên đứng cạnh robot dịch vụ hình người và quan sát màn hình của nó.",
        ),
        "robot_photo": (
            "A woman photographs a white-and-orange humanoid robot in a technology lab.",
            "A woman stands in front of a wheeled humanoid robot while holding up a phone in a laboratory with technology graphics on the wall.",
            "Một phụ nữ chụp ảnh robot hình người màu trắng cam trong phòng thí nghiệm công nghệ.",
            "Một phụ nữ đứng trước robot hình người có bánh xe và giơ điện thoại trong phòng thí nghiệm có các đồ họa công nghệ trên tường.",
        ),
        "black_transition": (
            "A black transition frame appears between video segments.",
            "The screen is fully black with no visible subjects or text.",
            "Khung hình chuyển cảnh màu đen xuất hiện giữa các đoạn video.",
            "Màn hình hoàn toàn đen, không có chủ thể hoặc chữ hiển thị.",
        ),
        "piano_stage": (
            "An empty concert hall stage is lit with colored spotlights around a grand piano.",
            "Rows of audience seats face a stage containing a grand piano, while blue and yellow beams shine through the dark auditorium.",
            "Sân khấu hòa nhạc trống được chiếu đèn màu xung quanh một cây đại dương cầm.",
            "Các hàng ghế khán giả hướng về sân khấu có một cây đại dương cầm, với các chùm đèn xanh và vàng chiếu trong khán phòng tối.",
        ),
    }
    return _record(frame_id, *captions[scene])


def _load_records(video_id: str) -> list[dict[str, object]]:
    path = CAPTION_ROOT / f"{video_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_records(video_id: str, records: list[dict[str, object]]) -> None:
    path = CAPTION_ROOT / f"{video_id}.json"
    records.sort(key=lambda item: str(item["frame_id"]))
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_missing(video_id: str, additions: list[dict[str, object]]) -> None:
    records = _load_records(video_id)
    existing = {str(record["frame_id"]) for record in records}
    duplicated = existing & {str(record["frame_id"]) for record in additions}
    if duplicated:
        raise RuntimeError(f"{video_id}: refusing to overwrite existing frame(s): {sorted(duplicated)}")
    records.extend(additions)
    _write_records(video_id, records)


def main() -> None:
    v031_indices = [*range(225, 234), *range(238, 247)]
    v031 = [
        _physics_teacher(f"L25_V031_f{index:04d}") if index <= 233 else _physics_slide(f"L25_V031_f{index:04d}", "LC circuit")
        for index in v031_indices
    ]

    v040: list[dict[str, object]] = []
    v040.extend(_physics_slide(f"L25_V040_f{index:04d}", "electromagnetic induction") for index in range(208, 230))
    v040.extend(_physics_slide(f"L25_V040_f{index:04d}", "transformer") for index in range(230, 255))
    scene_indices = {
        "university_aerial": [255],
        "university_front": [256],
        "accreditation": [257, 258],
        "lectern_speaker": [259, 260],
        "woman_interview": [261],
        "robot_demo": [262],
        "robot_photo": [263],
        "black_transition": [264],
        "piano_stage": [265, 266, 267],
    }
    for scene, indices in scene_indices.items():
        v040.extend(_scene_record(f"L25_V040_f{index:04d}", scene) for index in indices)

    _append_missing("L25_V031", v031)
    _append_missing("L25_V040", v040)
    print(f"Added {len(v031)} local fallback captions to L25_V031 and {len(v040)} to L25_V040.")


if __name__ == "__main__":
    main()
