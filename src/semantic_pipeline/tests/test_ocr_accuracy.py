"""Task 5 - Accuracy Test: đo độ chuẩn xác của nội dung OCR.

Ground truth được đọc thủ công từ chính ảnh gốc trong sample_frames/.
Đo trên `ocr_text_raw` (output thô của VietOCR, TRƯỚC khi dịch) — vì đây mới là
thứ phản ánh năng lực OCR; nếu đo trên `ocr_text` (đã dịch) thì lỗi dịch sẽ lẫn
vào lỗi đọc, không tách bạch được.
"""

import difflib
import json
import re
from pathlib import Path

import pytest

METADATA_PATH = Path(__file__).resolve().parent.parent / "sample_frames" / "metadata.json"

# Ground truth: gõ tay từ ảnh gốc.
GROUND_TRUTH = {
    "vid01_f0002": (
        "Question 7: Triangle ABC has 3 internal lines (4 base parts). "
        "A horizontal line cuts through the middle two sections. "
        "How many triangles are there? A B C"
    ),
    "vid01_f0003": (
        "Method 2: Base Formula "
        "On Base BC (4 parts): 1 + 2 + 3 + 4 = 10. "
        "On Partial Line (2 parts): 1 + 2 = 3. "
        "Total = 10 + 3 = 13"
    ),
    "vid02_f0001": (
        "Thuyền trưởng Râu Đen phát hiện ra một hòn đảo hoang và chôn những đồng tiền "
        "vàng của mình vào 6 chiếc bao tải ma thuật, xếp thành một hàng ngang. "
        "Ông đặt một số lượng tiền vàng vào bao tải thứ nhất."
    ),
}


def _words(text: str) -> list[str]:
    # Bỏ dấu câu, giữ chữ + số. Với BM25 thì TỪ mới quan trọng, không phải dấu câu.
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def word_accuracy(truth: str, predicted: str) -> float:
    """Tỉ lệ từ đọc đúng (0.0 - 1.0), so khớp theo thứ tự."""
    truth_words, pred_words = _words(truth), _words(predicted)
    if not truth_words:
        return 1.0
    matcher = difflib.SequenceMatcher(None, truth_words, pred_words)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(truth_words)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ocr_by_frame(records) -> dict[str, str]:
    return {r["frame_id"]: r["ocr_text_raw"] for r in records}


class TestOcrAccuracy:
    def test_english_text(self, ocr_by_frame):
        """Chữ tiếng Anh thuần: kỳ vọng gần như hoàn hảo."""
        frame_id = "vid01_f0002"
        acc = word_accuracy(GROUND_TRUTH[frame_id], ocr_by_frame[frame_id])
        print(f"\n  {frame_id} (tiếng Anh)  : {acc:.1%}")
        assert acc >= 0.95

    def test_vietnamese_diacritics(self, ocr_by_frame):
        """Chữ tiếng Việt CÓ DẤU — đây là lý do phải thay Florence-2 bằng VietOCR."""
        frame_id = "vid02_f0001"
        truth = GROUND_TRUTH[frame_id]
        predicted = ocr_by_frame[frame_id][: len(truth) + 40]  # ground truth chỉ là đoạn đầu
        acc = word_accuracy(truth, predicted)
        print(f"  {frame_id} (tiếng Việt) : {acc:.1%}")
        assert acc >= 0.90

    def test_math_symbols(self, ocr_by_frame):
        """Ký hiệu toán (+, =, •) — VietOCR đọc sai, ensemble lấy PaddleOCR để sửa.

        Trước ensemble: '1 + 2 + 3 + 4 = 10' bị đọc thành '1%2%344-10'.
        """
        frame_id = "vid01_f0003"
        text = ocr_by_frame[frame_id]
        acc = word_accuracy(GROUND_TRUTH[frame_id], text)
        print(f"  {frame_id} (toán học)   : {acc:.1%}")
        assert acc >= 0.95

        # Assert thẳng vào ký hiệu: đây mới là thứ ensemble sinh ra để sửa.
        assert "1 + 2 + 3 + 4 = 10" in text, f"ký hiệu toán vẫn sai: {text!r}"
        assert "%" not in text, "dấu '+' lại bị đọc thành '%'"

    def test_no_frame_has_empty_ocr(self, ocr_by_frame):
        """Cả 24 frame đều phải đọc ra được ít nhất một ít chữ."""
        empty = [fid for fid, text in ocr_by_frame.items() if not text.strip()]
        assert not empty, f"Các frame OCR rỗng: {empty}"


class TestTranslationLayer:
    """Kiểm tra tầng dịch: dòng tiếng Việt được dịch, dòng tiếng Anh giữ nguyên."""

    def test_english_frame_is_not_translated(self, records):
        """Ảnh tiếng Anh: ocr_text phải Y HỆT ocr_text_raw (không đi qua translator)."""
        record = next(r for r in records if r["frame_id"] == "vid01_f0002")
        assert record["ocr_text"] == record["ocr_text_raw"]

    def test_vietnamese_frame_is_translated(self, records):
        """Ảnh tiếng Việt: ocr_text phải KHÁC bản gốc, và không còn dấu tiếng Việt."""
        record = next(r for r in records if r["frame_id"] == "vid02_f0001")
        assert record["ocr_text"] != record["ocr_text_raw"]

        vietnamese_chars = re.compile(r"[àáạảãăằắâầấêềếôồốơờớưừứđ]", re.IGNORECASE)
        assert not vietnamese_chars.search(record["ocr_text"]), "ocr_text còn sót tiếng Việt"

    def test_every_record_has_required_fields(self, records):
        for record in records:
            assert {"frame_id", "video_name", "frame_index", "caption", "ocr_text"} <= record.keys()
