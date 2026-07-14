# Task 2: Setup Database Văn bản bằng rank_bm25
# Đọc file metadata.json, tokenize và nạp vào máy tìm kiếm BM25
# Task 3 (server.py) chỉ cần gọi: TextDatabase(...).search(keywords, top_k)

import json
import re
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

DEFAULT_METADATA_PATH = "src/semantic_pipeline/sample_frames/metadata.json"

_stemmer = PorterStemmer()

# Porter chỉ cắt hậu tố theo luật nên bó tay với biến đổi bất quy tắc
# ("men" không về "man"). WordNetLemmatizer cũng KHÔNG sửa được (vẫn ra "men"),
# lại cần POS tag và làm hỏng ca khác ("riding" -> "rid"). Nên xử lý thủ công
# đúng các từ về NGƯỜI/bộ phận cơ thể — thứ mà query KIS gặp liên tục.
IRREGULAR = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "people": "person",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
}

# Stopword: rank_bm25 có "epsilon floor" — từ xuất hiện ở gần hết corpus lẽ ra
# phải có IDF <= 0, nhưng bị thay bằng số DƯƠNG (0.25 * avg_idf) nên vẫn cộng
# điểm nhiễu. Phải lọc thủ công. Chiếm ~36% token trong corpus.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "into", "through", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "then", "than", "so", "that", "this", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "am", "it",
    "its", "as", "there", "here", "he", "she", "they", "them", "his", "her",
    "their", "you", "your", "we", "our", "i", "me", "my",
}


@lru_cache(maxsize=100_000)
def _normalize(word: str) -> str:
    # Cache: cùng một từ lặp lại rất nhiều lần trong corpus.
    return _stemmer.stem(IRREGULAR.get(word, word))


def tokenize(text: str) -> list[str]:
    """Lowercase -> tách từ -> bỏ stopword -> chuẩn hóa bất quy tắc -> stemming.

    PHẢI dùng y hệt hàm này cho cả document lẫn query, nếu không sẽ không khớp.
    Stemming giúp query "motorbike" khớp document ghi "motorbikes", "drop" khớp
    "dropping"; bảng IRREGULAR giúp "men" khớp "man".

    "A man DROPPING red shirts!" -> ['man', 'drop', 'red', 'shirt']
    """
    return [
        _normalize(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS
    ]


class TextDatabase:
    """BM25 full-text search trên caption + ocr_text của từng frame."""

    def __init__(self, metadata_path: str = DEFAULT_METADATA_PATH):
        self.records = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not self.records:
            raise ValueError(f"{metadata_path} rỗng — chạy extractor.py trước.")

        # corpus[i] tương ứng records[i] -> đây là cách map ngược ra frame_id.
        # Chỉ lấy caption + ocr_text (đều tiếng Anh); KHÔNG lấy ocr_text_raw (tiếng Việt)
        # vì keywords Dev 3 gửi sang là tiếng Anh.
        corpus = [
            tokenize(f"{r.get('caption', '')} {r.get('ocr_text', '')}")
            for r in self.records
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, keywords: list[str], top_k: int = 200) -> list[dict]:
        """Trả về top_k frame khớp nhất, điểm BM25 THÔ (không chuẩn hóa).

        Format tuân thủ API Contract: [frame_id, score, video_name, frame_index]
        """
        query_tokens = tokenize(" ".join(keywords))
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.records, scores), key=lambda pair: pair[1], reverse=True
        )

        results = []
        for record, score in ranked[:top_k]:
            if score <= 0:  # không chứa từ khóa nào -> bỏ, tránh nhiễu cho Dev 3
                continue
            results.append(
                {
                    "frame_id": record["frame_id"],
                    # ép float: numpy.float64 không JSON-serialize được
                    "score": float(score),
                    "video_name": record["video_name"],
                    "frame_index": record["frame_index"],
                }
            )
        return results


if __name__ == "__main__":
    db = TextDatabase()
    print(f"Đã nạp {len(db.records)} documents từ {DEFAULT_METADATA_PATH}\n")

    for keywords in (["booking", "entity", "record"], ["triangle", "worksheet"]):
        print(f"--- Query: {keywords} ---")
        results = db.search(keywords, top_k=5)
        if not results:
            print("  (không có kết quả nào khớp)")
        for r in results:
            print(
                f"  {r['frame_id']:<14} score={r['score']:6.2f}  "
                f"{r['video_name']} #{r['frame_index']}"
            )
        print()
