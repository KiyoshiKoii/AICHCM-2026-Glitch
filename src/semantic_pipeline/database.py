# Task 2: Setup Database Văn bản bằng rank_bm25
# Đọc file metadata.json, tokenize và nạp vào máy tìm kiếm BM25
# Task 3 (server.py) chỉ cần gọi: TextDatabase(...).search(keywords, top_k)

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

try:
    from code_classifier import classify_code
except ImportError:
    from .code_classifier import classify_code

# Tính từ vị trí file này, KHÔNG phụ thuộc thư mục đang chạy lệnh.
# (Trước đây dùng đường dẫn tương đối nên chỉ chạy được khi đứng ở gốc repo.)
DEFAULT_METADATA_PATH = Path(__file__).parent / "sample_frames" / "metadata.json"

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
    # High-frequency Vietnamese function words after ASCII folding.
    "va", "cua", "la", "mot", "nhung", "cac", "cho", "voi", "tai", "tu",
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
    folded = unicodedata.normalize(
        "NFKD", text.casefold().replace("đ", "d")
    )
    folded = "".join(
        character
        for character in folded
        if unicodedata.category(character) != "Mn"
    )
    return [
        _normalize(word)
        for word in re.findall(r"[a-z0-9]+", folded)
        if word not in STOPWORDS
    ]


def _metadata_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"{path} không chứa metadata JSON theo video.")
        return files
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {path} — chạy extractor/metadata_builder trước."
        )
    return [path]


def _load_search_records(path: Path) -> list[dict]:
    """Load only fields required by BM25 instead of retaining spatial payloads."""
    records: list[dict] = []
    for source in _metadata_files(path):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{source} phải chứa một JSON array.")
        for position, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"{source}[{position}] phải là JSON object.")
            missing = {"frame_id", "video_name", "frame_index"} - record.keys()
            if missing:
                raise ValueError(
                    f"{source}[{position}] thiếu field bắt buộc: {sorted(missing)}"
                )
            records.append(
                {
                    "frame_id": record["frame_id"],
                    "video_name": record["video_name"],
                    "frame_index": record["frame_index"],
                    "caption": record.get("caption", ""),
                    "ocr_text": record.get("ocr_text", ""),
                    "ocr_text_raw": record.get("ocr_text_raw", ""),
                    "object_text": record.get("object_text", ""),
                    "video_title": record.get("video_title", ""),
                    "video_description": record.get("video_description", ""),
                    "video_keywords": record.get("video_keywords", []),
                    "code": record.get("code", {}),
                }
            )
    return records


class TextDatabase:
    """BM25 over visual text, BTC objects/media, and derived code terms."""

    def __init__(self, metadata_path: str | Path = DEFAULT_METADATA_PATH):
        path = Path(metadata_path)
        self.records = _load_search_records(path)
        if not self.records:
            raise ValueError(f"{path} không có metadata record nào.")

        # corpus[i] tương ứng records[i] -> đây là cách map ngược ra frame_id.
        # Index caption + translated OCR trực tiếp. Raw OCR không được đưa nguyên
        # văn vào BM25, nhưng classifier có thể dùng nó làm evidence để sinh các
        # alias tiếng Anh như "correlated subquery".
        corpus = []
        for record in self.records:
            code = classify_code(
                record.get("caption", ""),
                record.get("ocr_text", ""),
                record.get("ocr_text_raw", ""),
            )
            # Classify legacy Task 1 metadata in memory; no source overwrite is
            # required just to make SQL semantic terms searchable.
            record["code"] = code.model_dump(mode="json")
            # Raw SQL syntax is already present in ocr_text. Add only semantic
            # aliases here; duplicating every matched pattern would distort the
            # established OCR ranking for queries such as NOT IN + NULL.
            derived_text = " ".join(code.search_terms)
            video_keywords = record.get("video_keywords", [])
            keyword_text = (
                " ".join(video_keywords)
                if isinstance(video_keywords, list)
                else str(video_keywords)
            )
            corpus.append(
                tokenize(
                    f"{record.get('caption', '')} {record.get('ocr_text', '')} "
                    f"{record.get('object_text', '')} "
                    f"{record.get('video_title', '')} "
                    f"{record.get('video_description', '')} {keyword_text} "
                    f"{derived_text}"
                )
            )
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
