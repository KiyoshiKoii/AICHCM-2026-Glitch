"""Task 5 - Test tầng BM25 (tokenizer, xếp hạng, khóa chính)."""

import json

import pytest

from database import tokenize


class TestTokenizer:
    """tokenize() dùng CHUNG cho document lẫn query -> hỏng ở đây là hỏng cả hệ thống."""

    def test_lowercase_and_split(self):
        assert tokenize("Red SHIRT!") == ["red", "shirt"]

    def test_removes_stopwords(self):
        # rank_bm25 có epsilon floor biến IDF âm thành dương -> stopword vẫn cộng
        # điểm nhiễu nếu không lọc thủ công.
        assert "a" not in tokenize("a man in the park")
        assert "the" not in tokenize("a man in the park")

    def test_stemming_unifies_word_forms(self):
        # Query "motorbike" (số ít) phải khớp document ghi "motorbikes" (số nhiều).
        assert tokenize("motorbike") == tokenize("motorbikes")
        assert tokenize("drop") == tokenize("dropping")
        assert tokenize("walk") == tokenize("walking")

    def test_irregular_plurals(self):
        # Porter bó tay với bất quy tắc; KIS toàn query về người nên phải xử lý.
        assert tokenize("men") == tokenize("man")
        assert tokenize("women") == tokenize("woman")
        assert tokenize("children") == tokenize("child")

    def test_empty_and_punctuation_only(self):
        assert tokenize("") == []
        assert tokenize("!!! ???") == []

    def test_vietnamese_diacritics_are_ascii_folded(self):
        assert tokenize("người và phương tiện") == tokenize("nguoi phuong tien")


class TestSearch:
    def test_finds_expected_frame(self, db):
        """Ảnh về ERD/Booking phải đứng đầu khi query đúng chủ đề của nó."""
        results = db.search(["booking", "entity", "record"], top_k=5)
        assert results[0]["frame_id"] == "vid01_f0001"

    def test_frame_id_maps_to_correct_metadata(self, db):
        """Khóa chính: frame_id phải khớp video_name + frame_index."""
        for item in db.search(["triangle"], top_k=10):
            video = item["video_name"].removesuffix(".mp4")
            assert item["frame_id"].startswith(video)
            assert item["frame_id"] == f"{video}_f{item['frame_index']:04d}"

    def test_stemming_works_end_to_end(self, db):
        """Query số nhiều vẫn tìm được document viết số ít."""
        assert db.search(["triangles"], top_k=3), "query số nhiều không ra kết quả nào"

    def test_zero_score_results_filtered_out(self, db):
        """Document không chứa từ khóa nào -> loại bỏ, không trả rác cho Dev 3."""
        for item in db.search(["booking"], top_k=100):
            assert item["score"] > 0

    def test_no_match_returns_empty(self, db):
        assert db.search(["zzzznonexistentkeyword"], top_k=5) == []

    def test_scores_json_serialisable(self, db):
        """get_scores() trả numpy.float64 -> phải ép float, nếu không API crash."""
        import json

        json.dumps(db.search(["booking"], top_k=5))  # không được ném TypeError


    def test_sql_semantic_terms_find_not_exists_frame(self, db):
        results = db.search(["sql query"], top_k=5)
        assert results[0]["frame_id"] == "vid03_f0004"
        record = next(r for r in db.records if r["frame_id"] == "vid03_f0004")
        assert record["code"]["language"] == "sql"
        assert "not exists" in record["code"]["patterns"]
        assert "sql query template" in record["code"]["search_terms"]

    def test_directory_metadata_indexes_btc_object_and_media_text(self, tmp_path):
        from database import TextDatabase

        first = [
            {
                "frame_id": "L21_V001_f0010",
                "video_name": "L21_V001",
                "frame_index": 10,
                "object_text": "1 motorcycle at center, aliases xe may",
                "video_title": "Giao thông thành phố",
            }
        ]
        second = [
            {
                "frame_id": "L21_V002_f0020",
                "video_name": "L21_V002",
                "frame_index": 20,
                "object_text": "1 teddy bear at center",
            }
        ]
        third = [
            {
                "frame_id": "L21_V003_f0030",
                "video_name": "L21_V003",
                "frame_index": 30,
                "object_text": "1 laptop at center",
            }
        ]
        (tmp_path / "L21_V001.json").write_text(
            json.dumps(first), encoding="utf-8"
        )
        (tmp_path / "L21_V002.json").write_text(
            json.dumps(second), encoding="utf-8"
        )
        (tmp_path / "L21_V003.json").write_text(
            json.dumps(third), encoding="utf-8"
        )

        database = TextDatabase(tmp_path)

        assert len(database.records) == 3
        assert database.search(["xe máy"], top_k=1)[0]["frame_id"] == "L21_V001_f0010"
        assert database.search(["teddy bear"], top_k=1)[0]["frame_id"] == "L21_V002_f0020"


class TestPerformance:
    """Task 5 - Speed Test: đo tốc độ query BM25 trên RAM."""

    def test_query_latency_under_50ms(self, db):
        import statistics
        import time

        keywords = ["man", "red shirt", "triangle", "booking", "record"]
        db.search(keywords, top_k=200)  # warm-up (lru_cache của stemmer)

        timings = []
        for _ in range(100):
            start = time.perf_counter()
            db.search(keywords, top_k=200)
            timings.append((time.perf_counter() - start) * 1000)

        mean_ms = statistics.mean(timings)
        p95_ms = sorted(timings)[94]
        print(f"\n  BM25 query: trung bình {mean_ms:.3f} ms | p95 {p95_ms:.3f} ms")

        # Task yêu cầu API < 500ms. Riêng BM25 phải nhanh hơn nhiều lần.
        assert p95_ms < 50, f"BM25 quá chậm: p95 = {p95_ms:.1f} ms"

    def test_index_build_time(self):
        import time

        from database import TextDatabase

        start = time.perf_counter()
        database = TextDatabase()
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n  Dựng index: {elapsed_ms:.1f} ms cho {len(database.records)} documents")

        # Index chỉ dựng 1 lần lúc server khởi động, nên vài giây vẫn chấp nhận được.
        assert elapsed_ms < 5000
