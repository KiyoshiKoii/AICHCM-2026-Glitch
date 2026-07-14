# Dev 2 — Semantic Pipeline (NLP Engineer)

Source: **`src/semantic_pipeline/`** · Branch: **`feat/semantic-pipeline`**

| Task | Nội dung | Trạng thái |
|---|---|---|
| **Task 1** | Trích xuất ngữ nghĩa (caption + OCR) → `metadata.json` | ✅ Xong |
| Task 2 | Text DB bằng BM25 (`rank_bm25`) | ⬜ |
| Task 3 | Internal API (FastAPI, port 8002) | ⬜ |
| Task 4 | R&D (Elasticsearch, Spatial Reasoning, Entity Extraction) | ⬜ |
| Task 5 | Unit Testing & Performance Testing | ⬜ |

---

## ✅ Đã làm được (Task 1)

Pipeline trích xuất ngữ nghĩa từ ảnh, chạy được **cả tiếng Việt lẫn tiếng Anh**:

```
Ảnh ─┬─► Florence-2      ──► caption (mô tả cảnh, tiếng Anh)
     └─► PaddleOCR (detect vùng chữ) ──► VietOCR (đọc chữ)
           └─► dòng có dấu tiếng Việt ──► dịch sang tiếng Anh
               dòng tiếng Anh          ──► giữ nguyên
```

- **Dịch theo từng dòng** (không dịch cả cục) → tên biến/mã tiếng Anh không bị dịch bậy.
- Ảnh không có chữ vẫn có `caption`.
- Output `metadata.json` gồm: `frame_id`, `video_name`, `frame_index`, `caption`, `ocr_text` (luôn tiếng Anh, để nạp BM25), `ocr_text_raw` (gốc, để đối chiếu).
- Đã tối ưu: **crop padding** (tránh cắt mất dấu thanh) + **batch VietOCR** (đọc cả loạt vùng chữ 1 lần thay vì tuần tự).
- Đã commit 24 ảnh mẫu (`sample_frames/`) làm fixture test.

**Chạy:**
```bash
python src/semantic_pipeline/extractor.py --limit 2   # test nhanh
python src/semantic_pipeline/extractor.py             # chạy full
```

---

## ⚠️ Điểm cần cải thiện

| # | Vấn đề | Hướng cải thiện |
|---|---|---|
| 1 | Nhận diện tiếng Việt **chỉ dựa vào dấu** → chữ Việt không dấu (`"khong tach"`) không được dịch | Dùng language detection thật (`fasttext lid.176`) thay regex. Cẩn thận: nới lỏng quá dễ dịch nhầm từ tiếng Anh |
| 2 | VietOCR còn sạn: ký tự `—`, `/` thành `?`; hallucination lặp từ; vài lỗi dấu lẻ | Lọc theo confidence (`return_prob=True`); bật `beamsearch=True` (chậm hơn nhưng chuẩn hơn); ensemble với PaddleOCR-rec |
| 3 | Caption Florence-2 tự "đọc" chữ và đọc sai tiếng Việt | Ưu tiên thấp (frame thật ít chữ). Nếu cần: bỏ phần trong ngoặc kép của caption |
| 4 | **Chậm trên CPU** (chạy 4 model tuần tự) | Giảm `num_beams` của Florence-2 (3 → 1); chạy GPU (code đã auto-detect `cuda`, riêng Paddle cần `paddlepaddle-gpu`); thêm cơ chế resume để chạy lại không mất công |

---

## 🪤 Bẫy kỹ thuật (đừng debug lại)

| Lỗi | Cách xử lý |
|---|---|
| Florence-2 crash khi load | Pin **`transformers==4.49.0`** (bản ≥ 4.50 bug `trust_remote_code`). ⚠️ Env dùng chung — Dev 3 cần bản mới hơn thì phải bàn tách env |
| `einops` bị hạ cấp → hỏng Florence-2; `No module named 'pkg_resources'` | `vietocr` ghim cứng lib cũ → cài bằng **`pip install vietocr --no-deps`** |
| `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` | Bug oneDNN của `paddlepaddle 3.x` trên CPU → truyền **`enable_mkldnn=False`** vào `PaddleOCR(...)`. Biến môi trường `FLAGS_use_mkldnn=0` **không có tác dụng** |
| `This tokenizer cannot be instantiated` | Thiếu **`sentencepiece`** (tokenizer MarianMT cần) |

---

## 🧭 Vì sao chọn các model này

- **Bỏ `<OCR>` của Florence-2**: đọc tiếng Việt sai nặng (`"Nếu giữ"` → `"Neu giüt"`).
- **PaddleOCR chỉ để detect**, không đọc chữ: PP-OCRv6 rớt dấu kép (`"Nếu giữ"` → `"Nu gi"`).
- **VietOCR để đọc chữ**: train chuyên tiếng Việt, đọc đúng dấu.
- **Giữ Helsinki-NLP**: đã kiểm chứng — cho ăn tiếng Việt chuẩn thì dịch rất tốt. Bản dịch rác ban đầu là do **OCR sai**, không phải lỗi translator.
