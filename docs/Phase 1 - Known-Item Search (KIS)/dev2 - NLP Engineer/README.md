# Dev 2 — Semantic Pipeline (NLP Engineer)

Source: **`src/semantic_pipeline/`** · Branch: **`feat/semantic-pipeline`**

| Task | Nội dung | Trạng thái |
|---|---|---|
| **Task 1** | Trích xuất ngữ nghĩa (caption + OCR) → `metadata.json` | ✅ Xong |
| **Task 2** | Text DB bằng BM25 (`rank_bm25`) | ✅ Xong |
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

## ✅ Đã làm được (Task 2)

Text DB bằng BM25 trong [`database.py`](../../../src/semantic_pipeline/database.py) — class `TextDatabase`:

```
metadata.json ──► ghép caption + ocr_text ──► tokenize ──► BM25Okapi
                                                              │
   keywords (tiếng Anh) ──► tokenize ──► get_scores() ────────┘
                                              └──► sort ──► top_k
```

- **Tokenize** = lowercase → tách từ → **bỏ stopword** → **chuẩn hóa bất quy tắc** → **stemming**. Dùng **chung một hàm** cho cả document lẫn query, nếu khác nhau sẽ không bao giờ khớp.
  - Stemming (Porter): query `"motorbike"` khớp document `"motorbikes"`, `"drop"` khớp `"dropping"`.
  - Bảng bất quy tắc: `men→man`, `women→woman`, `children→child`, `people→person`... (Porter bó tay với các từ này, mà query KIS lại toàn về **người**).
  - Bỏ stopword: cắt **36%** token thừa trong index.
- Chỉ index `caption` + `ocr_text` (đều tiếng Anh). **Không** index `ocr_text_raw` (tiếng Việt) vì keywords Dev 3 gửi sang là tiếng Anh.
- `corpus[i]` ↔ `records[i]` để map ngược ra `frame_id` (khóa chính).
- Trả **điểm BM25 thô, KHÔNG chuẩn hóa** (đúng luật task.md), lọc bỏ `score <= 0`.
- `float(score)` — bắt buộc, vì `numpy.float64` không JSON-serialize được (Task 3 sẽ crash).
- Output đúng API Contract: `frame_id`, `score`, `video_name`, `frame_index`.

**Đã kiểm chứng:** 12/12 test PASS trên corpus giả + xếp hạng chính xác trên 24 record thật (`booking/entity/record` → đúng ảnh ERD; `snowden/whistleblower` → đúng ảnh báo Guardian).

**Chạy:** `python src/semantic_pipeline/database.py`

> ⚠️ **Corpus phải đủ lớn.** Với ~2 document, công thức IDF của BM25 ra **0 hoặc số âm** → mọi điểm bằng 0, không có kết quả. Đây không phải bug — cần chạy extractor đủ nhiều ảnh trước.

---

## ⚠️ Điểm cần cải thiện

### Task 1

| # | Vấn đề | Hướng cải thiện |
|---|---|---|
| 1 | Nhận diện tiếng Việt **chỉ dựa vào dấu** → chữ Việt không dấu (`"khong tach"`) không được dịch | Dùng language detection thật (`fasttext lid.176`) thay regex. Cẩn thận: nới lỏng quá dễ dịch nhầm từ tiếng Anh |
| 2 | VietOCR còn sạn: ký tự `—`, `/` thành `?`; hallucination lặp từ; vài lỗi dấu lẻ | Lọc theo confidence (`return_prob=True`); bật `beamsearch=True` (chậm hơn nhưng chuẩn hơn); ensemble với PaddleOCR-rec |
| 3 | Caption Florence-2 tự "đọc" chữ và đọc sai tiếng Việt | Ưu tiên thấp (frame thật ít chữ). Nếu cần: bỏ phần trong ngoặc kép của caption |
| 4 | **Chậm trên CPU** (chạy 4 model tuần tự) | Giảm `num_beams` của Florence-2 (3 → 1); chạy GPU (code đã auto-detect `cuda`, riêng Paddle cần `paddlepaddle-gpu`); thêm cơ chế resume để chạy lại không mất công |

### Task 2

| # | Vấn đề | Trạng thái / Hướng cải thiện |
|---|---|---|
| 5 | Porter stemmer không xử lý được bất quy tắc (`men` ↛ `man`) → query về **người** bị trượt | ✅ **Đã sửa** bằng bảng ánh xạ thủ công. ⚠️ **Đừng dùng `WordNetLemmatizer`** — đã thử: nó **vẫn không sửa được `men`→`man`**, lại cần POS tag và làm hỏng ca khác (`riding`→`rid`) |
| 6 | Chưa loại stopword → nhiễu điểm | ✅ **Đã sửa**, cắt 36% token. ⚠️ Đừng tin "IDF tự hạ điểm từ phổ biến" — `rank_bm25` có **epsilon floor** biến IDF âm thành **dương** (`0.25 × avg_idf`), nên stopword **vẫn cộng điểm nhiễu** |
| 7 | Index dựng lại mỗi lần khởi động, giữ hết trên RAM | ⏸️ **Chưa cần.** Index chỉ dựng **1 lần lúc server start** (trong `__init__`), không phải mỗi request → lợi ích hiện tại bằng 0. Data lớn (100GB) mới cần → **Elasticsearch** (Task 4) |
| 8 | `caption` và `ocr_text` gộp chung, **trọng số bằng nhau** — OCR (chữ thật trong ảnh) đáng tin hơn caption do model sinh ra | ⏸️ **Chưa làm.** `rank_bm25` **không hỗ trợ BM25F**; muốn làm phải dựng 2 index rồi cộng trọng số — nhưng **chưa có ground truth để chỉnh trọng số**, chỉnh mò dễ làm tệ hơn. Đợi Task 5 có data đánh giá |

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
