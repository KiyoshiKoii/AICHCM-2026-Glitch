# Dev 2 — Semantic Pipeline (NLP Engineer)

Source: **`src/semantic_pipeline/`**

| Task | Nội dung | Trạng thái |
|---|---|---|
| **Task 1** | Trích xuất ngữ nghĩa (caption + OCR) → `metadata.json` | ✅ Xong |
| **Task 2** | Text DB bằng BM25 (`rank_bm25`) | ✅ Xong |
| **Task 3** | Internal API (FastAPI, port 8002) | ✅ Xong |
| Task 4 | R&D (Elasticsearch, Spatial Reasoning, Entity Extraction) | ⬜ |
| **Task 5** | Unit Testing & Performance Testing | ✅ Xong |

---

## Task 1 — Trích xuất ngữ nghĩa

### Đã làm

- Pipeline 4 model xử lý mỗi ảnh:
  - **Florence-2** (`microsoft/Florence-2-base`) → sinh caption mô tả cảnh (tiếng Anh).
  - **PaddleOCR** → detect vùng chữ trong ảnh.
  - **VietOCR** (`vgg_transformer`) → đọc chữ từ vùng đã detect, hỗ trợ tiếng Việt có dấu.
  - **Helsinki-NLP** (`opus-mt-vi-en`) → dịch từng dòng tiếng Việt sang tiếng Anh; dòng tiếng Anh giữ nguyên.
- **Ensemble OCR**: chọn model đọc chữ theo thế mạnh — dòng có dấu tiếng Việt lấy VietOCR, dòng tiếng Anh/ký hiệu toán lấy PaddleOCR. Không tốn thêm model vì cả hai vốn đã chạy sẵn.
- **Crop padding** (nới rộng 4px khi crop vùng chữ) → tránh cắt mất dấu thanh tiếng Việt.
- **Batch VietOCR** (`predict_batch`) → đọc cả loạt vùng chữ 1 lần thay vì tuần tự.
- Output `metadata.json` gồm: `frame_id`, `video_name`, `frame_index`, `caption`, `ocr_text` (tiếng Anh, cho BM25), `ocr_text_raw` (gốc, đối chiếu).
- Commit 24 ảnh mẫu (`sample_frames/`) làm fixture test.

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | Nhận diện tiếng Việt **chỉ dựa vào dấu** → chữ Việt không dấu (`"khong tach"`) không được dịch | Dùng language detection (`fasttext lid.176`) thay regex |
| 2 | VietOCR còn lỗi lẻ tẻ với ký tự đặc biệt (`—`, `/` → `?`) | Lọc theo confidence (`return_prob=True`); bật `beamsearch=True` (chậm hơn) |
| 3 | Caption Florence-2 tự "đọc" chữ trong ảnh và đọc sai tiếng Việt | Bỏ phần trong ngoặc kép của caption (ưu tiên thấp vì frame thật ít chữ) |
| 4 | **Chậm trên CPU** (chạy 4 model tuần tự) | Giảm `num_beams` Florence-2 (3→1); chạy GPU (`paddlepaddle-gpu`); thêm cơ chế resume |

### Lệnh chạy

```bash
# Test nhanh 2 ảnh
python src/semantic_pipeline/extractor.py --limit 2

# Chạy full toàn bộ sample_frames
python src/semantic_pipeline/extractor.py

# Chạy 1 ảnh cụ thể
python src/semantic_pipeline/extractor.py --image src/semantic_pipeline/sample_frames/vid01_f0001.png

# Tuỳ chỉnh input/output
python src/semantic_pipeline/extractor.py --input-dir <thư_mục_ảnh> --output <đường_dẫn_output.json>
```

---

## Task 2 — Text DB bằng BM25

### Đã làm

- Class `TextDatabase` trong `database.py` — đọc `metadata.json`, tokenize và nạp vào `BM25Okapi`.
- **Tokenize** = lowercase → tách từ → bỏ stopword → chuẩn hóa bất quy tắc → stemming (Porter). Dùng **chung 1 hàm** cho cả document lẫn query.
  - Stemming: `"motorbike"` khớp `"motorbikes"`, `"drop"` khớp `"dropping"`.
  - Bảng bất quy tắc: `men→man`, `women→woman`, `children→child`, `people→person`...
  - Bỏ stopword: cắt ~36% token thừa.
- Chỉ index `caption` + `ocr_text` (tiếng Anh). Không index `ocr_text_raw` (tiếng Việt).
- Trả điểm BM25 **thô, không chuẩn hóa**. Lọc bỏ `score <= 0`.
- `float(score)` — ép kiểu vì `numpy.float64` không JSON-serialize được.
- Output đúng API Contract: `frame_id`, `score`, `video_name`, `frame_index`.

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | Index dựng lại mỗi lần khởi động, giữ hết trên RAM | Chuyển sang **Elasticsearch** khi data lớn (Task 4) |
| 2 | `caption` và `ocr_text` gộp chung, **trọng số bằng nhau** — OCR đáng tin hơn caption | Cần BM25F (rank_bm25 không hỗ trợ) hoặc dựng 2 index cộng trọng số — cần ground truth để chỉnh |
| 3 | Corpus nhỏ (~2 doc) → IDF ra 0 hoặc âm → mọi điểm bằng 0 | Cần chạy extractor đủ nhiều ảnh trước khi test |

### Lệnh chạy

```bash
# Chạy demo search (query mẫu trên 24 frame)
python src/semantic_pipeline/database.py
```

---

## Task 3 — Internal API (FastAPI)

### Đã làm

- API server bằng **FastAPI** trong `server.py`, port **8002**.
- Endpoint `POST /internal/search/text`: nhận `{keywords, top_k}` → trả `{status, data: [{frame_id, score, video_name, frame_index}]}`.
- Endpoint `GET /health`: kiểm tra service sống chưa, index nạp bao nhiêu doc.
- Index BM25 dựng **1 lần lúc khởi động** (`lifespan`), không dựng lại mỗi request.
- Validate payload bằng Pydantic: `keywords: []` → 422, `keywords: ["zzzz"]` (không khớp) → 200 + `data: []`.
- Dùng `def` (không phải `async def`) → FastAPI chạy trong threadpool, BM25 (CPU) không chặn event loop.
- **Dual-stack socket**: tự tạo socket IPv6 với `IPV6_V6ONLY=0` để nghe cả `::1` lẫn `127.0.0.1` trên Windows.
- Có sẵn **Swagger UI** tại `/docs`.

### Hiệu năng (đo thật, 20 request)

| | Độ trễ |
|---|---|
| `db.search()` thuần (không HTTP) | **0.06 ms** |
| Qua HTTP (`localhost`) | ~37 ms |
| Qua HTTP + tái dùng connection | **~2 ms** |

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | Phải chạy bằng `python server.py`, KHÔNG dùng `uvicorn server:app` — fix dual-stack socket nằm trong `__main__` | Nếu deploy production cần wrap lại cấu hình socket |
| 2 | ~35ms latency do tạo kết nối TCP mới mỗi request | Khuyên Dev 3 dùng **connection pooling** (`httpx.Client()`) để xuống ~2ms |
| 3 | `metadata.json` commit lên git (24 frame, ~37KB) — khi chuyển dataset thật sẽ phình hàng trăm MB | Phải chặn trong `.gitignore` và đổi test sang fixture nhỏ |

### Lệnh chạy

```bash
# Khởi động server (mở http://localhost:8002/docs để test trên Swagger)
python src/semantic_pipeline/server.py

# Test nhanh bằng curl
curl -X POST http://localhost:8002/internal/search/text -H "Content-Type: application/json" -d "{\"keywords\": [\"booking\", \"entity\"], \"top_k\": 5}"

# Kiểm tra health
curl http://localhost:8002/health
```

---

## Task 5 — Unit Testing & Performance Testing

### Đã làm

- **37 test** trong `tests/` — lưới an toàn cho cả Task 1–3.
- `test_api.py` (17 test): endpoint `/internal/search/text` — đúng 4 field contract, HTTP 200, payload sai → 422, Swagger.
- `test_database.py` (13 test): tokenizer (stemming/stopword/bất quy tắc), xếp hạng BM25, khóa chính `frame_id`, đo tốc độ.
- `test_ocr_accuracy.py` (7 test): so sánh OCR với **ground truth gõ tay từ ảnh gốc**, kiểm tra tầng dịch.

### Kết quả đo thật

| Hạng mục | Kết quả |
|---|---|
| OCR tiếng Anh | **100%** |
| OCR tiếng Việt có dấu | **100%** |
| OCR ký hiệu toán | **100%** (trước ensemble: 92.3%) |
| BM25 query latency | **0.051 ms** (yêu cầu < 500ms) |
| Dựng index (24 doc) | 1.9 ms |

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | Ground truth chỉ 3 frame, chưa đủ đại diện | Thêm ground truth cho nhiều frame hơn, đa dạng loại nội dung |
| 2 | Test OCR accuracy cần metadata.json đã sinh sẵn — nếu thay đổi extractor phải chạy lại | Tách test OCR thành 2 loại: offline (metadata có sẵn) và integration (chạy extractor rồi đo) |
| 3 | Performance test chỉ đo trên 24 doc — chưa phản ánh dataset thật (hàng chục nghìn frame) | Thêm benchmark trên dataset lớn hơn |

### Lệnh chạy

```bash
# Chạy toàn bộ test (37 test)
pytest src/semantic_pipeline/tests/ -v

# Chạy riêng từng file test
pytest src/semantic_pipeline/tests/test_api.py -v          # 17 test API
pytest src/semantic_pipeline/tests/test_database.py -v     # 13 test BM25
pytest src/semantic_pipeline/tests/test_ocr_accuracy.py -v # 7 test OCR accuracy

# Chạy test với output chi tiết (hiện kết quả đo performance)
pytest src/semantic_pipeline/tests/ -v -s
```
