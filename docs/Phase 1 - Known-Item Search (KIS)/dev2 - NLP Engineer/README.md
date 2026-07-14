# Dev 2 — Semantic Pipeline (NLP Engineer)

Source: **`src/semantic_pipeline/`** · Branch: **`feat/semantic-pipeline`**

| Task | Nội dung | Trạng thái |
|---|---|---|
| **Task 1** | Trích xuất ngữ nghĩa (caption + OCR) → `metadata.json` | ✅ Xong |
| **Task 2** | Text DB bằng BM25 (`rank_bm25`) | ✅ Xong |
| **Task 3** | Internal API (FastAPI, port 8002) | ✅ Xong |
| Task 4 | R&D (Elasticsearch, Spatial Reasoning, Entity Extraction) | ⬜ |
| **Task 5** | Unit Testing & Performance Testing | ✅ Xong |

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

## ✅ Đã làm được (Task 3)

Internal API bằng **FastAPI** trong [`server.py`](../../../src/semantic_pipeline/server.py), chạy port **8002**.

**Chạy:** `python src/semantic_pipeline/server.py` → mở **http://localhost:8002/docs** (Swagger UI, test được ngay không cần code).

| Endpoint | Công dụng |
|---|---|
| `POST /internal/search/text` | Nhận `{keywords, top_k}` → trả `{status, data:[frame_id, score, video_name, frame_index]}` đúng API Contract |
| `GET /health` | Dev 3 kiểm tra service sống chưa + index nạp bao nhiêu doc |

- **Vì sao FastAPI chứ không Flask:** Dev 3 (người gọi API này) đã dùng FastAPI; env đã có sẵn `fastapi`+`uvicorn` (chưa có Flask); Pydantic tự validate và trả **422** khi payload sai; có sẵn **Swagger UI** để Dev 3 tự test.
- Index BM25 dựng **1 lần lúc khởi động** (qua `lifespan`), không dựng lại mỗi request.
- Phân biệt rõ 2 loại "rỗng": `keywords: ["zzzz"]` (từ khóa hợp lệ, không frame nào khớp) → **200 + `data: []`**; còn `keywords: []` (không gửi từ khóa) → **422** để Dev 3 biết là bug bên họ.

**Đã kiểm chứng:** 18/18 test PASS (contract, ca biên, hạ tầng).

### ⚠️ 3 điều BẮT BUỘC nhớ

1. **Phải chạy bằng `python server.py`**, KHÔNG dùng `uvicorn server:app`. Fix dual-stack socket nằm trong `__main__`; chạy bằng lệnh `uvicorn` sẽ mất fix và **độ trễ vọt lên 2 giây**.
2. **Dùng `def` chứ KHÔNG phải `async def`** cho endpoint. FastAPI chạy hàm `def` trong threadpool nên BM25 (nặng CPU) không chặn event loop. Đổi sang `async def` sẽ khiến mọi request xếp hàng chờ nhau.
3. `metadata.json` **được commit lên git** (24 frame mẫu chỉ ~37KB) để test chạy được ngay sau khi clone. ⚠️ Khi chuyển sang dataset thật của BTC (hàng chục nghìn frame), file này sẽ phình lên hàng trăm MB → **phải chặn lại trong `.gitignore`** và đổi test sang fixture nhỏ.

### 📊 Hiệu năng (đo thật, 20 request)

| | Độ trễ |
|---|---|
| `db.search()` thuần (không HTTP) | **0.06 ms** |
| Qua HTTP (`localhost`) | ~37 ms |
| Qua HTTP + tái dùng connection | **~2 ms** |

Yêu cầu task là < 500ms → **đạt thoải mái**. Lưu ý: ~35ms kia gần như toàn bộ là chi phí **tạo kết nối TCP mới mỗi request** của client, không phải xử lý của server. Nên khuyên Dev 3 dùng **connection pooling** (`httpx.Client()` tái sử dụng) để xuống ~2ms.

---

## ✅ Đã làm được (Task 5)

**37 test** trong [`tests/`](../../../src/semantic_pipeline/tests/) — lưới an toàn cho cả Task 1–3.

**Chạy:** `pytest src/semantic_pipeline/tests/ -v`

| File | Nội dung |
|---|---|
| `test_api.py` (17) | Endpoint `/internal/search/text`: đúng 4 field contract, HTTP 200, payload sai → 422, Swagger |
| `test_database.py` (13) | BM25: tokenizer (stemming/stopword/bất quy tắc), xếp hạng, khóa chính `frame_id`, **+ đo tốc độ** |
| `test_ocr_accuracy.py` (7) | Độ chính xác OCR so với **ground truth gõ tay từ ảnh gốc**, + kiểm tra tầng dịch |

### 📊 Kết quả đo thật

| Hạng mục | Kết quả |
|---|---|
| OCR tiếng Anh | **100%** |
| OCR tiếng Việt **có dấu** | **100%** |
| OCR ký hiệu toán | **100%** (trước ensemble: 92.3%) |
| BM25 query | **0.051 ms** (yêu cầu < 500ms) |
| Dựng index (24 doc) | 1.9 ms |

> Con số **100% tiếng Việt có dấu** chứng minh dứt khoát quyết định thay Florence-2 (vốn đọc `"Nếu giữ"` thành `"Neu giüt"`) bằng VietOCR là đúng.

### 🔀 Ensemble OCR (phát hiện & sửa nhờ Task 5)

Test lộ ra một lỗi chưa từng biết: VietOCR **đọc sai ký hiệu toán** — `1 + 2 + 3 + 4 = 10` thành `1%2%344-10` (dấu `+` → `%`, `=` → `-`).

Nguyên nhân **không phải** thiếu vocab (VietOCR có đủ `+`, `=`), mà là nhận diện sai về thị giác vì nó train chủ yếu trên văn bản tiếng Việt.

Cách sửa: PaddleOCR chạy để detect vùng chữ **cũng tự đọc text luôn** — mà code cũ **vứt bỏ**. Giờ dùng cả hai, chọn theo thế mạnh:

| Loại dòng | Model dùng | Vì sao |
|---|---|---|
| CÓ dấu tiếng Việt | **VietOCR** | Đọc dấu 100%; PaddleOCR rớt dấu (`"Nếu giữ"` → `"Nu gi"`) |
| KHÔNG dấu (Anh/toán) | **PaddleOCR** | Đọc ký hiệu đúng; tiếng Anh cũng tốt ngang VietOCR |

> **Không tốn thêm model, không chậm thêm** — cả hai vốn đã chạy sẵn.

---

## ⚠️ Điểm cần cải thiện

### Task 1

| # | Vấn đề | Hướng cải thiện |
|---|---|---|
| 1 | Nhận diện tiếng Việt **chỉ dựa vào dấu** → chữ Việt không dấu (`"khong tach"`) không được dịch | Dùng language detection thật (`fasttext lid.176`) thay regex. Cẩn thận: nới lỏng quá dễ dịch nhầm từ tiếng Anh |
| 2 | VietOCR còn sạn: ký tự `—`, `/` thành `?`; vài lỗi dấu lẻ | ✅ **Đã sửa phần lớn** bằng ensemble với PaddleOCR (xem Task 5) + crop padding. Còn có thể: lọc theo confidence (`return_prob=True`); bật `beamsearch=True` (chuẩn hơn nhưng chậm) |
| 3 | Caption Florence-2 tự "đọc" chữ và đọc sai tiếng Việt | Ưu tiên thấp (frame thật ít chữ). Nếu cần: bỏ phần trong ngoặc kép của caption |
| 4 | **Chậm trên CPU** (chạy 4 model tuần tự) | Giảm `num_beams` của Florence-2 (3 → 1); chạy GPU (code đã auto-detect `cuda`, riêng Paddle cần `paddlepaddle-gpu`); thêm cơ chế resume để chạy lại không mất công |

### Task 2

| # | Vấn đề | Trạng thái / Hướng cải thiện |
|---|---|---|
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
| **API chậm 2 GIÂY dù BM25 chỉ tốn 0.06ms** | Windows phân giải `localhost` thành `::1` (IPv6) **trước**. Server bind `0.0.0.0` (chỉ IPv4) → client phải chờ IPv6 timeout ~2s rồi mới fallback. ⚠️ Bind `host="::"` **KHÔNG sửa được** — trên Windows uvicorn tạo socket **IPv6-only**, làm `127.0.0.1` chết luôn. Phải **tự tạo socket** rồi tắt cờ `IPV6_V6ONLY` mới nghe được cả hai |
| `ModuleNotFoundError: No module named 'database'` khi viết test | `server.py` import `from database import ...` chỉ chạy khi gọi trực tiếp. Đã bọc `try/except ImportError` để fallback sang relative import (`from .database import ...`) cho pytest |

---

## 🧭 Vì sao chọn các model này

- **Bỏ `<OCR>` của Florence-2**: đọc tiếng Việt sai nặng (`"Nếu giữ"` → `"Neu giüt"`).
- **PaddleOCR chỉ để detect**, không đọc chữ: PP-OCRv6 rớt dấu kép (`"Nếu giữ"` → `"Nu gi"`).
- **VietOCR để đọc chữ**: train chuyên tiếng Việt, đọc đúng dấu.
- **Giữ Helsinki-NLP**: đã kiểm chứng — cho ăn tiếng Việt chuẩn thì dịch rất tốt. Bản dịch rác ban đầu là do **OCR sai**, không phải lỗi translator.
