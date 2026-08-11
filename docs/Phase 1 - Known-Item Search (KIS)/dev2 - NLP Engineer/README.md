# Dev 2 — Semantic Pipeline (NLP Engineer)

Source: **`src/semantic_pipeline/`**

| Task | Nội dung | Trạng thái |
|---|---|---|
| **Task 1** | Trích xuất ngữ nghĩa (caption + OCR) → `metadata.json` | ✅ Xong |
| **Task 2** | Text DB bằng BM25 (`rank_bm25`) | ✅ Xong |
| **Task 3** | Internal API (FastAPI, port 8002) | ✅ Xong |
| **Task 4** | R&D (Elasticsearch, SQL Classification, Spatial Reasoning, Entity Extraction) | ✅ Hoàn tất R&D; chưa production-ready ([báo cáo](TASK4_REPORT.md)) |
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

Các thiếu sót riêng của `extractor.py` ở Task 1 cần nhớ:

- Chưa có resume/checkpoint; chạy lại sẽ xử lý lại toàn bộ và ghi đè output. Entity và spatial ở Task 4 đã có checkpoint/resume.
- Tên file bắt buộc theo mẫu `video_f0001`; filename sai sẽ lỗi.
- Nhận diện ngôn ngữ chỉ dựa trên dấu tiếng Việt.
- Chưa có thống kê lỗi hoặc danh sách frame thất bại riêng.
- Chưa benchmark trên dữ liệu thực tế lớn.

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
- Index `caption` + `ocr_text` (tiếng Anh) cùng semantic `code.search_terms` được
  suy ra từ cả translated/raw OCR. Không index trực tiếp toàn bộ `ocr_text_raw`.
- Trả điểm BM25 **thô, không chuẩn hóa**. Lọc bỏ `score <= 0`.
- `float(score)` — ép kiểu vì `numpy.float64` không JSON-serialize được.
- Output đúng API Contract: `frame_id`, `score`, `video_name`, `frame_index`.

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | Backend BM25 dựng lại index khi khởi động và giữ corpus trên RAM | Đã có backend **Elasticsearch v5** ở Task 4; giữ BM25 làm baseline/rollback nhẹ |
| 2 | `caption` và `ocr_text` bị gộp, không đặt trọng số riêng được | Elasticsearch đã tách field và boost `ocr_text` cao hơn `caption`; vẫn cần tập relevance lớn hơn để tuning |
| 3 | Benchmark hiện chỉ có 24 document | Bổ sung benchmark/load test theo nhiều mức dữ liệu trước khi dùng với kho video lớn |

### Lệnh chạy

```bash
# Chạy demo search (query mẫu trên 24 frame)
python src/semantic_pipeline/database.py
```

---

## Task 3 — Internal API (FastAPI)

### Đã làm

- API server bằng **FastAPI** trong `server.py`, port **8002**.
- Endpoint `POST /internal/search/text`: nhận `{keywords, top_k, filters?}` → trả `{status, data: [{frame_id, score, video_name, frame_index}]}`.
- Endpoint `GET /health`: kiểm tra service sống chưa, index nạp bao nhiêu doc.
- Chọn backend bằng `SEMANTIC_SEARCH_BACKEND`: mặc định BM25; Elasticsearch là bản nâng cấp tùy chọn và giữ nguyên response contract.
- Với Elasticsearch, API hỗ trợ entity filter và spatial triple; mọi list filter dùng semantics **all-of (AND)**.
- Index/backend được khởi tạo **1 lần lúc startup** (`lifespan`), không dựng lại mỗi request.
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
| 3 | Fixture `sample_frames/metadata.json` 24 frame được commit có chủ đích; metadata full của dataset thật sẽ rất lớn | Chỉ giữ fixture nhỏ trong Git; generated metadata/report lớn đã được `.gitignore` và cần kho artifact dùng chung |

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

## Task 4 — Elasticsearch, SQL Classification, Entity Extraction & Spatial Reasoning

### Đã làm

#### 1. Elasticsearch v5

- Chạy Elasticsearch `9.4.2` bằng Docker Compose, chỉ bind local tại `127.0.0.1:9200`.
- Mapping `dynamic: strict`, English analyzer có lowercase/stopword/stemming; `ocr_text` được boost cao hơn `caption`.
- Bulk ingest idempotent bằng `frame_id` làm Elasticsearch `_id`.
- Dùng physical index version hóa (`semantic_frames_v5`) và alias ổn định (`semantic_frames`) để chuyển phiên bản atomically.
- CLI có thay đổi dữ liệu bắt buộc truyền `--index-name`, tránh vô tình trỏ alias ngược về index cũ.
- Search hỗ trợ `time_of_day`, `setting`, `locations`, `objects`, `actions`, `colors` và nested `spatial_relations`.
- Benchmark hiện có 17 query; 3 query filter được thực sự truyền xuống backend và đều đưa frame đúng lên rank 1.

Kết quả Elasticsearch v5 trên 24 frame mẫu:

| Metric | Kết quả |
|---|---:|
| Recall@5 | `1.0000` |
| MRR@5 | `0.9412` |
| NDCG@5 | `0.9566` |
| Local p95 (lần chạy cuối, 340 mẫu) | `7.474 ms` — đạt gate `< 50 ms` |
| Filter cases đúng ở rank 1 | `3/3` |

#### 2. Deterministic SQL Classification

- `code_classifier.py` phân loại caption/OCR thành `code.language`, `statement_type`, `patterns`, `search_terms`, `evidence`, `classifier_version`; không cần chạy lại OCR hay gọi Ollama.
- Không dùng `frame_id` trong inference. SQL `SELECT` phải có `FROM` cùng tín hiệu cú pháp bổ sung; template `SELECT ... FROM` được gắn search term riêng.
- Artifact hiện có classifier provenance đủ `24/24`: `7` frame SQL và `17` frame unknown.
- Cả BM25 và Elasticsearch v5 đều trả `vid03_f0004` ở rank 1 cho query không filter `sql query`; filter `code_language=sql` cùng `code_patterns=[not exists, correlated]` cũng trả đúng duy nhất frame này.
- Hướng dẫn và lệnh test: [`CODE_CLASSIFICATION.md`](../../../src/semantic_pipeline/CODE_CLASSIFICATION.md).

#### 3. Entity Extraction

- `entity_extractor.py` gọi Ollama `llama3.2:3b`, yêu cầu structured JSON và validate lại bằng metadata schema thống nhất.
- Prompt `entities-v1.3` có few-shot và evidence guard để loại location/color/relation không được caption hoặc OCR hỗ trợ.
- Có checkpoint/resume; không ghi đè metadata Task 1.
- Đã xử lý đủ `24/24` frame vào `sample_frames/metadata_entities.json` và ghi provenance model/prompt.
- API/Elasticsearch dùng entity metadata làm exact filters; list filter đã có cả inclusion và exclusion test cho semantics AND.
- Đã thêm visual ground truth độc lập cho `12` frame và `benchmark_entities.py`; truth được annotate từ pixel, không copy caption/model output.

Kết quả entity visual benchmark:

| Metric | Kết quả |
|---|---:|
| All-field micro-F1 | `0.5814` |
| Searchable-entity micro-F1 | `0.4194` |
| Object precision / recall / F1 | `0.3396 / 0.5455 / 0.4186` |
| Action F1 | `0.1667` |
| Color F1 | `0.5000` |

Benchmark đã khắc phục việc thiếu visual truth, đồng thời cho thấy entity quality vẫn chưa đạt production gate `F1 >= 0.70`.

#### 4. Spatial Reasoning

- `spatial_extractor.py` dùng Florence-2 `<OD>`, chuẩn hóa bbox về `[x1, y1, x2, y2]` trong khoảng `[0,1]` và tạo object ID ổn định.
- Suy luận bảo thủ các quan hệ `left_of`, `right_of`, `above`, `below`, `overlapping`; lưu đầy đủ object ID và label trong nested triple.
- Có input check, chạy chọn frame, checkpoint/resume, schema validation và full run `24/24` frame.
- Contextual label grounding sửa các confusion có evidence như `dice → domino`, `vase → sack`; không dùng `frame_id`/gold khi inference và lưu `raw_label`, nguồn/evidence cùng version provenance.
- Full sample hiện có `52` detection và `216` raw instance relations để evaluation.
- Khi index, các relation tương đương theo `(subject_label, predicate, object_label)` được collapse còn `9`; API không phải lưu/query 184 relation hoa trùng semantics.
- Spatial gold set độc lập đã mở rộng lên `10` frame, `35` object và `32` directional relation; 3 frame được annotate relation exhaustive.

Kết quả spatial pilot:

| Metric | Kết quả | Diễn giải |
|---|---:|---|
| Class-agnostic localization precision / recall / F1 | `0.9211 / 1.0000 / 0.9589` | 35/35 object gold được định vị |
| Relation geometry F1 | `1.0000` | Đúng trên 32 relation đã annotate |
| Object label accuracy sau grounding | `1.0000` | Trước grounding trên gold mở rộng: `0.5714` |
| Label-aware/queryable relation recall | `1.0000` | Trước grounding: `0.0000` |

#### 5. Acceptance và API end-to-end

- `task4_acceptance.py` tổng hợp schema, entity visual benchmark, spatial benchmark, text regression, filter benchmark, scale pilot và trạng thái Elasticsearch live.
- Test opt-in đi xuyên suốt FastAPI → Elasticsearch, gồm entity filter, spatial filter và case loại trừ cho list AND.
- Scale harness v2 đã index `10.000` document synthetic với mapping code mới trên physical index riêng: khoảng `1,558.9 docs/s`, p95 `47.915 ms`, không thay đổi alias production.
- Có profile Docker riêng dùng authentication + HTTPS/transport TLS tại `docker-compose.elasticsearch.secure.yml`; secret nằm ngoài repository, port/data/volume tách khỏi local stack.
- Trạng thái nghiệm thu hiện tại: `research_complete=True`, `production_ready=False`.
- Báo cáo chi tiết: [TASK4_REPORT.md](TASK4_REPORT.md); contract dùng chung: [API_CONTRACT.md](../../API_CONTRACT.md).

### Artifact cần giữ và file có thể tái tạo

Giữ các input/output dùng cho kiểm thử hiện tại:

- `sample_frames/metadata.json`: metadata gốc Task 1.
- `sample_frames/metadata_entities.json`: full output entity `24/24`.
- `sample_frames/metadata_spatial.json`: full output spatial `24/24`.
- `baseline_report_bm25_v0.json`: baseline để kiểm tra regression.
- `spatial_ground_truth.json`, `entity_visual_ground_truth.json`: ground truth cần commit để tái lập benchmark.
- `baseline_report_elasticsearch_v5_filters.json`, `spatial_benchmark_report_v1.json`, `entity_benchmark_report_v1.json`, `scale_benchmark_report_10k_v2.json`, `task4_acceptance_report.json`: các report mới nhất.

Các output chạy thử như `metadata_entities_test.json`, `metadata_spatial_test.json`, `metadata_v1.json` và report Elasticsearch v1/v2/v3 không-filter đã được xóa sau khi xác nhận có bản đầy đủ/mới hơn. Chúng đều có thể sinh lại bằng các lệnh trong `ENTITY_EXTRACTION.md`, `SPATIAL_REASONING.md` và phần test bên dưới. Metadata/report sinh ra được `.gitignore`; khi clone máy mới cần chạy lại pipeline hoặc nhận artifact từ kho dữ liệu chung của nhóm.

### Cách kiểm tra Task 4

Chạy từ thư mục gốc repository trong PowerShell, sau khi `conda activate aichcm2026`:

```powershell
# 0. Cài đúng dependency vào Conda environment hiện tại (chỉ cần khi setup/cập nhật)
python -m pip install -r requirements.txt

# 1. Khởi động Elasticsearch; đợi "health: starting" chuyển sang "healthy"
docker compose -f src/semantic_pipeline/docker-compose.elasticsearch.yml up -d
docker compose -f src/semantic_pipeline/docker-compose.elasticsearch.yml ps
python src/semantic_pipeline/elasticsearch_backend.py health

# Phân loại SQL từ OCR đã có; không tải lại OCR/Ollama
python src/semantic_pipeline/code_classifier.py `
  --input src/semantic_pipeline/sample_frames/metadata_entities.json `
  --in-place
python src/semantic_pipeline/code_classifier.py `
  --input src/semantic_pipeline/sample_frames/metadata_spatial.json `
  --in-place

# Nếu là máy mới/chưa có v5: validate, ingest rồi atomically kích hoạt alias
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v5 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json

# 2. Validate hai metadata đầy đủ
python src/semantic_pipeline/schemas.py `
  --metadata src/semantic_pipeline/sample_frames/metadata_entities.json
python src/semantic_pipeline/schemas.py `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json

# 3. Sinh lại các benchmark quality
python src/semantic_pipeline/benchmark_entities.py
python src/semantic_pipeline/benchmark_spatial.py
python src/semantic_pipeline/benchmark_elasticsearch.py `
  --index semantic_frames `
  --output src/semantic_pipeline/baseline_report_elasticsearch_v5_filters.json

# 4. Scale pilot lần đầu trên physical index riêng
python src/semantic_pipeline/benchmark_scale.py `
  --index-name semantic_frames_scale_10k_v2 `
  --documents 10000 `
  --warmup-runs 3 `
  --latency-runs 20 `
  --output src/semantic_pipeline/scale_benchmark_report_10k_v2.json

# Khi index scale đã tồn tại, đo lại mà không ingest
python src/semantic_pipeline/benchmark_scale.py `
  --index-name semantic_frames_scale_10k_v2 `
  --documents 10000 `
  --skip-ingest `
  --output src/semantic_pipeline/scale_benchmark_report_10k_v2.json

# 5. Nghiệm thu tổng hợp, gồm kiểm tra live index/alias
python src/semantic_pipeline/task4_acceptance.py `
  --check-live-elasticsearch

# 6. Toàn bộ unit/offline test
python -m pytest -q src/semantic_pipeline/tests

# 7. Riêng integration FastAPI -> Elasticsearch (Elasticsearch phải đang healthy)
$env:RUN_TASK4_LIVE = "1"
python -m pytest -q src/semantic_pipeline/tests/test_task4_live.py
Remove-Item Env:RUN_TASK4_LIVE

# 8. Live test trực tiếp Elasticsearch backend
$env:RUN_ELASTICSEARCH_INTEGRATION = "1"
python -m pytest -q src/semantic_pipeline/tests/test_elasticsearch_backend.py
Remove-Item Env:RUN_ELASTICSEARCH_INTEGRATION
```

Kỳ vọng: health báo alias có `24` document và live acceptance xác nhận alias trỏ `semantic_frames_v5`; hai schema hợp lệ `24/24`; classifier báo `7` SQL/`17` unknown; acceptance báo `research_complete=True`, `production_ready=False`. Cấu hình auth/TLS xem `src/semantic_pipeline/ELASTICSEARCH_SECURE.md`.

### Hạn chế còn lại & Ưu tiên cải thiện

| Ưu tiên | Hạn chế đã kiểm chứng | Việc nên làm tiếp |
|---|---|---|
| P0 | Searchable-entity visual F1 chỉ `0.4194`; object/action có nhiều FP từ text/layout | Grounded visual entity extraction hoặc detector/phrase grounding; bổ sung annotator thứ hai và holdout video |
| P0 | Grounding đưa spatial pilot lên `1.0` nhưng rule được rút từ sample hiện tại | Đánh giá trên holdout target video chưa dùng để thiết kế rule; nếu giảm mạnh phải thay detector thay vì thêm rule theo frame |
| P0 | Chỉ `3/10` spatial gold frame có relation annotation exhaustive | Annotate tối thiểu 5 relation-rich frame, đủ positive/negative cho từng predicate và repeated labels |
| P1 | Code classifier v1 mới hỗ trợ SQL và mới benchmark trên 24 frame | Thêm gold/holdout cho Python, JavaScript, DDL và OCR lỗi; đo precision/recall theo language/pattern |
| P1 | Raw metadata vẫn có `216` instance relation, dù indexed projection đã giảm còn `9` | Giữ raw cho evaluation; theo dõi indexed count/macro-by-scene và stress test scene dày đặc |
| P1 | Confidence detection `0.5` chỉ là proxy, không được calibration | Dùng detector trả score thật hoặc giữ confidence ra khỏi acceptance gate |
| P1 | Scale 10k là synthetic replication, chưa đại diện relevance/dung lượng thật | Chạy tiếp 100k/1M với corpus thật, concurrency, shard sizing, disk/RAM và p99 |
| P1 | Secure profile vẫn single-node | Staging/production cần HA, snapshot/restore drill, monitoring/audit, cert rotation, managed PKI/firewall/secret manager |

Không nên xem `52 detections` hoặc `216 relations` là accuracy. Gate quan trọng tiếp theo là chất lượng label-aware trên gold set lớn hơn, không phải chỉ tăng số object/relation được sinh ra.

---

## Task 5 — Unit Testing & Performance Testing

### Đã làm

- **166 test pass, 2 test skip** trong `tests/` ở lần chạy nghiệm thu gần nhất; các test skip là integration cần service live/biến môi trường.
- Task 1–3: API contract, payload validation, BM25/tokenizer/ranking, baseline evaluator và OCR ground truth.
- Metadata/Task 4: schema, Elasticsearch query/mapping/alias/bulk ingest, entity evidence guard/checkpoint, spatial geometry/checkpoint, spatial benchmark và acceptance aggregator.
- `test_task4_live.py`: integration opt-in đi từ FastAPI tới Elasticsearch thật, không chạy mặc định để unit suite vẫn dùng được khi Docker tắt.

### Kết quả đo thật

| Hạng mục | Kết quả |
|---|---|
| OCR tiếng Anh | **100%** |
| OCR tiếng Việt có dấu | **100%** |
| OCR ký hiệu toán | **100%** (trước ensemble: 92.3%) |
| BM25 query latency | **0.051 ms** (yêu cầu < 500ms) |
| Dựng index (24 doc) | 1.9 ms |
| Elasticsearch v5 Recall@5 / MRR@5 / NDCG@5 | **1.0000 / 0.9412 / 0.9566** |
| Elasticsearch v5 local p95 (lần chạy cuối, 340 mẫu) | **7.474 ms** — đạt gate `< 50 ms` |
| Elasticsearch scale 10k v2 throughput / p95 | **1,558.9 docs/s / 47.915 ms** |

### Hạn chế & Hướng cải thiện

| # | Hạn chế | Hướng cải thiện |
|---|---|---|
| 1 | OCR/entity/spatial ground truth vẫn nhỏ và chỉ có một annotator | Mở rộng bằng target video, tách theo scene/video và đo inter-annotator agreement |
| 2 | Test OCR accuracy cần metadata.json đã sinh sẵn — nếu thay đổi extractor phải chạy lại | Tách test OCR thành 2 loại: offline (metadata có sẵn) và integration (chạy extractor rồi đo) |
| 3 | Đã có scale 10k nhưng là synthetic/single-client | Thêm corpus thật 100k/1M và nhiều client đồng thời |
| 4 | Hai live integration test bị skip khi chưa bật biến môi trường/service | Chạy riêng live tests trong CI job có Elasticsearch; giữ offline suite nhanh cho mọi developer |

### Lệnh chạy

```powershell
# Chạy toàn bộ unit/offline test
python -m pytest src/semantic_pipeline/tests -v

# Chạy riêng từng file test
python -m pytest src/semantic_pipeline/tests/test_api.py -v
python -m pytest src/semantic_pipeline/tests/test_database.py -v
python -m pytest src/semantic_pipeline/tests/test_ocr_accuracy.py -v

# Chạy test với output chi tiết (hiện kết quả đo performance)
python -m pytest src/semantic_pipeline/tests -v -s
```

### Video-understanding pilot (L22_V001)

Pilot hướng tin tức được triển khai riêng tại
[`VIDEO_UNDERSTANDING_PILOT.md`](VIDEO_UNDERSTANDING_PILOT.md). Pipeline chỉ đọc
caption Gemini, ASR và `map-keyframes`; output cố định gồm đúng ba file
`timeline.json`, `video_summary.json` và `validation_report.json`.

Gemini dùng structured JSON schema cho từng evidence window và cho summary cuối
video. Story title/summary được yêu cầu bằng tiếng Việt; `summary_vi` và
`summary_en` là hai trường khác ngôn ngữ. Các card trùng do window overlap được
merge trước khi publish, còn đoạn mở đầu chương trình không được coi là story.

Model hiện dùng cho summary là `gemini-3.1-flash-lite`. Free-tier được giới hạn
tự động bằng `GEMINI_REQUEST_INTERVAL_SECONDS` (mặc định `4.2`) và retry bounded
khi gặp HTTP 429. Chạy pilot thật:

```powershell
python src/semantic_pipeline/video_understanding/cli.py `
  --video-id L22_V001 `
  --llm `
  --require-llm
```

Video retrieval theo ba tầng `video → segment → frame`. `search_text` của video
gồm cả title và summary của mọi segment, còn keyframe cuối được xếp hạng bằng
caption/object/OCR/ASR trong segment đã chọn. Chạy kiểm tra cục bộ:

```powershell
python src/semantic_pipeline/retrieval/video_cli.py "nhiệt độ Barcelona cao nhất trong 110 năm"
```
