# Kế hoạch tích hợp dữ liệu gốc BTC vào Semantic Pipeline (Dev 2)

Nhánh triển khai: `feat/semantic-pipeline-feature-btc-object-integration`

Mục tiêu: chuyển pipeline từ fixture 24 frame tự đặt tên sang **định danh và
dữ liệu gốc của BTC**, lấy `objects/` của BTC làm nguồn detection chính thay cho
Florence-2 `<OD>`.

---

## 0. Số liệu thực đo trên `data/raw/` (căn cứ mọi quyết định bên dưới)

| Hạng mục | Giá trị |
|---|---|
| Video có `objects/` | **873** (177.321 file JSON) |
| Video có ảnh keyframe | **60** (L21: 29, L22: 31) — 16.896 ảnh |
| Lệch `n_keyframes` vs `n_objects` | **0/60** → join theo số thứ tự an toàn |
| Detection mỗi frame | **100 cố định** (đã pad) |
| Detection `score >= 0.5` | ~4.3 / frame |
| Detection `score >= 0.3` | ~7.3 / frame |
| Detection `score >= 0.2` | ~10.9 / frame |
| Detection `score >= 0.1` | ~18.8 / frame |
| Taxonomy | Open Images V4 — 600 class, MID `/m/079cl` |
| Format box | `[ymin, xmin, ymax, xmax]`, chuẩn hoá `[0,1]`, **kiểu string** |
| `media-info` | `title`, `description`, `keywords`, `length`, `publish_date`, `watch_url` — **tiếng Việt**, chưa được index |
| `map-keyframes` | `n, pts_time, fps, frame_idx` |

Xác nhận thứ tự box: `Skyscraper` có box `[0.409, 0.031, 0.626, 0.095]` →
cao 0.217, rộng 0.065. Chỉ đúng nếu đọc theo `[ymin, xmin, ymax, xmax]`.

**Hệ quả lớn nhất:** 813/873 video **không có ảnh** → Florence-2 caption/OCR
không chạy được. `objects/` + `media-info/` là dữ liệu duy nhất phủ 100% kho.
Pipeline phải tìm được frame ngay cả khi không có caption.

---

## 1. Chốt định danh (BTC native)

| Field | Giá trị | Nguồn |
|---|---|---|
| `video_name` | `L21_V001` (không thêm `.mp4`) | tên thư mục BTC |
| `frame_index` | `frame_idx` **gốc của BTC** (vd `261`) | `map-keyframes/L21_V001.csv` |
| `keyframe_n` | `1` (file `001.json` / `001.jpg`) | tên file BTC |
| `timestamp_ms` | `round(pts_time * 1000)` | `map-keyframes` |
| `frame_id` | `L21_V001_f0261` | ghép |

Lý do chọn `frame_idx` chứ không phải `n`: dòng nộp bài AIC là
`video_id, frame_idx` → cả hai đọc thẳng từ field, không phải tính lại.

`frame_id` giữ nguyên **hình dạng** mà API Contract v1 quy định
(`<video><frame_index>`), chỉ đổi giá trị → Dev 3/Dev 4 không phải sửa logic
parse, nhưng **vẫn phải báo trước** vì mọi ID cũ hết hiệu lực.

`keyframe_n` là bắt buộc, không suy ra được từ `frame_idx`: nó là khoá để mở
`objects/L21_V001/001.json` và ảnh thumbnail cho Dev 4.

---

## STEP 0 — Lớp định danh dùng chung

**File:** `src/common/frame_ref.py` (đặt ở `common` vì Dev 1/3/4 đều cần)
**Công nghệ:** `csv` chuẩn, `functools.lru_cache`, `dataclass(frozen=True)`

Nội dung:

- `@dataclass(frozen=True) class FrameRef: video_id, keyframe_n, frame_idx, pts_time, fps`
- `load_keyframe_map(video_id) -> dict[int, FrameRef]` — đọc `map-keyframes/{video_id}.csv`,
  cache bằng `lru_cache(maxsize=64)`.
- `frame_id(ref) -> "L21_V001_f0261"` và `parse_frame_id(s) -> (video_id, frame_idx)`.
- `resolve(video_id, keyframe_n) -> FrameRef` và `resolve_by_frame_idx(...)`.
- `submission_row(ref) -> (video_id, frame_idx)`.

Cạm bẫy:

- `map-keyframes` mở bằng `encoding="utf-8-sig"` (có BOM — `build_manifest.py`
  đã xử lý đúng, làm theo).
- `fps` là float (`30.0`, có video `25.0`) → đừng hardcode.
- Video thiếu `map-keyframes` phải raise lỗi rõ ràng, không fallback thầm lặng.
- `pts_time` là nguồn chuẩn cho timestamp, **không** tự tính `frame_idx / fps`
  (có video BTC làm tròn lệch).

**Nghiệm thu:** test round-trip `frame_id → parse → resolve` trên toàn bộ 873
video; assert `n` liên tục 1..N và `frame_idx` không giảm. Dữ liệu thực tế có 192
map lặp `frame_idx`, vì vậy giữ `n` đầu tiên làm đại diện canonical cho mỗi dòng nộp.

---

## STEP 1 — Adapter `objects/` của BTC

**File:** `src/semantic_pipeline/btc_objects.py`
**Công nghệ:** `json` chuẩn + `Detection` schema có sẵn + NMS tự viết (numpy không cần thiết ở mức 100 box)

### 1.1 Chuyển đổi format

```
BTC:  detection_boxes[i] = ["0.4094", "0.0307", "0.6258", "0.0950"]  # ymin,xmin,ymax,xmax
Ta:   bbox = (x1, y1, x2, y2) = (float(b[1]), float(b[0]), float(b[3]), float(b[2]))
```

Giữ thêm `mid = detection_class_names[i]` (`/m/079cl`) — đây là khoá bền để join
ontology, tên chữ có thể đổi giữa các bản Open Images.

### 1.2 Lọc nhiễu (bắt buộc — 100 box/frame, đuôi xuống tới 0.007)

Áp theo thứ tự:

1. `score >= 0.20` — theo số đo, giữ ~10.9 box/frame. Đặt thành hằng
   `MIN_DETECTION_SCORE` để tune được.
2. `min_box_area >= 0.0001` (đã có sẵn trong `normalise_detections`).
3. **Class-wise NMS, IoU 0.60** — dùng lại `bbox_iou()` ở
   `spatial_extractor.py:208`, không viết lại. Sort giảm dần theo score, giữ box
   đầu, loại box cùng label IoU >= 0.6.
4. Cắt `top 15` sau khi đã sort theo score.

> **Cạm bẫy quan trọng:** `normalise_detections()`
> (`spatial_extractor.py:232`) sort candidate theo **alphabet rồi mới cắt**
> `max_detections`. Nếu đưa thẳng 100 box vào, bạn sẽ mất box điểm cao có label
> vần cuối bảng chữ cái. **Phải lọc + NMS trước, rồi mới gọi hàm này.**

### 1.3 Tái dùng `normalise_detections`

Box BTC đã chuẩn hoá sẵn → gọi với `image_width=1, image_height=1` (hàm chia cho
kích thước, nên truyền 1 là no-op). Không cần mở ảnh → chạy được cho cả 813 video
không có keyframe. Đặt `default_confidence` không dùng tới vì `RawDetection.confidence`
đã có score thật.

### 1.4 Field dẫn xuất tính luôn ở bước này

- `area = (x2-x1) * (y2-y1)`
- `grid_cell`: lưới 3×3 theo tâm box → `top-left … center … bottom-right`
- `object_counts: dict[label, int]` (sau NMS)
- `label_source = "btc_detector"` (thêm literal mới vào `Detection` schema)

### 1.5 Gộp 177k file JSON nhỏ → parquet

**File:** `scripts/build_objects_index.py`
**Công nghệ:** `pyarrow` (**thêm vào `requirements.txt`**, chú ý pin `numpy<2.0.0` đang có)

- Ghi `data/processed/objects_index/{video_id}.parquet` — hàm
  `paths.objects_index()` đã dựng sẵn chỗ.
- Lý do: 177k file nhỏ trên NTFS đọc rất chậm; mỗi lần thử nghiệm lại quét lại là không chịu nổi.
- Có `--resume` (bỏ qua video đã có parquet) và `--videos L21_V001,...`.

**Nghiệm thu:** chạy full 873 video; in phân bố detection/frame trước–sau lọc;
unit test: đảo trục box, NMS (case 2 box chồng cùng label / khác label), threshold,
frame 0 detection sau lọc không được crash.

---

## STEP 2 — Cầu nối từ vựng (mắt xích quyết định)

**File:** `src/semantic_pipeline/object_ontology.py`
**Dữ liệu:** `data/external/openimages/bbox_labels_600_hierarchy.json` + `class-descriptions-boxable.csv`
(tải từ `storage.googleapis.com/openimages/2018_04/`)
**Lexicon tự viết:** `src/semantic_pipeline/object_lexicon.json` — **commit lên git**

Đây là chỗ dữ liệu BTC thực sự "gắn" được với query của Dev 3.

### 2.1 Hierarchy Open Images

Cây có sẵn quan hệ cha–con theo MID. Viết `ancestors(mid) -> list[mid]`.
Ví dụ `Car → Land vehicle → Vehicle`.

- **Index-side expansion**: frame chứa `Car` thì index thêm `land vehicle`,
  `vehicle` vào `object_labels`. Query rộng ("có xe") hit được frame gắn nhãn hẹp.
- **Không** expand ngược (query hẹp không được match frame rộng) — sẽ tạo false positive.

### 2.2 Lexicon EN/VI thủ công

Cấu trúc mỗi entry:

```json
{"/m/01g317": {"label": "Person", "en": ["person","people","human"], "vi": ["người","đàn ông","phụ nữ"]}}
```

Ưu tiên ~120 label xuất hiện nhiều nhất trong 873 video (đếm bằng STEP 1 rồi mới
viết lexicon — đừng viết mò cả 600 class).

Nhóm phải xử lý riêng vì Open Images tách rất vụn:
`Man / Woman / Boy / Girl / Person / Human body / Human face / Human head`
→ thêm alias chung `person`, **giữ nguyên label gốc** để lọc giới tính vẫn được.

### 2.3 Hai chiều

- `expand_for_index(mid) -> set[str]` — dùng ở STEP 3.
- `map_query_terms(terms) -> (mids, unmapped)` — dùng ở STEP 5, khi Dev 3 gửi
  `filters.objects`. Trả luôn `unmapped` để biết query nào rơi ngoài 600 class.

**Nghiệm thu:** gold set ~50 cặp `(từ khoá query → label kỳ vọng)`, đo
precision/recall của mapping. Xuất báo cáo coverage: bao nhiêu % label thực tế
xuất hiện đã có lexicon.

---

## STEP 3 — Fusion vào metadata (schema v1.1)

**File:** sửa `schemas.py`, thêm `src/semantic_pipeline/metadata_builder.py`

### 3.1 Schema v1.1 — mọi field mới đều optional

Để metadata Task 1/Task 4 cũ vẫn validate được (đang là `extra="forbid"`,
thêm field mới không có default sẽ phá toàn bộ test hiện có):

| Field mới | Kiểu | Ghi chú |
|---|---|---|
| `keyframe_n` | `int \| None` | khoá join file BTC |
| `video_title` / `video_description` | `str` = `""` | từ `media-info` |
| `video_keywords` | `list[str]` = `[]` | từ `media-info` |
| `object_text` | `str` = `""` | xem 3.2 |
| `object_counts` | `dict[str,int]` = `{}` | |
| `Detection.mid` | `str \| None` | |
| `Detection.grid_cell` | `str \| None` | |
| `Detection.label_source` | thêm literal `"btc_detector"` | |
| `has_visual_text` | `bool` = `False` | frame này có caption/OCR hay không |

Giữ `SCHEMA_VERSION` bump lên `"1.1"` và cho `Literal["1.0","1.1"]`.

Validator `frame_id` hiện có (`schemas.py`, `validate_identity_and_references`)
vẫn đúng với `L21_V001` + `f0261` — không phải sửa.

### 3.2 Verbalization: structured → text

Kỹ thuật quan trọng nhất để **không phải sửa query DSL của Dev 3**. Sinh câu
tổng hợp từ detection rồi index như text thường:

```
"2 skyscraper, 1 lantern at center, 1 boat at bottom-left, 3 person"
```

BM25/Elasticsearch tự khớp keyword `boat`, `person` mà không cần nested query.
Song song vẫn giữ `objects` nested cho exact filter — hai biểu diễn cho hai mục đích.

Chèn cả term đã expand ở STEP 2 vào `object_text` (chỉ ancestor, tránh phình).

### 3.3 Luật ưu tiên khi merge nguồn

| Trường | Nguồn thắng | Lý do |
|---|---|---|
| `detections` | **BTC objects** | score thật, phủ 873 video |
| `detections` (fallback) | Florence `<OD>` | chỉ khi BTC không có file |
| `entities.objects` | **BTC objects** (qua lexicon) | LLM đang F1 `0.4194` — bỏ |
| `entities.time_of_day`, `setting`, `actions` | LLM | detector không suy ra được |
| `entities.colors` | STEP 4b nếu có ảnh, còn lại LLM | |
| `caption`, `ocr_text`, `code` | Task 1 (chỉ 60 video có ảnh) | giữ nguyên |

### 3.4 media-info

Đọc `media-info/{video_id}.json` (encoding **`utf-8`**, có tiếng Việt — dùng
`Path.read_text(encoding="utf-8")`, đừng để Python lấy cp1252 mặc định của Windows),
denormalize `title/description/keywords` xuống từng frame.

`description` khá dài và lặp (footer đăng ký kênh) → cắt còn ~500 ký tự đầu
hoặc bỏ phần sau dấu `►`.

**Nghiệm thu:** validate 100% record sinh ra; chạy lại
`benchmark_elasticsearch.py` trên 24 frame mẫu, metric **không được tụt**
(Recall@5 `1.0`, MRR@5 `0.9412`).

---

## STEP 4 — Spatial relation trên box BTC

**File:** sửa `spatial_extractor.py`

Code đã sẵn sàng cho việc này: có `ObjectDetector` Protocol
(`spatial_extractor.py:117`) và `infer_spatial_relations()` (`:431`) tách rời khỏi model.

Việc cần làm:

1. Viết `class BTCObjectDetector` implement Protocol, đọc từ parquet STEP 1,
   `model = "btc-openimages-v4"`.
2. Thêm đường chạy **không cần ảnh** — hàm `enrich_spatial_records()` hiện đang
   yêu cầu `image_dir` (`preflight_inputs()` ở `:518`). Thêm nhánh
   `--source btc-objects` bỏ qua kiểm tra ảnh.
3. `confidence` của relation = `min(score_subject, score_object)` thay cho hằng
   `0.5`. → đóng được gate P1 "confidence chỉ là proxy" trong TASK4_REPORT.
4. Giữ nguyên contextual label grounding (`ground_detection_labels()` `:350`),
   chạy đè lên label BTC — Open Images cũng nhầm kiểu `Tower` cho toà nhà.
5. Bump `SPATIAL_RULE_VERSION` → `spatial-v3`, ghi `detection_model` vào provenance.

**Cảnh báo chi phí:** 24 frame Florence đã sinh 216 raw relation. Với ~11 object/frame
BTC, số cặp là `11×10 = 110` relation thô/frame × 177k frame. **Bắt buộc** giữ
projection collapse theo `(subject_label, predicate, object_label)` mà bạn đã làm
(216 → 9), và chỉ index bản collapse.

**Nghiệm thu:** chạy lại `benchmark_spatial.py` trên gold set 10 frame; localization
và relation F1 không tụt; in `relations_indexed / frame` trung bình.

### STEP 4b (tuỳ chọn — chỉ 60 video có ảnh)

Color grounding: crop bbox → histogram HSV (`Pillow` + `colorsys`, không cần model)
→ màu trội → ghi vào `entities.colors` và `Detection.color`. Vá `colors` F1 `0.5000`
bằng pixel thật. Lọc bỏ vùng xám/độ bão hoà thấp trước khi lấy hue.

---

## STEP 5 — Elasticsearch v6

**File:** `elasticsearch_index.json` (bản v6), `elasticsearch_backend.py`, `server.py`

### 5.1 Mapping thêm (giữ `dynamic: strict`)

```jsonc
"keyframe_n":   { "type": "integer" },
"object_text":  { "type": "text", "analyzer": "kis_english" },
"object_labels":{ "type": "keyword", "normalizer": "lowercase_keyword" },  // đã expand ancestor
"object_counts":{ "type": "flattened" },
"objects": {                       // nested cho exact/spatial filter
  "type": "nested",
  "properties": {
    "label": {"type":"keyword","normalizer":"lowercase_keyword"},
    "mid":   {"type":"keyword"},
    "score": {"type":"float"},
    "grid_cell": {"type":"keyword"},
    "bbox":  {"type":"float","index":false}
  }
},
"video_title":       { "type":"text", "analyzer":"kis_vietnamese" },
"video_description": { "type":"text", "analyzer":"kis_vietnamese" },
"video_keywords":    { "type":"text", "analyzer":"kis_vietnamese" },
"has_visual_text":   { "type":"boolean" }
```

`kis_vietnamese`: analyzer mới = `standard` tokenizer + `lowercase` + `asciifolding`.
Không cần plugin tiếng Việt; `asciifolding` cho phép gõ không dấu vẫn hit.

### 5.2 Trọng số truy vấn (đề xuất khởi điểm, tune sau bằng benchmark)

`ocr_text^3` > `object_text^2` ≈ `caption^2` > `video_title^1.5` > `video_description^0.5`

### 5.3 API — mở rộng, không phá contract

- `filters.objects` đi qua `map_query_terms()` của STEP 2 trước khi build query.
- Thêm `filters.min_object_score` (mặc định `0.2`).
- Thêm `filters.object_counts` (vd `{"person": 3}` → `>= 3`).
- Response giữ nguyên `{frame_id, score, video_name, frame_index}`.

### 5.4 Ingest

Index vật lý `semantic_frames_v6`, alias `semantic_frames` chuyển atomically
(cơ chế bạn đã có). Ingest streaming theo từng video, đừng nạp 177k doc vào RAM.
Theo throughput đã đo (`1.558 docs/s`) → ~2 phút cho toàn kho.

**Nghiệm thu:** health báo đủ doc; p95 `< 50 ms`; case filter object mới đúng rank 1;
live test FastAPI → ES.

---

## STEP 6 — Đánh giá trên dữ liệu thật

1. **Query set thật:** ~30 câu KIS trên L21/L22 (chọn 2 nhóm này vì **có ảnh** để
   verify bằng mắt). Ghi vào `evaluation_queries_btc.json`.
2. **Cross-validation entity:** dùng BTC objects làm tín hiệu độc lập chấm lại
   `entities.objects` của LLM trên 12 frame đã có visual ground truth →
   trả lời dứt điểm gate P0 `F1 >= 0.70`. Dự đoán: thay LLM bằng detector sẽ vượt gate.
3. Cập nhật `task4_acceptance.py` với gate mới; viết `TASK6_REPORT.md`.

---

## STEP 7 — Backlog: thoát giới hạn 600 class

Chỉ làm nếu `unmapped_terms` ở STEP 2 cao. Lựa chọn: CLIP re-labeling trên crop,
hoặc open-vocab detector (OWL-ViT / GroundingDINO). **Phải thống nhất với Dev 1**
vì trùng model và trùng GPU budget.

---

## Thứ tự thực thi

```
STEP 0 ──> STEP 1 ──> STEP 2 ──> STEP 3 ──> STEP 5 ──> STEP 6
                 └──────────────> STEP 4 ─────┘
```

Đường găng là 0 → 1 → 2. STEP 4 chạy song song được ngay sau STEP 1.

## Việc cần đồng bộ với team

| Với ai | Nội dung |
|---|---|
| Dev 3, Dev 4 | `frame_id` đổi giá trị sang `L21_V001_f0261`; mọi ID cũ hết hiệu lực |
| Dev 4 | thumbnail phải trỏ theo `keyframe_n`: `keyframes/{video_id}/{n:03d}.jpg` |
| Dev 3 | 813/873 video không có caption/OCR → cần chỉnh trọng số RRF theo `has_visual_text` |
| Dev 1 | STEP 7 trùng model, đừng làm song song |

## Dependency cần thêm

- `pyarrow` (STEP 1) — chú ý pin `numpy<2.0.0` đang có trong `requirements.txt`.
