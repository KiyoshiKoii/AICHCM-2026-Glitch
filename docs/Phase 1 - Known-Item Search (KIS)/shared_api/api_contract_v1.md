# API Contract v1 - Phase 1: Known-Item Search (KIS)

Tài liệu này đặc tả luồng giao tiếp dữ liệu giữa Frontend (Dev 4) và Backend (Dev 3) cho Phase 1 của dự án AICHCM-2026-Glitch.

## PHASE 1: CORE KIS
**Base URL:** `http://localhost:8000/api/v1`
**Required Headers:** `Content-Type: application/json`

### 1. Query by Text
**Interface:** UI & LLM Parser → RRF (Vector + Text DB)
**Endpoint:** `POST /search/text`
**Mô tả:** Nhận câu hỏi tiếng Việt, trả về danh sách khung hình đã được gộp điểm RRF từ cả 2 hệ thống Text và Visual.

**Request Body:**
```json
{
  "query": "người đàn ông mặc áo đỏ làm rơi ví",
  "top_k": 50
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Retrieved successfully",
  "data": {
    "total_results": 50,
    "results": [
      {
        "frame_id": "L21_V022_f1024",
        "video_name": "L21_V022",
        "frame_index": 1024,
        "score": 0.98,
        "thumbnail_url": "/media/thumbnails/L21_V022_f1024.jpg",
        "metadata": {
          "video_name": "L21_V022",
          "frame_index": 1024
        }
      }
    ]
  }
}
```

### 2. Tìm kiếm bằng Hình ảnh (Visual Query Expansion)
**Interface:** UI & Vector DB Pipeline
**Endpoint:** `POST /search/image`
**Mô tả:** Nhận file ảnh upload từ người dùng, nhúng qua mô hình CLIP và truy vấn Qdrant để tìm khung hình tương tự.

**Request (multipart/form-data):**
- `image_file`: [File ảnh đính kèm] (Định dạng .jpg, .png)
- `top_k`: `50` (Dạng Form Data Text)

**Response (200 OK):**
*(Tái sử dụng chung một Model Response với API `/search/text` ở trên để Frontend dùng chung 1 Component vẽ lưới ảnh cho nhàn).*
```json
{
  "status": "success",
  "message": "Image retrieved successfully",
  "data": {
    "total_results": 50,
    "results": [...]
  }
}
```

### 3. Điều hướng Dòng thời gian (Temporal Navigation)
*High-speed API for context verification (No AI required).*
**Endpoint:** `GET /frames/context/{frame_id}`
**Query Parameters:** `?window=5` (Lấy 5 frame trước và 5 frame sau)
**Mô tả:** Frontend gọi API này khi user click đúp vào 1 bức ảnh.
**Request URL Example:** `GET /frames/context/L21_V022_f1024?window=5`

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "center_frame": { "frame_id": "L21_V022_f1024", "thumbnail_url": "..." },
    "before_frames": [...],
    "after_frames": [...]
  }
}
```

> 💡 **Bí kíp Quản trị (PM Note):**
> - **Thumbnail Optimization:** Dev 3 must host static files. Frontend must use direct URLs in `<img src="...">`. Avoid Base64 encoding to preserve bandwidth.
> - **Frame ID Convention:** IDs (e.g., `L21_V022_f1024`) include Video Name and Frame Index theo format chuẩn của BTC `L\d+_V\d+_f\d{4}` để Frontend hiển thị dễ dàng và giúp thí sinh copy nộp bài (submit) ngay lập tức.

### 4. Truy vấn Vector (Gọi Dev 1 - Visual Pipeline)
**Endpoint:** `POST http://localhost:8001/internal/search/visual`
**Mô tả:** Dev 3 truyền câu prompt tiếng Anh đã được LLM dịch vào đây. Dev 1 nhúng qua CLIP và query Qdrant.

**Request Body:**
```json
{
  "visual_prompt": "A person dropping a pink teddy bear keychain",
  "top_k": 200
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {"frame_id": "L21_V022_f1024", "score": 0.88, "video_name": "L21_V022", "frame_index": 1024},
    {"frame_id": "L22_V012_f0055", "score": 0.81, "video_name": "L22_V012", "frame_index": 55}
  ]
}
```

### 5. Truy vấn Ngữ nghĩa (Gọi Dev 2 - Semantic Pipeline)
**Endpoint:** `POST http://localhost:8002/internal/search/text`
**Mô tả:** Dev 3 truyền mảng từ khóa tiếng Anh đã được LLM mở rộng vào đây. Dev 2 query Elasticsearch/BM25.

**Request Body:**
```json
{
  "keywords": ["pink teddy bear", "keychain", "dropping", "falling"],
  "top_k": 200
}
```

**Metadata filters (tùy chọn, tương thích ngược):** Khi Dev 2 chạy backend
Elasticsearch, request có thể thêm `filters`. Request cũ không có field này vẫn
hoạt động như trước.

```json
{
  "keywords": ["person", "red car"],
  "top_k": 200,
  "filters": {
    "time_of_day": "night",
    "setting": "outdoor",
    "locations": ["street"],
    "objects": ["person", "car"],
    "actions": ["standing"],
    "colors": ["red"],
    "code_language": "sql",
    "code_patterns": ["not exists", "correlated"],
    "spatial_relations": [
      {"subject": "person", "predicate": "left_of", "object": "car"}
    ]
  }
}
```

- Các field hỗ trợ: `time_of_day`, `setting`, `locations`, `objects`,
  `actions`, `colors`, `code_language`, `code_patterns`, `spatial_relations`.
- `code_language` hiện nhận `unknown` hoặc `sql`; `code_patterns` là các tín
  hiệu cú pháp đã chuẩn hóa như `not exists`, `group by`, `correlated`.
- Mảng dùng semantics **all-of (AND)**: `objects=["person","car"]` yêu cầu
  frame có đủ cả hai object.
- Predicate không gian: `left_of`, `right_of`, `above`, `below`, `overlapping`.
- Filter chỉ khả dụng với Elasticsearch; backend BM25 trả HTTP 400 thay vì âm
  thầm bỏ qua filter.

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {"frame_id": "L21_V022_f1024", "score": 15.6, "video_name": "L21_V022", "frame_index": 1024},
    {"frame_id": "L23_V008_f0200", "score": 12.1, "video_name": "L23_V008", "frame_index": 200}
  ]
}
```

### 6. API Phục vụ Ảnh tĩnh (Static File Server)
**Endpoint:** `GET /media/thumbnails/{frame_id_ext}` (Ví dụ: `L21_V022_f1024.jpg`)
**Mô tả:** API phục vụ hình ảnh. Route Backend hiện tại đang bóc tách `frame_id_ext` bằng regex `r"(L\d+)_V(\d+)_f(\d+)\.jpg"` để trỏ đúng vào thư mục phẳng `../data/keyframes/{l_part}_V{v_part}/`.
**Response:** Trả về file ảnh dạng `image/jpeg`. Frontend gọi thẳng URL này trong thẻ `<img>`.
