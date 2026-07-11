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
        "frame_id": "vid05_f1024",
        "video_name": "vid05.mp4",
        "frame_index": 1024,
        "score": 0.98,
        "thumbnail_url": "/media/thumbnails/vid05_f1024.jpg",
        "metadata": {
          "camera_id": "cam_02",
          "timestamp": "08:15:22"
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
**Request URL Example:** `GET /frames/context/vid05_f1024?window=5`

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "center_frame": { "frame_id": "vid05_f1024", "thumbnail_url": "..." },
    "before_frames": [...],
    "after_frames": [...]
  }
}
```

> 💡 **Bí kíp Quản trị (PM Note):**
> - **Thumbnail Optimization:** Dev 3 must host static files. Frontend must use direct URLs in `<img src="...">`. Avoid Base64 encoding to preserve bandwidth.
> - **Frame ID Convention:** IDs (e.g., `vid05_f1024`) include Video Name and Frame Index to allow stateless temporal calculations in the Backend.

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
    {"frame_id": "vid05_f1024", "score": 0.88, "video_name": "vid05.mp4", "frame_index": 1024},
    {"frame_id": "vid12_f055", "score": 0.81, "video_name": "vid12.mp4", "frame_index": 55}
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

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {"frame_id": "vid05_f1024", "score": 15.6, "video_name": "vid05.mp4", "frame_index": 1024},
    {"frame_id": "vid08_f200", "score": 12.1, "video_name": "vid08.mp4", "frame_index": 200}
  ]
}
```

### 6. API Phục vụ Ảnh tĩnh (Static File Server)
**Endpoint:** `GET /media/thumbnails/{frame_id}.jpg`
**Mô tả:** API này không cần code logic phức tạp. Dev 3 chỉ cần dùng tính năng `StaticFiles` của FastAPI (hoặc `send_from_directory` của Flask) để trỏ thẳng vào thư mục chứa ảnh frame trên ổ cứng.
**Response:** Trả về file ảnh dạng `image/jpeg`. Frontend gọi thẳng URL này trong thẻ `<img>`.
