# API Contract v1 - Phase 1: Known-Item Search (KIS)

Tài liệu này đặc tả luồng giao tiếp dữ liệu giữa Frontend (Dev 4) và Backend (Dev 3) cho Phase 1 của dự án AICHCM-2026-Glitch.

## 1. Tìm kiếm bằng văn bản (Text Search)
- **Endpoint**: `POST /api/v1/search/text`
- **Description**: Trả về danh sách các frames tương ứng với câu truy vấn văn bản.
- **Request Payload**:
  ```json
  {
    "query": "string (Mục tiêu tìm kiếm) - Bắt buộc",
    "top_k": "integer (Số kết quả muốn lấy, mặc định: 50) - Optional"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success|error",
    "data": {
      "results": [
        {
          "frame_id": "string - Bắt buộc",
          "score": "float - Bắt buộc",
          "thumbnail_url": "string - Bắt buộc",
          "metadata": "object - Optional"
        }
      ]
    }
  }
  ```

## 2. Tìm kiếm bằng ảnh (Image Search)
- **Endpoint**: `POST /api/v1/search/image`
- **Description**: Trả về danh sách các frames tương tự với ảnh upload.
- **Request Payload**: `multipart/form-data`
  - `image`: File ảnh (jpeg, png) - Bắt buộc
  - `top_k`: `integer` - Optional
- **Response**: Cấu trúc tương tự như API Text Search.

## 3. Lấy Context Frames (Timeline)
- **Endpoint**: `GET /api/v1/frames/context/{frame_id}`
- **Description**: Lấy các frames liền kề trước/sau frame_id để tạo timeline.
- **Query Parameters**:
  - `window`: `integer` (Số lượng frames trước và sau cần lấy, mặc định: 5) - Optional
- **Response**:
  ```json
  {
    "status": "success|error",
    "data": {
      "context_frames": [
        {
          "frame_id": "string",
          "thumbnail_url": "string",
          "timestamp": "float|string"
        }
      ]
    }
  }
  ```
