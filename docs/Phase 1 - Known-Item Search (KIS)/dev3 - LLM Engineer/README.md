# Dev 3 Workspace
Thư mục làm việc dành cho Dev 3 (Backend).

## Nhật ký Cập nhật (Change Log)

### Refactor Code theo Chuẩn API Contract & Cấu trúc gốc (Mới nhất)
Code đã được tự động refactor để tuân thủ 100% `task.md` và `API_CONTRACT.md` công cộng:
- **Cấu trúc lại thư mục**: Chuyển các model vào `schemas/search.py` và `schemas/frames.py`. Tách pipeline client về đúng `clients/visual_client.py` và `clients/semantic_client.py`. Chuyển logic về `query_analyzer.py` và `search_orchestrator.py` đúng như sườn dự án ban đầu.
- **Prefix & CORS**: Thêm `/api/v1` và cấu hình CORS đầy đủ cho Frontend `http://localhost:5173`.
- **Response Format**: Chuyển schema sang dạng chuẩn `{"status", "message", "data"}` và xoá các trường nội bộ (warnings, parsed_query...) không cần thiết cho UI.
- **Tính năng mở rộng**: Bổ sung `top_k=50` cho UI, tạo cơ chế fallback cho LLM khi bị lỗi, thiết lập StaticFiles tại `/media/thumbnails`, và xử lý chuẩn frame biên (window frames) đầu/cuối video.
- **Unit Tests**: Thêm 3 test suit kiểm tra `RRF`, `LLM Fallback` và `Frame logic` (xem thư mục `src/backend/tests/`).
