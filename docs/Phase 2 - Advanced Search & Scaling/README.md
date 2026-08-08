# Architecture Overview - Phase 2 (Advanced Search & Scaling)

Tài liệu này tổng hợp kiến trúc luồng dữ liệu của hệ thống Truy xuất Video đa phương thức cho Phase 2. Sau khi đã chứng minh được tính khả thi ở Phase 1 (MVP), Phase 2 tập trung vào việc **scale hệ thống lên dữ liệu 100GB**, cải thiện **độ chính xác (Visual F1)** và hỗ trợ **Truy vấn không gian (Spatial Reasoning)** sâu hơn.

## ⚙️ CÁC NÂNG CẤP CHÍNH Ở PHASE 2

1. **Thay thế BM25 bằng Elasticsearch**: Chuyển đổi công cụ tìm kiếm văn bản từ chạy trên RAM (BM25) sang ổ cứng (Elasticsearch) để xử lý lượng metadata khổng lồ, đồng thời kích hoạt tính năng Exact Metadata Filtering.
2. **Nâng cấp Bóc tách Không gian (Spatial Reasoning)**: Thay thế Florence-2 bằng các mô hình khủng hơn (Qwen2-VL) để giảm thiểu ảo giác, tăng độ chuẩn xác của tọa độ (Bounding Box).
3. **Mở rộng API Contract**: Hỗ trợ các Query Filter phức tạp và trả về tọa độ để Frontend có thể vẽ khung (Box) đè lên hình ảnh.
4. **Tối ưu Hàng chờ & Caching**: Tích hợp Redis và Message Queue cho Backend LLM để chịu tải lúc thi đấu.

## Phân bổ thư mục:
- `dev1`: Nâng cấp Vector DB, Indexing phân tán.
- `dev2`: Nâng cấp Elasticsearch & VLM Model.
- `dev3`: Tối ưu hóa LLM Prompt & Caching.
- `dev4`: Trích xuất Script Video (ASR/Whisper) & Vẽ Bounding Box.
- API dùng chung: [`docs/API_CONTRACT.md`](../API_CONTRACT.md).
