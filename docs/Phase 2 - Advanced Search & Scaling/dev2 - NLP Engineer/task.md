# Tasks for Dev 2 (NLP Engineer / Semantic Pipeline) - Phase 2

Trong Phase 2 (Advanced Search & Scaling), Dev 2 sẽ tập trung vào việc tận dụng tối đa lượng dữ liệu siêu dữ liệu mới được trích xuất (Objects, Spatial Relations) và giải quyết các bài toán về hiệu suất, độ chính xác (Visual F1) cũng như khả năng mở rộng (Scale).

> [!IMPORTANT]
> **Yêu cầu BẮT BUỘC trước khi code:** Dev 2 phải dành thời gian mở và đọc thật kỹ cấu trúc của các file JSON trong thư mục `data/metadata/` (đặc biệt là `metadata_spatial.json` và `metadata_youtube.jsonl`). Phải hiểu rõ data đầu vào có những trường (field) nào, cấu trúc dữ liệu ra sao trước khi bắt tay vào thiết kế thuật toán hay Mapping. Tuyệt đối không đoán mò cấu trúc dữ liệu!

Dưới đây là các hạng mục công việc cần thực hiện:

- [ ] **Task 1: Cập nhật BM25 để hỗ trợ Spatial & Objects (Giải pháp tạm thời)**
  - **Sửa file dữ liệu nguồn**: Cập nhật file `src/semantic_pipeline/database.py` để trỏ đường dẫn đọc vào file `data/metadata/metadata_spatial.json` thay vì `metadata/metadata.json` cũ.
  - **Hack logic BM25**: Viết script biến đổi các object và quan hệ không gian thành chuỗi văn bản (ví dụ: biến quan hệ `{subject: "poster", predicate: "overlapping", object: "skyscraper"}` thành chuỗi text `"poster overlapping skyscraper"`). Nhồi chuỗi này vào khối Corpus của BM25 để cho phép user query dựa trên quan hệ không gian cơ bản bằng cơ chế khớp từ khóa (Keyword matching).

- [ ] **Task 2: Thiết lập Elasticsearch (Scale & Filter)**
  - **Dựng server**: Viết cấu hình `docker-compose.yml` để khởi chạy Elasticsearch qua Docker.
  - **Data Indexing & Mapping Research**: Tự nghiên cứu và định nghĩa Mapping Index phù hợp nhất để nạp dữ liệu từ `data/metadata/metadata_spatial.json` vào Elasticsearch. Mục tiêu là phải hỗ trợ được cả tìm kiếm full-text (cho Caption/OCR) và lọc dữ liệu đa chiều phức tạp (như Objects, Spatial Relations, Time, Setting).
  - **API Integration**: Cập nhật hàm search trong `server.py` để sử dụng Elasticsearch client. Đảm bảo hỗ trợ các tính năng `Filters` được định nghĩa trong tài liệu `api_contract_v1.md`.

- [ ] **Task 3: Nâng cấp Mô hình Trích xuất (Fix Visual F1)**
  - **Thực trạng**: Theo báo cáo Task 4 của Phase 1, `Florence-2` sinh ra nhiều lỗi ảo giác (nhận diện sai vật thể), khiến chỉ số Entity Visual F1 chỉ đạt `0.4194` (chưa đạt chuẩn Production gate 0.70).
  - **Nhiệm vụ**: Tích hợp và thử nghiệm các mô hình VLM (Vision-Language Model) mạnh hơn (ví dụ: `Qwen2-VL` hoặc `LLaVA`) vào bước Spatial Reasoning (`spatial_extractor.py`) để cải thiện độ chuẩn xác khi bóc tách Bounding Box.

- [ ] **Task 4: Tích hợp YouTube Metadata (Bí kíp tăng Precision)**
  - **Sử dụng Dữ liệu mới**: Hai file `metadata_youtube.jsonl` và `metadata_youtube_bm25.pkl` vừa được sao chép vào thư mục `data/metadata/`. Đây là dữ liệu gốc của BTC (Tiêu đề, Kênh, Description).
  - **Logic Tìm kiếm 2 Tầng**: Cập nhật hàm search trong `server.py` để tra cứu trong file YouTube Metadata này trước (hoặc load cái cục pickle BM25 dựng sẵn kia lên). Nếu câu hỏi liên quan đến Tên chương trình (VD: "Bản tin 60 giây HTV"), thì lọc ra ID video trước, rồi mới ép kết quả tìm kiếm Frame (Visual BM25) rơi vào trong phạm vi của Video đó.
