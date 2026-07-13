# Tasks for Dev 2 (NLP Engineer / Semantic Pipeline)

Dưới đây là các hạng mục công việc cần hoàn thiện:

- [ ] **Task 1: Xây dựng Luồng Trích xuất Ngữ nghĩa (Text Generation)**
  - **Tải Model**: Cài đặt mô hình `microsoft/Florence-2-base` (hoặc `Florence-2-large` nếu VRAM > 12GB). Đây là con VLM sinh text SOTA và nhẹ nhất hiện nay. *(Dự phòng: Moondream2)*. **Lưu ý**: Tương tự Dev 1, nhớ code nhận diện thiết bị tự động (`device = "cuda" if torch.cuda.is_available() else "cpu"`). Bản Florence-2-base cực kỳ nhỏ nhẹ nên máy cá nhân chạy CPU vẫn test rất mượt (chỉ mất vài giây/ảnh), sau này quăng lên Cloud GPU thì code vẫn tương thích 100%.
  - **Trích xuất Text**: Viết script đọc thư mục ảnh đầu vào. Với mỗi bức ảnh, ép model Florence-2 chạy 2 tác vụ (tasks) độc lập:
    - `<DETAILED_CAPTION>`: Sinh ra một đoạn văn miêu tả chi tiết bối cảnh, con người, hành động.
    - `<OCR>`: Đọc toàn bộ chữ viết xuất hiện trong ảnh (chữ trên áo, biển số xe, bảng hiệu...).
  - **Lưu trữ nháp**: Gom 2 cục text này lại, cộng thêm cái tên file ảnh (`frame_id`), lưu tạm thành một file `metadata.json` hoặc `.csv` để chuẩn bị nạp DB.

- [ ] **Task 2: Thiết lập Database Văn bản (Text DB)**
  *(Ghi chú PM: Để code MVP nhanh nhất trong tuần này mà không cần cài Elasticsearch nặng nề, Dev 2 hãy dùng thư viện BM25 thuần Python trước).*
  - **Khởi tạo DB Local**: Sử dụng thư viện `rank_bm25` (Python) hoặc Whoosh để tạo một bộ máy tìm kiếm Full-text search lưu thẳng trên RAM/Ổ cứng local.
  - **Nạp Data**: Đọc file `metadata.json` (ở Task 1), Tokenize (cắt từ) các đoạn Caption và OCR, sau đó nạp vào bộ máy BM25.
  - **Khóa chính**: Đảm bảo mỗi Document nạp vào đều được map chuẩn với cái `frame_id`.

- [ ] **Task 3: Dựng API Nội bộ (Internal API)**
  - **Setup Server**: Khởi tạo server bằng FastAPI hoặc Flask, cấu hình chạy cố định ở Port 8002.
  - **Mở Endpoint**: Xây dựng API `POST /internal/search/text`.
  - **Xử lý Request**: Nhận JSON payload chứa mảng `keywords` (đã được Dev 3 dùng LLM mở rộng) và `top_k`.
  - **Xử lý Logic DB**: Quăng mảng `keywords` này vào hàm Search của BM25 để tính điểm mức độ liên quan. Sắp xếp từ cao xuống thấp.
  - **Luật Điểm số**: Khác với Vector, điểm BM25 không có mức trần (có thể lên tới 10, 20 hoặc 100). Dev 2 KHÔNG CẦN chuẩn hóa, cứ trả về điểm gốc. Dev 3 sẽ tự dùng thuật toán thứ hạng (Rank) để xử lý.
  - **Trả Response**: Nhả cục JSON kết quả chứa mảng `[frame_id, score, video_name, frame_index]` tuân thủ nghiêm ngặt theo API Contract.

- [ ] **Task 4: Nghiên cứu Công nghệ Nâng cao (R&D)**
  *(Dự phòng cho tuần sau khi cần nâng cấp độ "khôn" của hệ thống).*
  - **Chuyển đổi sang Elasticsearch**: Khi data thực tế của BTC lên tới 100GB, BM25 thuần Python sẽ bị phình RAM. Dev 2 bắt buộc phải học cách dùng Docker để dựng Elasticsearch, biết cách tạo Index và cấu hình Analyzer chuẩn cho tiếng Anh.
  - **Xử lý quan hệ không gian (Spatial Reasoning)**: Các câu KIS rất hay có kiểu "A đứng BÊN TRÁI B". Florence-2 đôi khi miêu tả chung chung. Nghiên cứu cách bắt model sinh ra Bounding Box (Tọa độ) của vật thể để lọc logic Trái/Phải/Trên/Dưới.
  - **Entity Extraction (Trích xuất thực thể)**: Viết thêm một bước hậu xử lý (Post-processing) dùng LLM nhỏ (như Llama-3) đọc cái Caption của Florence-2 và tách ra thành các trường siêu dữ liệu chuẩn xác: `{"Time": "Night", "Location": "Indoor", "Objects": [...]}` để sau này làm bộ lọc Filter cho API.

- [ ] **Task 5: Unit Testing & Performance Testing**
  - **API Test**: Xây dựng bài test cho endpoint `/internal/search/text` để đảm bảo kết quả JSON trả về đúng format `[frame_id, score, video_name, frame_index]` với HTTP Status 200.
  - **Accuracy & Speed Test**: Đo lường tốc độ query của BM25 trên local RAM và test chéo độ chuẩn xác của nội dung OCR sinh ra từ Florence-2.

- [ ] **Task 6: Benchmark & So sánh Model Semantic**
  - **Thiết lập benchmark cố định**: Tạo và sử dụng chung manifest query/ground truth (`qrels.jsonl`) trong `tests/semantic_pipeline/benchmark/`, đồng thời có tập annotation nhỏ cho OCR/caption.
  - **Chấm baseline trước**: Đo `microsoft/Florence-2-base` + BM25 trước khi thử Florence-2-large, Moondream hoặc model mới. Mọi model phải dùng cùng corpus, prompt/task, tokenizer, schema metadata và `top_k`.
  - **Đo chỉ số bắt buộc**: Tính Recall@1, Recall@5, Recall@10, MRR@10; trên tập có nhãn tính thêm OCR CER, Word F1 và Concept Recall của caption; đo p50/p95 query, tốc độ sinh metadata và RAM/VRAM.
  - **Lưu kết quả có thể tái lập**: Ghi model revision, commit, hash manifest, prompt/task, batch size, precision, phần cứng và cấu hình BM25. Raw output đặt trong `tests/semantic_pipeline/benchmark/results/` (không commit); thêm bảng tóm tắt vào `benchmark_results.md`.
  - **Làm theo hướng dẫn**: Đọc và tuân thủ [benchmark_guide.md](benchmark_guide.md) trước khi báo cáo model tốt hơn baseline.

## 📂 Hướng dẫn Tổ chức Thư mục & Code (Codebase Guidelines)
Toàn bộ source code của bạn sẽ được phát triển trong thư mục **`src/semantic_pipeline/`**.
- **Cài đặt môi trường**: 
  1. **Cài đặt PyTorch thủ công** sao cho khớp với phần cứng của bạn. (VD: Nếu máy có GPU NVIDIA thì chạy `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`, nếu chỉ có CPU thì xem lệnh trên trang chủ PyTorch).
  2. Cài các thư viện dùng chung còn lại: Chạy `conda env create -f environment.yml` (khuyên dùng) hoặc `pip install -r requirements.txt` tại thư mục gốc.
- Code đã được tạo sẵn khung (scaffold) với các file: `extractor.py` (cho Task 1), `database.py` (cho Task 2), và `server.py` (cho Task 3).
- **Lưu ý**: Đối với dữ liệu thô `metadata.json` hay `.csv`, file `.gitignore` chung ở thư mục gốc đã được cấu hình sẵn để tự động chặn không cho push lên Git vì dung lượng chúng rất nặng.

## 📚 Tài liệu Tham khảo (Reference Papers cho Task 4)
- **Elasticsearch Documentation & Advanced BM25 Tuning**: Tài liệu nền tảng về hệ thống tìm kiếm Full-text search hiện đại, giúp dễ dàng chuyển đổi cấu trúc BM25 thuần lên kiến trúc Microservice và tinh chỉnh trọng số tìm kiếm chuẩn xác.
- **Visual Spatial Reasoning (VSR) & Grounding**: Các nghiên cứu về cách buộc mô hình VLM sinh mô tả gắn liền với tọa độ không gian (Bounding Boxes) để giải quyết các truy vấn yêu cầu định hướng không gian khắt khe.
- **Information Extraction using LLMs**: Khai thác sức mạnh của LLM (như Llama-3, Mistral) để cấu trúc hóa dữ liệu phi cấu trúc (Unstructured Text) thành các schema JSON tĩnh như Thời gian, Địa điểm, Đối tượng nhằm hỗ trợ Metadata Filtering.

## 🌿 Hướng dẫn Git & Đặt tên Branch (Git Workflow & Branching)
Mỗi dev sẽ có một branch chính để làm việc. Branch chính của bạn là: **`feat/semantic-pipeline`**.

- **Quy tắc đặt tên nếu bạn tạo thêm branch phụ (mở rộng tính năng/sửa bug):**
  - Cú pháp: `feat/semantic-pipeline-<loại>-<tên_chức_năng>`
  - Ví dụ thêm tính năng: `feat/semantic-pipeline-feature-yolo-model`
  - Ví dụ sửa lỗi: `feat/semantic-pipeline-bugfix-vector-dim`
- Hãy nhớ **push code thường xuyên** lên branch của mình trên GitHub để backup.
