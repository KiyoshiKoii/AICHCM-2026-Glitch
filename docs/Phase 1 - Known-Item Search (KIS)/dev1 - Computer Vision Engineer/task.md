# Tasks for Dev 1 (Computer Vision Engineer / Visual Pipeline)

Dưới đây là các hạng mục công việc cần hoàn thiện:

- [x] **Task 1: Xây dựng Luồng Trích xuất Vector (Visual Pipeline)**
  - **Tải Model**: Cài đặt và sử dụng model `openai/clip-vit-base-patch32` thông qua thư viện `sentence-transformers` hoặc `transformers` trên máy local.
  - **Trích xuất ảnh**: Viết script đọc thư mục ảnh đầu vào và nhúng (embed) chúng thành các Vector đặc trưng 512 chiều.
  - **Ép hiệu năng (Batch Inference)**: Viết code đẩy data vào model theo lô với Batch Size = 16 hoặc 32 để tránh nghẽn cổ chai. **Lưu ý quan trọng**: Phải code tự động nhận diện phần cứng (`device = "cuda" if torch.cuda.is_available() else "cpu"`). Khi test ở local máy tính chỉ có CPU thì hệ thống vẫn chạy bình thường (dù chậm), sau này quăng lên Cloud có GPU thì code vẫn tương thích 100%.

- [x] **Task 2: Thiết lập Vector DB (Chạy Local, KHÔNG Docker)**
  - **Khởi tạo Qdrant**: Dùng thư viện Python `qdrant-client` cấu hình chế độ Local Storage (`path="./local_qdrant_db"`) để tạo DB lưu trực tiếp trên ổ cứng.
  - **Tạo Collection**: Định nghĩa cấu trúc lưu trữ và nạp toàn bộ Vector sinh ra ở Task 1 vào database.
  - **Gắn Payload**: Đảm bảo mỗi vector nạp vào đều được gắn metadata (Payload) chính là tên file ảnh (`frame_id`).

- [x] **Task 3: Dựng API Nội bộ (Internal API)**
  - **Setup Server**: Khởi tạo server bằng FastAPI hoặc Flask, cấu hình chạy cố định ở Port 8001.
  - **Mở Endpoint**: Xây dựng API `POST /internal/search/visual`.
  - **Xử lý Request**: Nhận JSON payload chứa `visual_prompt` và `top_k`. Gọi model CLIP để nhúng câu prompt tiếng Anh này thành một vector text.
  - **Xử lý Logic DB**: Truy vấn vào Qdrant bằng phép đo Cosine Similarity để lấy ra top kết quả khớp nhất.
  - **Chuẩn hóa Điểm**: Bắt buộc phải ép điểm Cosine về thang đo chuẩn 0 - 1.
  - **Trả Response**: Nhả cục JSON kết quả chứa mảng gồm `frame_id, score, video_name, frame_index` tuân thủ nghiêm ngặt theo API Contract đã chốt với Dev 3.

- [ ] **Task 4: Nghiên cứu Công nghệ Nâng cao (R&D)**
  *(Đây là task song song/Dự phòng để nâng cấp hệ thống sau khi Baseline đã chạy thông luồng).*
  - **Nâng cấp Model**: Tìm hiểu các mô hình nhận diện mạnh hơn như SigLIP (nhận diện vật thể nhỏ cực tốt) hoặc InternVideo2 (nắm bắt hành động, chuyển động thời gian).
  - **Tối ưu RAM (Quantization)**: Nghiên cứu tính năng lượng tử hóa ép vector xuống kiểu int8 của Qdrant và kỹ thuật MRL (Matryoshka) để chuẩn bị cho việc nạp khối dữ liệu 100GB.
  - **Late Interaction**: Đọc hiểu cơ chế ColBERT/ColPali (lưu nhiều vector cho một tấm ảnh thay vì 1 vector duy nhất) để tối ưu hóa việc tìm kiếm các vật thể li ti trong khung hình.

- [x] **Task 5: Unit Testing & Performance Testing**
  - **API Test**: Xây dựng test cho endpoint `/internal/search/visual` đảm bảo response trả về đúng format `[frame_id, score, video_name, frame_index]` và HTTP status 200.
  - **Performance Test**: Đo tốc độ trích xuất Vector với Batch Inference và tốc độ truy vấn Qdrant để đảm bảo đạt độ trễ cho phép (dưới 500ms).

- [ ] **Task 6: Benchmark & So sánh Model Visual**
  - **Thiết lập benchmark cố định**: Tạo và sử dụng chung manifest query/ground truth (`qrels.jsonl`) trong `tests/visual_pipeline/benchmark/`; không chọn thủ công query có lợi cho model.
  - **Chấm baseline trước**: Đo `openai/clip-vit-base-patch32` trước khi thử SigLIP, InternVideo hoặc model mới. Mọi model phải dùng cùng corpus, preprocessing, metric Cosine, cấu hình Qdrant và `top_k`.
  - **Đo chỉ số bắt buộc**: Tính Recall@1, Recall@5, Recall@10, MRR@10; đo p50/p95 query, tốc độ embedding/index và RAM/VRAM/dung lượng index.
  - **Lưu kết quả có thể tái lập**: Ghi model revision, commit, hash manifest, batch size, precision, phần cứng và cấu hình index. Raw output đặt trong `tests/visual_pipeline/benchmark/results/` (không commit); thêm bảng tóm tắt vào `benchmark_results.md`.
  - **Làm theo hướng dẫn**: Đọc và tuân thủ [benchmark_guide.md](benchmark_guide.md) trước khi báo cáo model tốt hơn baseline.

## 📂 Hướng dẫn Tổ chức Thư mục & Code (Codebase Guidelines)
Toàn bộ source code của bạn sẽ được phát triển trong thư mục **`src/visual_pipeline/`**.
- **Cài đặt môi trường**: 
  1. **Cài đặt PyTorch thủ công** sao cho khớp với phần cứng của bạn. (VD: Nếu máy có GPU NVIDIA thì chạy `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`, nếu chỉ có CPU thì xem lệnh trên trang chủ PyTorch).
  2. Cài các thư viện dùng chung còn lại: Chạy `conda env create -f environment.yml` (khuyên dùng) hoặc `pip install -r requirements.txt` tại thư mục gốc.
- Code đã được tạo sẵn khung (scaffold) với các file: `extractor.py` (cho Task 1), `database.py` (cho Task 2), và `server.py` (cho Task 3).
- Lưu trữ local database của Qdrant thẳng vào thư mục `local_qdrant_db/` trong `src/visual_pipeline` (thư mục này đã được cấu hình chặn sẵn trong file `.gitignore` chung ở gốc để không bị push nhầm lên mạng).

## 📚 Tài liệu Tham khảo (Reference Papers cho Task 4)
- **Sigmoid Loss for Language Image Pre-Training (SigLIP)**: Đề xuất hàm mất mát Sigmoid thay cho Softmax của CLIP truyền thống, giúp model có khả năng nhận diện các chi tiết nhỏ lẻ, cục bộ trong khung hình một cách vượt trội.
- **InternVideo2 (Video Foundation Model)**: Bài báo giới thiệu kiến trúc nắm bắt chuỗi hành động và chuyển động theo thời gian cực kỳ mạnh mẽ, chuyên trị các câu query yêu cầu tính logic diễn biến (ví dụ: "đang đứng rồi ngồi xuống").
- **Matryoshka Representation Learning (MRL)**: Hướng dẫn kỹ thuật ép nhỏ kích thước Vector (từ 512 chiều xuống còn 64 chiều hoặc nhỏ hơn) mà vẫn giữ được độ chính xác gần như nguyên vẹn, là "cứu cánh" để giải quyết bài toán tràn RAM khi nạp 100GB dữ liệu vào Qdrant.
- **ColPali / ColBERT (Late Interaction)**: Cung cấp cơ chế tìm kiếm lai bằng cách lưu nhiều vector cho các vùng nhỏ (patch) của cùng một bức ảnh, giúp không bỏ sót các vật thể li ti mà mô hình sinh vector đơn (Single-vector model) hay bị mất thông tin.

## 🌿 Hướng dẫn Git & Đặt tên Branch (Git Workflow & Branching)
Mỗi dev sẽ có một branch chính để làm việc. Branch chính của bạn là: **`feat/visual-pipeline`**.

- **Quy tắc đặt tên nếu bạn tạo thêm branch phụ (mở rộng tính năng/sửa bug):**
  - Cú pháp: `feat/visual-pipeline-<loại>-<tên_chức_năng>`
  - Ví dụ thêm tính năng: `feat/visual-pipeline-feature-yolo-model`
  - Ví dụ sửa lỗi: `feat/visual-pipeline-bugfix-vector-dim`
- Hãy nhớ **push code thường xuyên** lên branch của mình trên GitHub để backup.
