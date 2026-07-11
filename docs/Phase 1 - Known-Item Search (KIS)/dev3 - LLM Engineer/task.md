# Tasks for Dev 3 (LLM Engineer / Backend)

Dưới đây là các hạng mục công việc cần hoàn thiện cho (Online Serving):

- [ ] **Nghiên cứu & Tích hợp LLM Parser**: Đánh giá và tìm hiểu các mô hình ngôn ngữ (bao gồm cả các bản trả phí như GPT-4, Claude,... để đảm bảo chất lượng). Tuy nhiên, trong quá trình test code ban đầu, tạm thời dùng Ollama tự host ở local. Yêu cầu: Nhận câu query tiếng Việt -> Prompt tiếng Anh (Visual) & Keywords đồng nghĩa (Semantic).
- [ ] **API Query by Text (`POST /search/text`)**: Gọi song song 2 API nội bộ (Dev 1 & Dev 2), dùng công thức **RRF** để trộn điểm, gộp kết quả, cắt lấy Top 20 (để tránh tải quá nặng cho UI) và gán URL thumbnail.
- [ ] **API Query by Image (`POST /search/image`)**: Forward trực tiếp ảnh upload từ UI sang Dev 1 (Vector DB pipeline) mà không qua LLM hay Dev 2.
- [ ] **API Temporal Navigation (`GET /frames/context/{frame_id}`)**: Xử lý logic chuỗi, tách số thứ tự trong `frame_id` (VD: `vid05_f1024` -> `1024`), cộng trừ để lấy ra mảng 11 frames liền kề (5 trước, 5 sau).
- [ ] **Unit Testing & API Testing**: Xây dựng Unit test cho các function cốt lõi (đặc biệt là hàm tính điểm RRF và module phân tích LLM JSON parser) để đảm bảo không bị lỗi dữ liệu đầu ra.

## 📂 Hướng dẫn Tổ chức Thư mục & Code (Codebase Guidelines)
Toàn bộ source code của bạn sẽ được phát triển trong thư mục **`src/backend/`**.
- Nên thiết lập cấu trúc thư mục theo chuẩn của FastAPI: chia thành `routers/` (chứa các API endpoint), `services/` (chứa logic gọi LLM, gọi API nội bộ), `utils/` (chứa thuật toán RRF, xử lý chuỗi frame_id).
- Hãy viết code module hóa để sau này dễ dàng mở rộng khi hệ thống phình to.
