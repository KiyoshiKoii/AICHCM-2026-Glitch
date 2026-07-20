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

## 📚 Tài liệu Tham khảo (Reference Papers)
- **Leveraging LLMs and Generative Models for Interactive Known-Item Video Search**: Hướng dẫn dùng LLM để viết lại câu và mở rộng ngữ nghĩa tránh lỗi Out-of-vocabulary, kèm chiến lược chống ảo giác (hallucination).
- **LLandMark: A Multi-Agent Framework...**: Thiết kế kiến trúc Multi-Agent để phân rã câu hỏi (parsing & planning) và tách các trường logic độc lập.
- **Reciprocal Rank Fusion outperforms condorcet...**: Bài báo nền tảng về thuật toán RRF, giải thích lý do dùng công thức $1/(60+rank)$ thay vì cộng điểm thô.
- **Exploring the Best Practices of Query Expansion with LLMs**: Nghiên cứu cách LLM tự viết lại và mở rộng câu truy vấn, giới thiệu toolkit QueryGym.
- **NTCIR-18 Lifelog 6 Task papers**: Tham khảo cách các đội quốc tế dùng LLM trích xuất thực thể và tạo prompt cho mô hình Vision-Language.

## 🌿 Hướng dẫn Git & Đặt tên Branch (Git Workflow & Branching)
Mỗi dev sẽ có một branch chính để làm việc. Branch chính của bạn là: **`feat/backend-services`**.

- **Quy tắc đặt tên nếu bạn tạo thêm branch phụ (mở rộng tính năng/sửa bug):**
  - Cú pháp: `feat/backend-services-<loại>-<tên_chức_năng>`
  - Ví dụ thêm tính năng: `feat/backend-services-feature-yolo-model`
  - Ví dụ sửa lỗi: `feat/backend-services-bugfix-vector-dim`
- Hãy nhớ **push code thường xuyên** lên branch của mình trên GitHub để backup.
