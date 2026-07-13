# Tasks for Dev 3 (LLM Engineer / Backend)

Dưới đây là các hạng mục công việc cần hoàn thiện cho (Online Serving) - Phase 1:

## 📂 Kiến trúc & Tổ chức Source Code (Codebase Guidelines)

Toàn bộ source code của bạn sẽ được phát triển trong thư mục **`src/backend/`**. 
Cấu trúc đã được tổ chức theo từng Domain cụ thể để dễ bảo trì và mở rộng cho các Phase tiếp theo:

```text
src/backend/
├── main.py                          # FastAPI app factory + mount static files
├── config.py                        # Cấu hình tập trung (ports, model names, paths)
├── routers/                         # Các API endpoint (thin layer, chỉ nhận/trả request)
├── services/                        # Business logic chính
├── clients/                         # HTTP clients gọi các service nội bộ
├── schemas/                         # Pydantic models cho request/response
└── utils/                           # Hàm tiện ích thuần túy (stateless, dễ unit test)
```

## ✅ Checklist Công việc

### 1. Phân tích Truy vấn & Tích hợp LLM (`services/query_analyzer.py`)
- [ ] **Nghiên cứu & Tích hợp LLM Parser**: Đánh giá và tìm hiểu các mô hình ngôn ngữ (bao gồm cả các bản trả phí như GPT-4, Claude,... để đảm bảo chất lượng). Tạm thời dùng Ollama tự host ở local.
- [ ] Viết hàm nhận câu query tiếng Việt -> Nhả JSON gồm: Prompt tiếng Anh (Visual) & Keywords đồng nghĩa (Semantic).

### 2. Giao tiếp Dịch vụ Nội bộ (`clients/`)
- [ ] **`visual_client.py`**: Viết hàm HTTP POST gọi sang `localhost:8001/internal/search/visual` (Dev 1).
- [ ] **`semantic_client.py`**: Viết hàm HTTP POST gọi sang `localhost:8002/internal/search/text` (Dev 2).

### 3. Điều phối Tìm kiếm (`services/search_orchestrator.py` & `utils/rrf.py`)
- [ ] Cài đặt thuật toán Reciprocal Rank Fusion (RRF) trong `utils/rrf.py` (công thức $1/(60+rank)$).
- [ ] Tại `search_orchestrator.py`: Viết logic gọi song song 2 clients trên, nhận kết quả, dùng RRF trộn điểm, gộp kết quả, cắt lấy Top 20 và gán URL thumbnail.

### 4. API Endpoints (`routers/`)
- [ ] Định nghĩa Pydantic Models trong `schemas/search.py` và `schemas/frames.py` bám sát tài liệu API Contract.
- [ ] **`routers/search.py`**: 
  - `POST /search/text`: Nhận query văn bản -> Gọi `query_analyzer` -> Gọi `search_orchestrator` -> Trả về kết quả.
  - `POST /search/image`: Forward trực tiếp ảnh upload sang Dev 1 (Vector DB pipeline).
- [ ] **`routers/frames.py`**:
  - `GET /frames/context/{frame_id}`: Xử lý logic chuỗi, tách số thứ tự trong `frame_id` (VD: `vid05_f1024` -> `1024`), cộng trừ để lấy ra mảng 11 frames liền kề (5 trước, 5 sau). Viết hàm tiện ích xử lý frame_id trong `utils/frame_id.py`.

### 5. Kiểm thử (Unit Testing)
- [ ] Xây dựng Unit test cho các function cốt lõi (đặc biệt là hàm tính điểm RRF và module phân tích LLM JSON parser) để đảm bảo không bị lỗi dữ liệu đầu ra.

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
