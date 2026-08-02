# Công việc của Dev 3 (Kỹ sư LLM / Backend)

## Quy ước luồng dữ liệu: Frontend → Backend → các Pipeline

Dev 3 phụ trách lớp Backend phục vụ Frontend. Dev 4 chỉ gọi Backend tại
`http://localhost:8000/api/v1`; không gọi trực tiếp Dev 1 (`:8001`), Dev 2
(`:8002`), Qdrant hay Elasticsearch. Backend chịu trách nhiệm kiểm tra đầu
vào, giới hạn thời gian chờ, quy đổi lỗi, chuẩn hoá kết quả và tạo URL ảnh.

```text
Dev 4 (giao diện)
  | POST /api/v1/search/text | POST /api/v1/search/image
  | GET  /api/v1/frames/context/{frame_id}
  v
Dev 3 (Backend :8000)
  |- query_analyzer: phân tích truy vấn chữ
  |- visual_client: gọi Dev 1 (:8001)
  |- semantic_client: gọi Dev 2 (:8002)
  `- static media: /media/thumbnails/{frame_id}.jpg
```

### Quy ước công khai Dev 3 phải tuân thủ

- URL gốc của giao diện phải lấy từ cấu hình, ví dụ
  `VITE_API_BASE_URL=http://localhost:8000/api/v1`; không ghi cứng URL của
  pipeline trong giao diện.
- `frame_id` (ví dụ `vid05_f1024`) là khoá chung, không được đổi format từ lúc
  nhận kết quả pipeline cho đến khi trả về giao diện.
- `top_k` do giao diện gửi là **số kết quả cuối cùng** cần hiển thị; dùng đúng
  trường và cách gọi được mô tả trong API Contract (ví dụ `top_k=50`). Backend
  có thể xin `candidate_k=200` từ từng pipeline trước khi gộp, rồi cắt còn
  `top_k`.
- `thumbnail_url` trả về theo đúng API Contract là URL tương đối, ví dụ
  `/media/thumbnails/vid05_f1024.jpg`. Giao diện phải ghép URL này với địa chỉ
  Backend, không ghép với địa chỉ của máy chủ giao diện.
- CORS chỉ cho phép origin của giao diện (thường là `http://localhost:5173`);
  không dùng wildcard origin cùng credentials.

### Luồng A: Tìm kiếm KIS bằng văn bản — `POST /search/text`

1. Giao diện gửi JSON `{ "query": "...", "top_k": 50 }` với
   `Content-Type: application/json`; giao diện không tự tạo prompt tiếng Anh
   hay keyword từ LLM.
2. `routers/search.py` kiểm tra `query` không rỗng và `top_k` hợp lệ, rồi chỉ
   chuyển việc xử lý. Router không chứa logic LLM, RRF hoặc gọi HTTP trực tiếp.
3. `services/query_analyzer.py` đổi truy vấn tiếng Việt thành object hợp lệ gồm
   `visual_prompt` (prompt tiếng Anh) và `keywords` (mảng từ khoá đồng nghĩa).
   Nếu không parse được JSON của LLM thì dùng fallback an toàn từ query gốc;
   không trả JSON thô của LLM về giao diện.
4. `services/search_orchestrator.py` gọi song song Dev 1 bằng `visual_prompt`
   và Dev 2 bằng `keywords`, mỗi bên với `candidate_k=200`.
5. `utils/rrf.py` gộp kết quả theo `frame_id`, tính `1 / (60 + rank)`, sắp xếp
   giảm dần và cắt theo `top_k`. Không cộng trực tiếp điểm CLIP và BM25 vì hai
   thang điểm không cùng đơn vị.
6. Backend thêm `thumbnail_url`, đóng gói theo `SearchResponse` trong API
   Contract rồi trả về. Giao diện chỉ render `data.results`.

### Luồng B: Tìm kiếm KIS bằng hình ảnh — `POST /search/image`

1. Giao diện gửi `multipart/form-data`: `image_file` bắt buộc và `top_k` tuỳ
   chọn. Khi dùng `FormData`, không tự đặt `Content-Type`; trình duyệt sẽ tự thêm
   multipart boundary.
2. Backend kiểm tra đuôi file (`.jpg`, `.jpeg`, `.png`), loại MIME và dung lượng
   tối đa cấu hình được; trả `400`, `413` hoặc `415` nếu không hợp lệ.
3. Backend chuyển file sang chức năng tìm kiếm ảnh của Dev 1 bằng
   `visual_client`; không gọi LLM hoặc Dev 2. Chi tiết giao tiếp nội bộ này do
   Dev 1 và Dev 3 thống nhất trong phần triển khai, không tạo thêm API public:
   API công khai duy nhất cho image query vẫn là **API số 2**.
4. Backend chuẩn hoá kết quả Dev 1, cắt theo `top_k`, thêm `thumbnail_url` và
   trả đúng `SearchResponse` như text search.

### Luồng C: Xem ngữ cảnh theo thời gian — `GET /frames/context/{frame_id}?window=5`

1. Khi double-click một kết quả, giao diện gửi nguyên `frame_id` đã nhận từ
   search response.
2. Backend kiểm tra ID và `window`; `frame_navigator` dùng `utils/frame_id.py`
   để tạo danh sách frame trước/giữa/sau.
3. Endpoint này không gọi LLM hay pipeline. Ở đầu/cuối video, chỉ trả frame tồn
   tại; không được sinh ID âm hoặc vượt giới hạn.

### Quy tắc lỗi và kiểm thử

- Client nội bộ phải có timeout cấu hình được và log `request_id`, pipeline,
  thời gian xử lý, loại lỗi; không log nội dung ảnh upload.
- Khi một nguồn trong text search lỗi hoặc timeout, Dev 3 phải xử lý theo error
  response đã thống nhất với team; không tự thêm trường response mới ngoài API
  Contract. Khi cả hai nguồn lỗi, không trả `200` với danh sách rỗng không rõ
  nguyên nhân.
- Không trả stack trace, URL nội bộ, credentials hoặc raw model output cho UI.
- Unit test fallback parser, RRF, ánh xạ response và biên frame. Integration
  test router bằng HTTP client giả; không yêu cầu chạy model, Qdrant hay
  Elasticsearch thật.

Dưới đây là các hạng mục công việc cần hoàn thiện cho phần phục vụ trực tuyến - Phase 1:

## 📂 Kiến trúc & Tổ chức mã nguồn

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
- [x] **Nghiên cứu & Tích hợp LLM Parser**: Đánh giá và tìm hiểu các mô hình ngôn ngữ (bao gồm cả các bản trả phí như GPT-4, Claude,... để đảm bảo chất lượng). Tạm thời dùng Ollama tự host ở local.
- [x] Viết hàm nhận câu query tiếng Việt -> Nhả JSON gồm: Prompt tiếng Anh (Visual) & Keywords đồng nghĩa (Semantic).

### 2. Giao tiếp Dịch vụ Nội bộ (`clients/`)
- [x] **`visual_client.py`**: Viết client HTTP gọi Dev 1 cho cả truy vấn chữ và truy vấn ảnh; có timeout, quy đổi lỗi và chuẩn hoá response. Chi tiết endpoint nội bộ cần thống nhất với Dev 1, không thêm API public mới.
- [x] **`semantic_client.py`**: Viết client HTTP POST gọi `localhost:8002/internal/search/text`; có timeout, quy đổi lỗi và chuẩn hoá response.

### 3. Điều phối Tìm kiếm (`services/search_orchestrator.py` & `utils/rrf.py`)
- [x] Cài đặt thuật toán Reciprocal Rank Fusion (RRF) trong `utils/rrf.py` (công thức $1/(60+rank)$).
- [x] Tại `search_orchestrator.py`: Với text search, gọi song song 2 client, dùng RRF theo `frame_id`, gộp kết quả, cắt theo `top_k` giao diện yêu cầu và gắn URL thumbnail theo API Contract.

### 4. Các endpoint API (`routers/`)
- [x] Định nghĩa Pydantic Models trong `schemas/search.py` và `schemas/frames.py` bám sát tài liệu API Contract.
- [x] **`routers/search.py`**: 
  - `POST /search/text`: Nhận query văn bản -> Gọi `query_analyzer` -> Gọi `search_orchestrator` -> Trả về kết quả.
  - `POST /search/image`: Forward trực tiếp ảnh upload sang Dev 1 (Vector DB pipeline).
- [x] **`routers/frames.py`**:
  - `GET /frames/context/{frame_id}`: Xử lý logic chuỗi, tách số thứ tự trong `frame_id` (VD: `vid05_f1024` -> `1024`), cộng trừ để lấy ra mảng 11 frames liền kề (5 trước, 5 sau). Viết hàm tiện ích xử lý frame_id trong `utils/frame_id.py`.

### 5. Kiểm thử đơn vị
- [x] Xây dựng Unit test cho các function cốt lõi (đặc biệt là hàm tính điểm RRF và module phân tích LLM JSON parser) để đảm bảo không bị lỗi dữ liệu đầu ra.

## 📚 Tài liệu tham khảo
- **Leveraging LLMs and Generative Models for Interactive Known-Item Video Search**: Hướng dẫn dùng LLM để viết lại câu và mở rộng ngữ nghĩa tránh lỗi Out-of-vocabulary, kèm chiến lược chống ảo giác (hallucination).
- **LLandMark: A Multi-Agent Framework...**: Thiết kế kiến trúc Multi-Agent để phân rã câu hỏi (parsing & planning) và tách các trường logic độc lập.
- **Reciprocal Rank Fusion outperforms condorcet...**: Bài báo nền tảng về thuật toán RRF, giải thích lý do dùng công thức $1/(60+rank)$ thay vì cộng điểm thô.
- **Exploring the Best Practices of Query Expansion with LLMs**: Nghiên cứu cách LLM tự viết lại và mở rộng câu truy vấn, giới thiệu toolkit QueryGym.
- **NTCIR-18 Lifelog 6 Task papers**: Tham khảo cách các đội quốc tế dùng LLM trích xuất thực thể và tạo prompt cho mô hình Vision-Language.

## 🌿 Hướng dẫn Git & Đặt tên nhánh
Mỗi dev sẽ có một branch chính để làm việc. Branch chính của bạn là: **`feat/backend-services`**.

- **Quy tắc đặt tên nếu bạn tạo thêm branch phụ (mở rộng tính năng/sửa bug):**
  - Cú pháp: `feat/backend-services-<loại>-<tên_chức_năng>`
  - Ví dụ thêm tính năng: `feat/backend-services-feature-yolo-model`
  - Ví dụ sửa lỗi: `feat/backend-services-bugfix-vector-dim`
- Hãy nhớ **push code thường xuyên** lên branch của mình trên GitHub để backup.
