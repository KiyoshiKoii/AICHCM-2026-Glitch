# Architecture Overview - Phase 1
Tài liệu này tổng hợp kiến trúc luồng dữ liệu của hệ thống Truy xuất Video đa phương thức (AICHCM-2026-Glitch). Hệ thống chia làm 2 pha chính: Ingestion (xử lý dữ liệu offline) và Serving (chạy thực tế lúc thi).

## ⚙️ PHA 1: OFFLINE INGESTION (Xử lý dữ liệu đầu vào)
*(Đây là bước làm trước khi đi thi, cày nát 100GB Video của BTC)*

- **Chuẩn hóa Frame**: Cả team dùng chung 1 script cắt video ra thành các ảnh tĩnh. Đặt tên chuẩn hóa: `[TênVideo]_f[SốThứTự].jpg` (VD: `vid05_f1024.jpg`). Lưu chung vào 1 thư mục ổ cứng.
- **Luồng Dev 1 (Thị giác)**: Quét qua thư mục ảnh $\rightarrow$ Đưa vào CLIP lấy Vector $\rightarrow$ Lưu vào Qdrant với khóa chính là `frame_id`.
- **Luồng Dev 2 (Ngữ nghĩa)**: Quét qua thư mục ảnh $\rightarrow$ Đưa vào Florence-2 lấy mô tả (Caption) và chữ (OCR) $\rightarrow$ Lưu vào Elasticsearch với khóa chính là `frame_id`.

## 🚀 PHA 2: ONLINE SERVING (Luồng chạy thực tế lúc thi KIS)
*Đây là xương sống của hệ thống lúc đi thi.*

### Luồng 1: Truy vấn bằng Văn bản (Textual KIS)
1. Giám khảo đọc đề "Tìm cảnh rơi ví". Dev 4 gõ vào giao diện Web.
2. Web (Dev 4) bắn API `POST /search/text` sang cho Dev 3.
3. Dev 3 quăng câu đó vào LLM. LLM nhả ra JSON gồm: Câu prompt tiếng Anh (Cho Visual) và Bộ từ khóa đồng nghĩa (Cho Semantic).
4. Dev 3 bắn 2 API nội bộ gọi Dev 1 và Dev 2 tìm kiếm song song (top_k=200).
5. Dev 1 và Dev 2 trả về 2 mảng điểm. Dev 3 dùng công thức RRF trộn lại, cắt lấy Top 20 (hoặc số lượng nhỏ) để tối ưu UI. Nhét thêm cái link URL ảnh vào rồi trả về cho Dev 4.
6. Giao diện Web (Dev 4) load danh sách tấm ảnh đó lên màn hình.

### Luồng 2: Truy vấn bằng Hình ảnh (Video KIS)
- Khác Luồng 1 ở chỗ: Dev 4 upload thẳng file ảnh $\rightarrow$ Dev 3 chuyển thẳng file đó cho Dev 1 để search trong Vector DB (Bỏ qua khâu LLM và Dev 2).

### Luồng 3: Xem ngữ cảnh thời gian (Temporal Timeline)
1. Dev 4 thấy 1 tấm ảnh nghi ngờ là Ground Truth (đáp án), bèn click đúp vào nó.
2. Web bắn API `GET /frames/context/{frame_id}?window=5` sang Dev 3.
3. Dev 3 tách cái chuỗi `frame_id` ra, cộng trừ số thứ tự để lấy ra 5 frame trước và 5 frame sau. Trả link URL về cho Web.
4. Giao diện Web (Dev 4) xếp 11 tấm ảnh đó thành hàng ngang. Dev 4 dùng chuột kéo qua lại để xem hành động diễn ra như thế nào trước khi bấm nộp bài.
