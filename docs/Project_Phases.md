# 🚀 Lộ trình Phát triển Dự án (Project Phases)

Dựa trên 4 bài toán cốt lõi của cuộc thi AICHCM 2026, toàn bộ quá trình phát triển hệ thống được chia thành 4 Giai đoạn (Phases) nối tiếp và kế thừa lẫn nhau.

## 📍 Phase 1: Known-Item Search (KIS) - Nền tảng Tìm kiếm Cốt lõi
* **Mục tiêu**: Xây dựng bộ khung sườn (Retrieval Pipeline) vững chắc nhất để tìm ra duy nhất 1 khoảnh khắc cụ thể (Ground Truth) dựa trên mô tả văn bản hoặc hình ảnh.
* **Các thành phần cốt lõi (MVP)**:
  - **Visual Pipeline**: Dùng CLIP (hoặc SigLIP) trích xuất đặc trưng hình ảnh thành Vector và lưu vào Vector DB (Qdrant).
  - **Semantic Pipeline**: Dùng VLM (như Florence-2) tự động sinh Caption/OCR từ ảnh và lưu vào Text DB (BM25/Elasticsearch).
  - **Backend & Orchesration**: Sử dụng RRF (Reciprocal Rank Fusion) để gộp kết quả chấm điểm từ 2 pipeline trên.
* **Vai trò**: Đây là phase nền tảng sống còn. Nếu công cụ lọc KIS không nhanh và chuẩn xác, các bài toán phức tạp ở sau sẽ sụp đổ vì tốc độ quét dữ liệu thô quá chậm.

## 📍 Phase 2: Ad-hoc Video Search (AVS) - Tìm kiếm Diện rộng theo Ngữ nghĩa
* **Mục tiêu**: Mở rộng khả năng truy xuất từ "1 mục tiêu cụ thể" thành "tất cả các phân cảnh thỏa mãn một sự kiện hay hành động khái quát".
* **Sự kế thừa từ Phase 1**: Tái sử dụng lại toàn bộ cấu trúc Database (Vector + Text) và API truy xuất của Phase 1.
* **Công việc nâng cấp**:
  - Sử dụng LLM để viết lại truy vấn (Query Expansion/Paraphrasing) từ mô tả ngắn gọn thành các từ khóa đồng nghĩa, giải quyết lỗi Out-of-Vocabulary.
  - Tinh chỉnh thuật toán xếp hạng lại (Re-ranking) để đưa các video chứa cùng một chủ đề lên Top kết quả thay vì chỉ thiên vị một mục tiêu duy nhất.

## 📍 Phase 3: Video Question Answering (VQA) - Suy luận Ràng buộc
* **Mục tiêu**: Không chỉ tìm kiếm, hệ thống phải trả lời được các câu hỏi phức tạp yêu cầu đếm số lượng (Counting) hoặc nhận thức logic thời gian (Temporal reasoning). (VD: "Có bao nhiêu người bước vào sau khi chiếc xe dừng lại?").
* **Sự kế thừa từ Phase 2**: Dùng hệ thống AVS ở Phase 2 để lọc thô ra Top 5-10 video tiềm năng nhất.
* **Công việc nâng cấp**:
  - Không dừng ở việc trả về ID video, hệ thống phải bơm trực tiếp các video tiềm năng này vào các mô hình **Large Vision Language Models (LVLM)** mạnh mẽ hơn (như LLaVA, InternVideo2) để bắt máy đọc chi tiết từng frame ảnh và sinh ra câu trả lời chữ (Text).

## 📍 Phase 4: Conversational KIS (KISC) - Tương tác Agent Chủ động
* **Mục tiêu**: Đưa hệ thống lên mức độ tự trị hóa thành một AI Agent. Khi câu query của người dùng quá mập mờ, Agent phải có khả năng phản hồi hỏi ngược lại (Clarification) để thu hẹp dần kết quả.
* **Sự kế thừa từ Phase 3**: Vận dụng bộ máy truy xuất (Phase 1+2) và khả năng sinh ngôn ngữ/suy luận tự nhiên (Phase 3).
* **Công việc nâng cấp**:
  - Xây dựng luồng hội thoại đa vòng (Multi-turn conversation).
  - Tích hợp bộ nhớ (Memory) và lịch sử ngữ cảnh (Chat History).
  - Frontend biến đổi từ giao diện tìm kiếm 1 chiều sang giao diện Chatbot kết hợp hiển thị Video động (Real-time grid update).

---
**Tóm lược:**
Dự án được thăng cấp dần từ hệ thống **Truy xuất Mù** (Phase 1, 2) $\rightarrow$ **Suy luận Có ý thức** (Phase 3) $\rightarrow$ **Giao tiếp Chủ động** (Phase 4).
