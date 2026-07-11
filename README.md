# 🏆 AIC 2026: Multimedia Retrieval Project

## 📌 Tổng quan Cuộc thi (Overview)
Cuộc thi Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM (AIC) năm 2026 tập trung vào mảng truy xuất video nhiều giờ (Multimedia Retrieval). Trọng tâm dữ liệu đã dịch chuyển từ hệ thống giám sát công cộng (Surveillance) sang các thiết bị ghi hình cá nhân (Sousveillance - wearable cameras, lifelogging) với góc nhìn thứ nhất (Ego-centric/POV). 

Dữ liệu đặc thù mang tính phi cấu trúc (Unstructured), khối lượng lớn (Big Data), chứa nhiều thông tin nhiễu nhưng giàu ngữ nghĩa cá nhân.

## 🎯 Các dạng bài toán chính (Core Tasks)
Dự án cần giải quyết 4 tác vụ cốt lõi:

| Tác vụ | Mô tả | Đầu ra yêu cầu (Expected Output) |
| :--- | :--- | :--- |
| **KIS** (Known-Item Search) | Tìm duy nhất một khoảnh khắc cụ thể (Ground Truth) dựa trên mô tả hoặc hình ảnh. Có 2 dạng: Video KIS và Textual KIS. | Đoạn video hoặc Keyframe chính xác (ưu tiên Top-1, Top-5). |
| **AVS** (Ad-hoc Video Search) | Tìm kiếm trên diện rộng tất cả các phân cảnh thỏa mãn ngữ nghĩa mô tả. | Danh sách phân đoạn video xếp hạng theo độ tương đồng ngữ nghĩa. |
| **VQA** (Video Question Answering) | Hỏi đáp dựa trên video yêu cầu khả năng đếm (counting) và suy luận theo thời gian (temporal reasoning). | Câu trả lời (Text) ngắn gọn, chính xác. |
| **KISC** (Conversational KIS) | Tìm kiếm thông qua hội thoại tương tác liên tục, AI Agent chủ động làm rõ ý định (Clarification). | Kết quả chính xác nhất sau khi đã thu hẹp phạm vi tìm kiếm. |

## 🚧 Thách thức Kỹ thuật (Technical Challenges)
* **Semantic gap** (Khoảng cách ngữ nghĩa): Sự chênh lệch giữa ngôn ngữ tự nhiên của con người và điểm ảnh thô máy tính nhận diện. Đòi hỏi sử dụng các mô hình Large Vision Language Models (LVLM) để lý luận.
* **Data sparsity** (Sự thưa thớt dữ liệu): Khoảnh khắc cần tìm (KIS) thường chỉ kéo dài 2-3 giây, ẩn trong hàng nghìn giờ video. Bắt buộc phải có bộ lọc thô cực nhanh (Filtering).
* **Temporal logic constraints** (Ràng buộc logic thời gian): Đòi hỏi AI phân biệt được trật tự trước - sau của chuỗi hành động.

---

## 👥 Thông tin Nhóm (Team Members)

| STT | Họ và Tên | Vai trò trong hệ thống AI / Data Science |
| :--- | :--- | :--- |
| 1 | Phi Anh Khôi | Trưởng nhóm / Architect |
| 2 | Đoàn Khánh Như | Computer Vision Engineer |
| 3 | Nguyễn Thị Yến Nhi | NLP Engineer |
| 4 | Võ Thành Đạt | LLM Engineer  |
| 5 | Lâm Vĩnh Khang | Full-Stack Engineer |

---

# 🏆 AIC 2026: Multimedia Retrieval Project

## 📌 Overview
The Ho Chi Minh City Artificial Intelligence Challenge (AIC) 2026 focuses on the field of multi-hour video retrieval (Multimedia Retrieval). The data focus has shifted from public surveillance systems to personal recording devices (Sousveillance - wearable cameras, lifelogging) with a first-person perspective (Ego-centric/POV). 

The specific data is unstructured, large in volume (Big Data), contains a lot of noise but is rich in personal semantics.

## 🎯 Core Tasks
The project needs to solve 4 core tasks:

| Task | Description | Expected Output |
| :--- | :--- | :--- |
| **KIS** (Known-Item Search) | Find exactly one specific moment (Ground Truth) based on a description or image. There are 2 types: Video KIS and Textual KIS. | Exact video segment or Keyframe (Top-1, Top-5 preferred). |
| **AVS** (Ad-hoc Video Search) | Broad search for all scenes that satisfy the semantic description. | List of video segments ranked by semantic similarity. |
| **VQA** (Video Question Answering) | Video-based Q&A requiring counting and temporal reasoning capabilities. | Concise, accurate answer (Text). |
| **KISC** (Conversational KIS) | Search through continuous interactive conversation, the AI Agent proactively clarifies intent (Clarification). | The most accurate result after narrowing down the search scope. |

## 🚧 Technical Challenges
* **Semantic gap**: The disparity between natural human language and raw pixels recognized by computers. Requires the use of Large Vision Language Models (LVLM) for reasoning.
* **Data sparsity**: The moment to find (KIS) usually lasts only 2-3 seconds, hidden in thousands of hours of video. Extremely fast coarse filtering is mandatory.
* **Temporal logic constraints**: Requires AI to distinguish the before - after order of an action sequence.

---

## 👥 Team Members

| No. | Full Name | Role in AI / Data Science system |
| :--- | :--- | :--- |
| 1 | Phi Anh Khoi | Team Leader / Architect |
| 2 | Doan Khanh Nhu | Computer Vision Engineer |
| 3 | Nguyen Thi Yen Nhi | NLP Engineer |
| 4 | Vo Thanh Dat | LLM Engineer  |
| 5 | Lam Vinh Khang | Full-Stack Engineer |