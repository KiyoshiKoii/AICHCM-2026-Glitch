# Tasks for Dev 3 (LLM Engineer & Backend Integration) - Phase 2

Trong Phase 2, Dev 3 đóng vai trò là "Tổng tư lệnh" đứng giữa, chịu trách nhiệm kết hợp LLM thế hệ mới và đảm bảo hệ thống chịu được lượng request dồn dập lúc thi.

*(Đã cập nhật theo bài toán VQA và định hướng xử lý Top kết quả từ vòng sơ tuyển AIC 2026)*

- [ ] **Task 1: LLM Re-ranking & Verification (Kiểm duyệt Top 10)**
  - Sau khi hệ thống truy xuất thô trả về Top 50 hình ảnh/khoảnh khắc tiềm năng nhất, xây dựng luồng pipeline đưa Top 10 ảnh đầu tiên qua mô hình Vision LLM.
  - Prompt cho LLM đối chiếu kỹ hình ảnh với mô tả truy vấn ban đầu để loại bỏ các "false positive" (nhận diện sai) và xếp hạng lại (re-rank) sao cho kết quả chuẩn xác nhất vươn lên Top 1.

- [ ] **Task 2: Trích xuất câu trả lời cho bài toán VQA (Visual Question Answering)**
  - Xử lý Truy vấn dạng 2 của vòng sơ tuyển: Vừa tìm khoảnh khắc, vừa trả lời câu hỏi chi tiết.
  - Thiết kế prompt để Vision LLM dựa vào các khung hình Top đầu phân tích thông tin cụ thể (VD: đếm số lượng, màu sắc, hành động,...) và sinh ra câu trả lời cực kỳ ngắn gọn (bằng tiếng Việt hoặc tiếng Anh) đúng chuẩn định dạng đầu ra của BTC.

- [ ] **Task 3: Tích hợp API các mô hình Vision LLM thương mại (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet)**
  - Thay vì chạy các mô hình local (như LLaVA, InternVideo2) vốn nặng và suy luận chưa đủ sâu cho VQA phức tạp, xây dựng module gọi API trực tiếp đến các provider.
  - Xử lý các vấn đề kĩ thuật khi gọi API: resize/compress hình ảnh trước khi gửi (base64) để tiết kiệm token và thời gian; quản lý Rate Limit (giới hạn request); và áp dụng xử lý bất đồng bộ (Asynchronous calls/Concurrency) để tăng tốc độ phản hồi lúc thi đấu.
