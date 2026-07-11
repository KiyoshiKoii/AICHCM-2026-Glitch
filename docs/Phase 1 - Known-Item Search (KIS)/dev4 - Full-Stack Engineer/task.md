# Tasks for Dev 4 (Full-Stack Engineer / Frontend)

Dưới đây là các hạng mục công việc cần hoàn thiện cho Phase 2 (Online Serving):

- [ ] **Search UI Form**: Xây dựng thanh tìm kiếm hỗ trợ nhập Text (Textual KIS) và Upload Image (Video KIS).
- [ ] **Result Grid rendering**: Vẽ lưới giao diện hiển thị các kết quả trả về từ Backend (giảm số lượng xuống khoảng 20 tấm ảnh để cân đối và tối ưu).
- [ ] **Event Handling**: Lắng nghe sự kiện Double-click vào một bức ảnh bất kỳ trên lưới kết quả, từ đó kích hoạt gọi API `GET /frames/context`.
- [ ] **Temporal Timeline Viewer**: Dựng Component hiển thị cuộn ngang (chuỗi 11 frames) cho phép user dùng chuột kéo qua lại (Drag to scroll) để xem diễn biến của hành động (VD: rơi ví).
- [ ] **Mock Testing & UI Testing**: Sử dụng file dữ liệu mẫu (`mock/search_response_v1.json`) để test giao diện độc lập. Viết Unit Test cho các Component chính (như thanh cuộn ngang, lưới ảnh) nhằm đảm bảo Responsive Layout không bị vỡ.

## 📂 Hướng dẫn Tổ chức Thư mục & Code (Codebase Guidelines)
Toàn bộ source code của bạn sẽ được phát triển trong thư mục **`src/frontend/`**.
- Khuyến nghị sử dụng các Component có khả năng tái sử dụng cao (VD: `ImageCard.jsx`, `TimelineViewer.jsx`).
- Tuân thủ nguyên tắc Graceful Degradation đã được thống nhất: UI không được lỗi (crash) khi data bị thiếu các trường không bắt buộc (dùng optional chaining `?.`).
- Các file giả lập dữ liệu (như `search_response_v1.json`) nên được để gọn gàng trong thư mục `src/frontend/mock/`.

## 🌿 Hướng dẫn Git & Đặt tên Branch (Git Workflow & Branching)
Mỗi dev sẽ có một branch chính để làm việc. Branch chính của bạn là: **`feat/frontend-ui`**.

- **Quy tắc đặt tên nếu bạn tạo thêm branch phụ (mở rộng tính năng/sửa bug):**
  - Cú pháp: `feat/frontend-ui-<loại>-<tên_chức_năng>`
  - Ví dụ thêm tính năng: `feat/frontend-ui-feature-layout`
  - Ví dụ sửa lỗi: `feat/frontend-ui-bugfix-timeline`
- Hãy nhớ **push code thường xuyên** lên branch của mình trên GitHub để backup.
