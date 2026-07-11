# Hướng dẫn Mock Testing cho Frontend (Dev 4)

Tài liệu này hướng dẫn cách tích hợp file mock data để phát triển UI trước khi Backend hoàn thiện API.

## 1. Import Mock Data
Bạn có thể import trực tiếp file JSON vào React Component:
```javascript
import searchMockData from '../../src/mock/search_response_v1.json';
```

## 2. Giả lập API Call với Delay (Loading State)
Để UI có thể test được các trạng thái Loading (Spinner), hãy viết hàm fetch giả lập sử dụng `setTimeout` với độ trễ khoảng 1 giây:

```javascript
export const fetchSearchResults = async (query) => {
  return new Promise((resolve) => {
    // Giả lập độ trễ mạng 1 giây
    setTimeout(() => {
      resolve(searchMockData);
    }, 1000);
  });
};
```
Sử dụng hàm này trong component để set state `isLoading = true` trước khi gọi và `false` sau khi resolve.

## 3. Nguyên tắc Graceful Degradation
Khi binding data lên UI, **chỉ phụ thuộc vào các trường MUST-HAVE**:
- `frame_id`
- `thumbnail_url`

Đối với các trường `metadata` hoặc các trường optional khác, UI không được crash nếu dữ liệu bị thiếu. Luôn sử dụng Optional Chaining (`?.`) hoặc Default Fallback (`||` / `??`):

```javascript
// Tốt: UI sẽ không crash nếu không có metadata hoặc video_id
const videoId = result.metadata?.video_id || "Unknown Video";

// Xấu: Có thể gây lỗi "Cannot read properties of undefined"
// const videoId = result.metadata.video_id;
```
