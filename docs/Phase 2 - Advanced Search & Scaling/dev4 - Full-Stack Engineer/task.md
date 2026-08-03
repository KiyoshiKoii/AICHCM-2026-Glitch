# Tasks for Dev 4 (Full-Stack Engineer / Data Pipeline) - Phase 2

Trong Phase 2, Dev 4 tạm thời hoãn các công việc làm giao diện Frontend (UI/UX) để tập trung vào nhiệm vụ tiền xử lý âm thanh/lời thoại từ video (ASR - Automatic Speech Recognition).

---
## 🎯 Task: Trích xuất Script/Phụ đề (ASR) từ Video

- **Nhiệm vụ:** Viết script Python (khuyến khích dùng thư viện `faster-whisper` hoặc `openai-whisper`) để quét qua tất cả các file Video gốc trong tập dữ liệu, trích xuất toàn bộ lời thoại (transcript) tiếng Việt / tiếng Anh.
- **Nơi lưu trữ Output:** `data/metadata/metadata_asr.json`

### 📄 Quy định Format Output (`data/metadata/metadata_asr.json`)
File output bắt buộc phải ở định dạng JSON array, mỗi element đại diện cho 1 video:

```json
[
  {
    "video_name": "L21_V001",
    "full_transcript": "Chào mừng các bạn đến với bản tin thời sự 60 giây hôm nay...",
    "segments": [
      {
        "start": 0.0,
        "end": 4.5,
        "text": "Chào mừng các bạn đến với bản tin thời sự"
      },
      {
        "start": 4.5,
        "end": 8.2,
        "text": "60 giây hôm nay"
      }
    ]
  }
]
```

### ⚠️ Lưu ý kỹ thuật:
1. `video_name`: Bắt buộc chuẩn hóa tên video (Ví dụ: `L21_V001`), khớp với thư mục trong `data/keyframes/`.
2. `full_transcript`: Chuỗi tổng hợp toàn bộ lời thoại trong video để Dev 2 nạp thẳng vào BM25 / Elasticsearch.
3. `segments`: Danh sách các câu thoại kèm timestamp (`start`, `end` tính bằng giây) để hỗ trợ tìm kiếm mốc thời gian chính xác sau này.  
