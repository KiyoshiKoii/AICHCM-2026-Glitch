# Cấu trúc Thư mục Dữ liệu (Data Structure)

Thư mục `data/` là nơi chứa toàn bộ dữ liệu thô (ảnh) và các metadata (JSON) được sinh ra từ các luồng trích xuất của hệ thống (Pipeline).
**Lưu ý:** Các thư mục `data/keyframes/` (ảnh thô), `data/objects/` (JSON vật thể rác/lẻ), `data/videos/` (video gốc), `data/audio_cache/` (WAV tạm) và `data/models/` (model đã convert) đã được chặn trong `.gitignore`, các file metadata chính khác vẫn được phép push lên Git.

## Cấu trúc chi tiết

```text
data/
├── keyframes/                  # Chứa toàn bộ hình ảnh (Frames) được cắt ra từ Video
│   ├── L21_V001/               # Thư mục Video (Tập L21, Video 001)
│   │   ├── 001.jpg             # Tên file ảnh (ID tương ứng: L21_V001_f0001)
│   │   ├── 017.jpg             # Tên file ảnh (ID tương ứng: L21_V001_f0017)
│   │   └── ...
│   └── L21_V002/
│       └── ...
├── videos/                     # Chứa video gốc (VD: L21_V001.mp4) — input cho ASR (Dev 4)
├── audio_cache/                # WAV 16kHz tạm do ffmpeg tách ra khi chạy ASR (tự xoá sau mỗi video)
├── models/                     # Model PhoWhisper đã convert sang CTranslate2 (~3GB, sinh bởi convert_model.py)
├── map-keyframes/              # Chứa các file CSV map giữa frame_id và timestamp thực tế của video (từ BTC).
├── media-info/                 # Chứa thông tin YouTube gốc của video (Tiêu đề, Kênh, Description, Keywords, URL...).
├── npy_features/               # Chứa các file Vector Embeddings (.npy) do mô hình CLIP trích xuất ra (để nạp vào Qdrant).
├── objects/                    # Chứa file JSON liệt kê tất cả vật thể (object) phát hiện được từ mô hình Faster R-CNN pretrained trên OpenImages V4.
├── metadata/                   # Thư mục gom chung các file metadata sinh ra từ hệ thống
│   ├── metadata.json           # Output Bước 1: Chứa Detailed Caption và raw OCR text (từ Florence-2 và PaddleOCR).
│   ├── metadata_entities.json  # Output Bước 2: Chứa danh sách đối tượng (Objects) được Llama-3 chắt lọc từ metadata.json.
│   ├── metadata_spatial.json   # Output Bước 3: Chứa tọa độ Bounding Box và quan hệ không gian (Trái/Phải/Trên/Dưới) từ Florence-2 Object Detection.
│   ├── metadata_youtube.jsonl  # Dữ liệu sạch cào từ YouTube (Tiêu đề, Kênh, Description) copy từ luồng preprocess.
│   ├── metadata_youtube_bm25.pkl # Index BM25 dựng sẵn từ metadata_youtube.jsonl (để Dev 2 tích hợp tìm kiếm Video).
│   ├── metadata_asr/           # Output thô của ASR (Dev 4): MỖI VIDEO 1 FILE (L21_V001.json...) để checkpoint/resume nhanh.
│   └── metadata_asr.json       # File ASR bàn giao cho Dev 2 (mảng JSON), gộp từ metadata_asr/ bằng `python merge_asr.py`.
```

## Quy ước Nguồn dữ liệu
- **Khóa chính (Primary Key):** Mọi file JSON, Database và API đều bắt buộc tuân thủ khóa chính theo định dạng `L<Tập>_V<Video>_f<Số_Frame>` (Ví dụ chuẩn: `L21_V022_f1024`). 
- **Quy trình Offline Ingestion:** Tiền xử lý dữ liệu phải được chạy tuần tự: Sinh `metadata.json` $\rightarrow$ Sinh `metadata_entities.json` $\rightarrow$ Sinh `metadata_spatial.json`. Bỏ qua bất kỳ bước nào cũng sẽ gây đứt gãy luồng xử lý của hệ thống.
