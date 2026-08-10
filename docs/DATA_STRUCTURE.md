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
│   ├── caption/                # Metadata thị giác từ Gemini, checkpoint theo từng video.
│   │   └── L21/L21_V001.json   # Một mảng record của một video, đặt trong batch LXX.
│   ├── metadata_youtube.jsonl  # Dữ liệu sạch cào từ YouTube (Tiêu đề, Kênh, Description) copy từ luồng preprocess.
│   ├── metadata_youtube_bm25.pkl # Index BM25 dựng sẵn từ metadata_youtube.jsonl (để Dev 2 tích hợp tìm kiếm Video).
│   ├── metadata_asr/           # Output thô của ASR (Dev 4): MỖI VIDEO 1 FILE (L21_V001.json...) để checkpoint/resume nhanh.
│   └── metadata_asr.json       # File ASR bàn giao cho Dev 2 (mảng JSON), gộp từ metadata_asr/ bằng `python merge_asr.py`.
```

## Quy ước Nguồn dữ liệu
- **Khóa chính (Primary Key):** Mọi file JSON, Database và API đều bắt buộc tuân thủ khóa chính theo định dạng `L<Tập>_V<Video>_f<Số_Frame>` (Ví dụ chuẩn: `L21_V022_f1024`). 
- **Quy trình Offline Ingestion (visual):** Chạy một bước `gemini_visual_extractor.py` để sinh `metadata/caption/LXX/<video_id>.json`. Caption, OCR, detection và quan hệ không gian cùng thuộc một bản ghi; không còn các artifact visual trung gian `metadata_entities.json` hoặc `metadata_spatial.json`.

## Gemini compact visual metadata

`src/semantic_pipeline/gemini/extractor.py` creates the compact visual
artifact with the model configured by `GEMINI_VISUAL_MODEL` in `.env`. It
processes 25 keyframes per request by default (maximum 100, and additionally
constrained by a 14 MiB raw-image budget).
The persisted JSON array intentionally contains only data that cannot be
derived from the primary key:

```json
{
  "frame_id": "L21_V001_f0017",
  "caption": "A warning sign beside a flooded road.",
  "detailed_caption": "A warning sign stands beside a wet road bordered by vegetation. Floodwater covers part of the road.",
  "caption_vi": "Một biển cảnh báo nằm cạnh con đường ngập nước.",
  "detailed_caption_vi": "Một biển cảnh báo đặt bên con đường ướt có cây cối bao quanh. Nước ngập che phủ một phần mặt đường.",
  "ocr_text": "CẢNH BÁO SẠT LỞ NGUY HIỂM",
  "news_ticker_text": "",
  "detections": [
    {
      "object_id": "traffic_sign_0",
      "label": "traffic sign",
      "bbox": [0.18, 0.13, 0.42, 0.45],
      "description": "a red triangular warning sign beside a flooded road",
      "description_vi": "biển cảnh báo hình tam giác màu đỏ bên đường ngập nước",
      "attributes": ["red", "triangular", "warning sign"],
      "action": ""
    }
  ],
  "spatial_relations": [
    {"subject_id": "person_0", "predicate": "right_of", "object_id": "traffic_sign_0"}
  ]
}
```

`video_name` and the keyframe number are derived from `frame_id` when an API or
index needs them. The actual video timestamp requires the corresponding
`data/map-keyframes/<video>.csv` mapping and is not persisted here. Gemini
response boxes `[ymin, xmin, ymax, xmax]` in `[0,1000]` are converted to the
stored `[x1, y1, x2, y2]` range `[0,1]`.

Gemini decides whether a detection has enough distinctive visual evidence to
receive semantic details. At most five salient objects per frame are enriched
with `description`, directly generated `description_vi`, up to six normalized
English `attributes`, and an optional visible `action`. Other useful boxes are
kept with empty detail fields. Elasticsearch stores detections as nested
objects, so attributes belonging to different people are not mixed together.

`caption_vi` and `detailed_caption_vi` are generated directly from the image in
natural Vietnamese within the same Gemini request. They are not produced by a
post-processing translation step. The English fields remain available for the
CLIP/English retrieval route, while Elasticsearch indexes the Vietnamese fields
separately with accent folding for Vietnamese queries.

Gemini prompt context is selected per video from the `L21`–`L30` profile catalog
and the small video-level fields in `metadata_youtube.jsonl`. The context only
specializes extraction priorities; it is not copied into every frame record.
Each API batch is kept within one `video_id`, so frames from different programs
cannot accidentally share a domain prompt.

For L21 and L22, `news_ticker_text` separately stores the visible segment of
the scrolling news crawl at the bottom of the broadcast frame. It is empty for
other collections and when the ticker is not legible; `ocr_text` remains for
all other scene text.

`data/global_filter_results.csv` maps each keyframe to a visually unique
representative in the same video. The extractor sends only representative
frames to Gemini, then copies the resulting metadata to the mapped frames while
replacing only `frame_id`. This is enabled by default and can be disabled with
`--without-global-filter` for controlled comparisons.

The Gemini scheduler starts at most 15 parallel requests in each 60-second
window. RPM and temporary service-overload responses are retried in later
windows. Daily quota is checked manually on the provider dashboard and is not
tracked in a local state file.

Smoke test one batch before a full run:

```powershell
$env:PYTHONPATH = "src"
python -m semantic_pipeline.gemini.extractor --input-dir data/keyframes/L21_V001 --output-dir data/metadata/caption --limit 25 --batch-size 25
```

After reviewing that artifact, replace `--limit 25` with no limit. The full
result remains split by video, for example
`data/metadata/caption/L21/L21_V001.json`.
