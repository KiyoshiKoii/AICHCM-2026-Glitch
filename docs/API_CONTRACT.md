# API Contract

Đây là đặc tả API duy nhất của dự án AICHCM-2026-Glitch. Frontend gọi Backend
qua base URL `http://localhost:8000/api/v1`; Backend gọi hai pipeline nội bộ ở
port `8001` và `8002`.

## Quy ước chung

- JSON request dùng header `Content-Type: application/json`, trừ API upload ảnh.
- `frame_id` là ID keyframe nội bộ, dạng `L<Tập>_V<Video>_f<Số_Keyframe>`.
- `video_name` là tên video không có đuôi file.
- `frame_index` là chỉ số frame gốc được ánh xạ từ keyframe; dùng giá trị này
  khi tạo kết quả nộp bài.
- Lỗi dịch vụ có dạng `{ "error": { "code": "...", "message": "..." } }`.

## Backend public API

### `POST /search/text`

Tìm keyframe theo mô tả sự kiện. Backend gộp kết quả Visual và Semantic bằng
RRF. Đặt `use_rerank: true` khi cần Gemini chấm lại thứ tự; khi đó
`llm_reranked_results` chứa thứ tự sau rerank.

```json
{
  "query": "người đàn ông mặc áo đỏ làm rơi ví",
  "top_k": 50,
  "use_rerank": false
}
```

- `use_rerank`: mặc định `false` để không tốn Gemini request. Khi bật, backend
  gửi một request Gemini để re-rank các kết quả retrieval.

```json
{
  "status": "success",
  "message": "Retrieved successfully from Visual & Semantic Pipelines",
  "data": {
    "total_results": 50,
    "results": [
      {
        "frame_id": "L21_V022_f0012",
        "video_name": "L21_V022",
        "frame_index": 1024,
        "score": 0.98,
        "thumbnail_url": "/media/thumbnails/L21_V022_f0012.jpg",
        "metadata": {}
      }
    ],
    "llm_reranked_results": []
  }
}
```

### `POST /search/image`

Tìm keyframe tương tự từ ảnh truy vấn.

- Request: `multipart/form-data`.
- Fields: `image_file` (`.jpg`, `.jpeg`, `.png`) và `top_k` (mặc định `50`).
- Response: cùng schema với `POST /search/text`.

### `POST /vqa`

Trả lời câu hỏi bằng Gemini từ các keyframe đã được retrieval/rerank. API này
không dùng ASR hay transcript: một batch chứa toàn bộ ảnh candidate được gửi
trong **một** request Gemini, và model trả một answer độc lập cho từng `frame_id`.

```json
{
  "query": "cảnh lễ trao giải có nhiều người đứng trên sân khấu",
  "question": "Có bao nhiêu người trên sân khấu?",
  "retrieval_top_k": 50,
  "answer_top_k": 10
}
```

- `retrieval_top_k`: số hit đưa vào retrieval/rerank, từ `1` đến `100`.
- `answer_top_k`: số hit rerank đầu gọi Gemini, từ `1` đến `20`, không lớn hơn
  `retrieval_top_k`.
- Mỗi truy vấn VQA gọi Gemini đúng một lần, giúp phù hợp hạn mức RPM thấp.
- Nếu không có `llm_reranked_results`, API dùng kết quả RRF làm fallback.
- Cần cấu hình `GEMINI_API_KEY`; nếu chưa có, API trả `503 VQA_UNAVAILABLE`.

```json
{
  "status": "success",
  "message": "Answered from LLM re-ranked frames",
  "data": {
    "total_candidates": 2,
    "candidates": [
      {
        "frame_id": "L21_V022_f0012",
        "video_name": "L21_V022",
        "frame_index": 1024,
        "score": 0.98,
        "thumbnail_url": "/media/thumbnails/L21_V022_f0012.jpg",
        "metadata": {},
        "answer": "5",
        "confidence": 0.91
      }
    ]
  }
}
```

Để nộp một candidate VQA cho BTC, dùng `video_name`, `frame_index` và `answer`;
không dùng suffix ordinal trong `frame_id` làm frame gốc.

### `GET /frames/context/{frame_id}?window=5`

Lấy frame trung tâm và keyframe lân cận để người dùng kiểm chứng kết quả.

```json
{
  "status": "success",
  "data": {
    "center_frame": {"frame_id": "L21_V022_f0012", "thumbnail_url": "..."},
    "before_frames": [],
    "after_frames": []
  }
}
```

### `GET /media/thumbnails/{frame_id}.jpg`

Phục vụ thumbnail JPEG của keyframe, ví dụ
`/media/thumbnails/L21_V022_f0012.jpg`.

## Pipeline nội bộ

### `POST http://localhost:8001/internal/search/visual`

```json
{
  "visual_prompt": "A person dropping a pink teddy bear keychain",
  "prompt_variants": ["A pink keychain falling from a hand"],
  "candidate_k": 50,
  "temporal_window": 5,
  "top_k": 200
}
```

`score` là cosine similarity gốc trong `[-1, 1]`; `normalized_score` là
`(score + 1) / 2`, chỉ dùng để hiển thị. Pipeline deduplicate keyframe gần nhau
trong cùng video theo `temporal_window`.

```json
{
  "status": "success",
  "data": [
    {
      "frame_id": "L21_V022_f0012",
      "score": 0.27472,
      "normalized_score": 0.63736,
      "video_name": "L21_V022",
      "frame_index": 12
    }
  ]
}
```

### `POST http://localhost:8002/internal/search/text`

```json
{
  "keywords": ["person", "red car"],
  "top_k": 200,
  "filters": {
    "time_of_day": "night",
    "setting": "outdoor",
    "locations": ["street"],
    "objects": ["person", "car"],
    "actions": ["standing"],
    "colors": ["red"],
    "code_language": "sql",
    "code_patterns": ["not exists"],
    "spatial_relations": [
      {"subject": "person", "predicate": "left_of", "object": "car"}
    ]
  }
}
```

`filters` là tùy chọn và hiện chỉ khả dụng khi Semantic Pipeline chạy
Elasticsearch. Các mảng filter dùng all-of (AND); predicate hợp lệ là `left_of`,
`right_of`, `above`, `below`, `overlapping`.

```json
{
  "status": "success",
  "data": [
    {
      "frame_id": "L21_V022_f0012",
      "score": 15.6,
      "video_name": "L21_V022",
      "frame_index": 12,
      "detections": [
        {"label": "traffic sign", "bbox": [0.18, 0.12, 0.42, 0.45]}
      ]
    }
  ]
}
```
