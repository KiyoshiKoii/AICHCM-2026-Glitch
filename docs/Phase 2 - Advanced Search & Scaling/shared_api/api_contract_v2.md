# API Contract v2 - Phase 2: Advanced Search & Scaling

Tài liệu này kế thừa và mở rộng từ `api_contract_v1.md`. Dưới đây là các thay đổi chính trong Phase 2:

## CẬP NHẬT 1: API /search/text (Thêm Tọa độ)
Response của API tổng giờ đây sẽ trả về thêm trường `detections` (nếu có) để Frontend vẽ khung.

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "total_results": 50,
    "results": [
      {
        "frame_id": "L21_V022_f1024",
        "score": 0.98,
        "thumbnail_url": "/media/thumbnails/L21_V022_f1024.jpg",
        "detections": [
          {
            "label": "traffic sign",
            "bbox": [0.18, 0.12, 0.42, 0.45]
          }
        ]
      }
    ]
  }
}
```

## CẬP NHẬT 2: Elasticsearch Filters (Semantic API)
API gọi qua Dev 2 (Semantic) giờ bắt buộc phải truyền `filters` chuẩn nếu LLM parse được.

**Request Body:**
```json
{
  "keywords": ["person", "car"],
  "top_k": 200,
  "filters": {
    "spatial_relations": [
      {"subject": "person", "predicate": "left_of", "object": "car"}
    ]
  }
}
```
