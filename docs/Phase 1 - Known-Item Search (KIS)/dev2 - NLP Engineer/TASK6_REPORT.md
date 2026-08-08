# Báo cáo tích hợp BTC object vào Semantic Pipeline

Ngày kiểm chứng: **2026-08-09**

Nhánh: `feat/semantic-pipeline-feature-btc-object-integration`

## Kết quả

Kế hoạch trong `BTC_INTEGRATION_PLAN.md` đã được triển khai cho dữ liệu hiện có.
Identity, detection và metadata bây giờ dùng trực tiếp object BTC/Open Images thay
vì sinh object giả hoặc phụ thuộc Florence-2. Batch kiểm chứng là toàn bộ keyframe
đang gắn ở **L21 + L22**; các script giữ tùy chọn `--resume` để mở rộng khi có thêm
keyframe.

| Hạng mục | Kết quả |
|---|---:|
| Map BTC được kiểm tra round-trip | 873 video |
| Video L21/L22 đã build object index | 60 |
| Keyframe/object nguồn L21/L22 | 16.896 |
| Metadata canonical sinh ra | 16.886 document |
| Detection nguồn | 1.689.600 (100/frame) |
| Detection đạt score >= 0,20 | 191.158 (11,314/frame) |
| Detection sau area + NMS + top 15 | 160.559 (9,503/frame) |
| Detection trong metadata canonical | 160.418 (9,500/frame) |
| Relation thô | 1.331.648 (78,861/frame) |
| Relation đưa vào projection Elasticsearch | 604.347 (35,790/frame) |
| Mức giảm relation khi index | 54,62% |
| Bộ query KIS BTC | 30 case |

Metadata JSON sinh ra khoảng **430,2 MB** và nằm dưới `data/processed/`, không đưa
artifact tái sinh này vào Git.

## Những phần đã hoàn thiện

- Identity BTC native: `video_name`, `keyframe_n`, `frame_index`, `timestamp_ms`
  và `frame_id` lấy từ `map-keyframes`; có helper tạo dòng nộp `(video_id,
  frame_idx)`.
- Adapter object: đổi box YXYX sang XYXY, lọc score/area, class-wise NMS, top 15,
  giữ MID, confidence, grid cell, area và provenance `btc_detector`.
- Parquet theo video: giảm việc đọc hàng trăm nghìn JSON nhỏ, hỗ trợ resume và chọn
  video/nhóm video.
- Ontology Open Images và lexicon EN/VI: expand ancestor theo chiều lên, map query
  sang MID, verbalize object để BM25/Elasticsearch tìm được cả frame không có
  caption/OCR.
- Metadata schema v1.1: fusion object, media-info, optional visual metadata và
  spatial relation; `entities.objects` được dẫn xuất nhất quán từ detection.
- Elasticsearch v6/API: analyzer tiếng Việt, text fields mới, nested object filter,
  `min_object_score`, `object_counts`, ingest streaming theo từng file video và
  collapse relation trước khi index. Response API cũ không đổi.
- Bộ acceptance mới kiểm tra identity, provenance, score/top-15 và consistency
  giữa detection với `entities.objects`.

## Phát hiện về dữ liệu gốc

Giả định ban đầu rằng `frame_idx` tăng nghiêm ngặt không đúng với dữ liệu BTC:
**192/873** map có frame index lặp, tổng cộng 614 dòng dư theo identity nộp bài.
Riêng L21/L22 có 10 dòng như vậy. Pipeline vẫn join object bằng `keyframe_n`, sau
đó giữ keyframe `n` đầu tiên cho mỗi `frame_idx`; nhờ vậy không sinh hai document
cùng `frame_id` và vẫn tạo đúng dòng nộp AIC.

## Audit chất lượng

- 16.886/16.886 record validate schema v1.1.
- Identity failures: **0**.
- Object filter/provenance failures: **0**.
- `entities.objects` so với nhãn detection: precision **1,0000**, recall **1,0000**,
  F1 **1,0000**; vượt gate 0,70.
- Lexicon có 120 MID ưu tiên, phủ **98,24%** số detection thực tế sau lọc; độ phủ
  theo số class phân biệt là 120/430 (**27,91%**).
- Đã kiểm tra bằng mắt các frame đại diện ở L21/L22 và sửa bộ query để không biến
  nhãn sai rõ ràng của detector thành ground truth. Open Images vẫn có thể nhầm
  class ở một số ảnh; open-vocabulary re-labeling tiếp tục thuộc STEP 7 backlog.

## Kiểm thử

```text
python -m pytest src/semantic_pipeline/tests -q -rs
187 passed, 2 skipped in 28.55s
```

Hai test bị skip là live Elasticsearch, cần bật `RUN_ELASTICSEARCH_INTEGRATION=1`
hoặc `RUN_TASK4_LIVE=1` sau khi bootstrap Elasticsearch local. Máy kiểm chứng hiện
không có Elasticsearch đang chạy, nên chưa đo live p95/rank-1 trên index v6; mapping,
query DSL, projection ingest và API contract đã được kiểm tra bằng unit test.

Baseline 24 frame không regression:

| Metric | Kết quả |
|---|---:|
| Recall@5 | 1,0000 |
| MRR@5 | 0,9412 |
| NDCG@5 | 0,9566 |
| p95 | 0,558 ms |

## Chạy tiếp khi gắn thêm data

```powershell
python scripts/build_objects_index.py --videos L23,L24 --resume
python -m src.semantic_pipeline.metadata_builder --videos L23,L24 --resume
```

Có thể thay danh sách nhóm bằng video cụ thể, ví dụ
`--videos L23_V001,L23_V002`. Trước khi chạy cần có đủ `map-keyframes`, `objects`
và `media-info`; ảnh keyframe là tùy chọn đối với object/spatial pipeline này.
