# Task 4 Acceptance Report — Dev 2 Semantic Pipeline

## Kết luận

**Trạng thái: R&D hoàn tất, chưa production-ready.**

`task4_acceptance.py --check-live-elasticsearch` hiện trả:

```text
status=research_complete_with_known_limitations
research_complete=True
production_ready=False
```

Các core check về schema, entity/spatial benchmark, text regression, filter
end-to-end, Elasticsearch live và scale pilot đều pass. Production vẫn false
vì visual entity quality, relation annotation coverage và target-video holdout
chưa đạt gate.

## Bằng chứng nghiệm thu

### Elasticsearch v4

- Alias `semantic_frames` trỏ duy nhất tới `semantic_frames_v4`; đủ đúng 24 `_id`.
- Mapping strict có provenance cho contextual label grounding.
- Raw instance relations được giữ trong metadata để evaluation; indexed
  projection collapse relation trùng label triple từ `216` xuống `9`.
- 15 query benchmark, gồm 2 metadata/spatial filter case; cả hai relevant frame
  ở rank 1.
- Recall@5 `1.0000`; MRR@5 `0.9333`; NDCG@5 `0.9508`.
- Local p95 lần cuối (300 mẫu): `10.592 ms`, đạt gate `< 50 ms`.
- Scale pilot trên physical index riêng có `10.000` synthetic document:
  throughput khoảng `1,386 docs/s`, steady-state p95 `48.058 ms`.
- Mutating CLI bắt buộc `--index-name`; scale index không được gắn alias.

### Entity Extraction + visual truth

- `24/24` record có model/prompt provenance; model `llama3.2:3b`, prompt
  `entities-v1.3`.
- Structured JSON được validate và hậu kiểm evidence; list filter dùng all-of.
- Ground truth độc lập được annotate từ pixel cho 12 frame.
- All-field micro-F1 `0.5814`; searchable-entity micro-F1 `0.4194`.
- Object P/R/F1 `0.3396 / 0.5455 / 0.4186`; action F1 `0.1667`; color F1
  `0.5000`.

Benchmark đã sửa hạn chế “chưa có visual truth”, nhưng cũng chứng minh entity
metadata chưa đủ chính xác cho production filtering.

### Spatial Reasoning + contextual label grounding

- Full sample: 52 detection, 216 raw reciprocal instance relation trên 24 frame.
- Gold set mở rộng từ 3 frame/9 object lên 10 frame/35 object; 32 relation trên
  3 frame được annotate exhaustive.
- Contextual grounding sửa confusion chỉ khi caption/OCR/entity có evidence,
  không dùng `frame_id` hoặc gold: ví dụ `dice → domino`, `vase → sack`.
- Mỗi correction giữ `raw_label`, `label_source`, `label_evidence` và
  `label_grounding_version`.
- Localization P/R/F1 `0.9211 / 1.0000 / 0.9589`.
- Label accuracy sau grounding `1.0000` (trước grounding trên gold mở rộng:
  `0.5714`).
- Relation geometry F1 và queryable relation recall đều `1.0000` trên phần gold
  đã annotate (queryable recall trước grounding: `0.0000`).

### Security và testing

- Local Compose cũ vẫn bind loopback để development nhanh.
- Profile `docker-compose.elasticsearch.secure.yml` chạy project/port/volume
  riêng, bật authentication, HTTPS và transport TLS; password dùng Docker
  secret ngoài repository, public CA được verify bởi Python client.
- Python client từ chối gửi credential qua HTTP và không có đường tắt
  `verify_certs=False`.
- Offline suite: `142 passed, 2 skipped`; live FastAPI → Elasticsearch: `1 passed`.
- `httpx2` thay backend TestClient cũ nên không còn Starlette deprecation warning.

## Production gates còn fail

- Chỉ `3/10` spatial gold frame có exhaustive relation annotation; gate yêu cầu
  tối thiểu 5.
- Searchable-entity visual F1 chỉ `0.4194`; gate yêu cầu tối thiểu `0.70`.
- Spatial grounding rule được thiết kế từ sample hiện tại; chưa có holdout target
  video để chứng minh khả năng tổng quát hóa.
- 20/35 spatial gold object là hoa; dataset còn mất cân bằng và mới một annotator.
- Scale 10k dùng synthetic replication/single client; chưa chứng minh relevance,
  concurrency hoặc capacity trên corpus 100GB thật.
- Secure profile vẫn single-node; production cần HA, snapshot/restore drill,
  monitoring/audit, certificate rotation, managed PKI/firewall/secret manager.
- Florence `<OD>` không trả calibrated box score; confidence `0.5` vẫn là proxy.

## Lệnh tái kiểm tra

```powershell
python src/semantic_pipeline/benchmark_entities.py
python src/semantic_pipeline/benchmark_spatial.py

python src/semantic_pipeline/benchmark_elasticsearch.py `
  --index semantic_frames `
  --output src/semantic_pipeline/baseline_report_elasticsearch_v4_filters.json

python src/semantic_pipeline/benchmark_scale.py `
  --index-name semantic_frames_scale_10k_v1 `
  --documents 10000 `
  --skip-ingest `
  --output src/semantic_pipeline/scale_benchmark_report_10k_steady_v1.json

python src/semantic_pipeline/task4_acceptance.py --check-live-elasticsearch
python -m pytest -q src/semantic_pipeline/tests

$env:RUN_TASK4_LIVE = "1"
python -m pytest -q src/semantic_pipeline/tests/test_task4_live.py
Remove-Item Env:RUN_TASK4_LIVE
```

Machine-readable report: `src/semantic_pipeline/task4_acceptance_report.json`.
Generated metadata/reports/secrets are ignored; source, tests, ground truth,
Compose templates and documentation must be version-controlled.
