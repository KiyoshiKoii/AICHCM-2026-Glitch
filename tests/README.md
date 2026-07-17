# Khu vực kiểm thử

Thư mục `tests/` chỉ chứa mã kiểm thử, cấu hình benchmark và fixture nhỏ có thể
chia sẻ qua Git. Không đặt mã nguồn chạy thật của pipeline, model checkpoint,
ảnh/video thô, vector index hoặc kết quả benchmark sinh ra tại đây.

```text
tests/
├── visual_pipeline/benchmark/    # Benchmark Dev 1
└── semantic_pipeline/benchmark/  # Benchmark Dev 2
```

Mỗi benchmark phải dùng cùng tập `qrels` (ground truth) và cùng định dạng dự
đoán để so sánh công bằng giữa các model. Dữ liệu lớn, index và output của từng
lần chạy đặt tại các thư mục `data/`, `artifacts/`, `results/` bên trong thư mục
benchmark tương ứng; chúng đã được `.gitignore` loại trừ. Chỉ commit fixture
nhỏ đã được ẩn danh và không chứa dữ liệu BTC.

Định dạng tối thiểu dùng chung:

```json
// qrels.jsonl
{"query_id":"q-001","relevant_frame_ids":["vid05_f1024"]}

// predictions.jsonl
{"query_id":"q-001","ranked_frame_ids":["vid05_f1024","vid05_f1023"]}
```

`query_id` phải xuất hiện đúng một lần trong mỗi file. Thứ tự
`ranked_frame_ids` là thứ tự xếp hạng giảm dần. Dev 1 và Dev 2 chịu trách nhiệm
lưu raw output theo định dạng của pipeline mình, sau đó xuất một file dự đoán
chuẩn này để tính Recall@K và MRR.
