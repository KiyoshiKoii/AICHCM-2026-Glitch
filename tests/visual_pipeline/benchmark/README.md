# Benchmark Visual Pipeline (Dev 1)

Thư mục này chỉ dành cho benchmark của model truy xuất thị giác. Khi triển khai,
đặt mã chạy benchmark và fixture nhỏ tại đây; không đặt implementation production
vào `tests/`.

```text
benchmark/
├── fixtures/       # Manifest/query/qrels nhỏ được phép commit
├── scripts/        # Script chạy và tính metric benchmark
├── data/           # Dữ liệu benchmark lớn, bị ignore
├── artifacts/      # Vector/index tạm, bị ignore
└── results/        # JSON/CSV theo từng lần chạy, bị ignore
```

Mỗi script phải xuất `predictions.jsonl` theo quy ước trong
[`tests/README.md`](../../README.md). Hướng dẫn đầy đủ và bảng lịch sử kết quả
ở thư mục tài liệu Dev 1.
