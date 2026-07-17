# Hướng dẫn benchmark Visual Pipeline

## Mục tiêu

Mỗi model thị giác (ví dụ CLIP baseline, SigLIP hoặc model khác) phải được chấm
trên **cùng một tập benchmark** trước khi thay thế hoặc so sánh. Không kết luận
model mới tốt hơn chỉ dựa vào vài query được chọn thủ công.

Benchmark này đánh giá cả chất lượng truy xuất và chi phí vận hành của luồng:
ảnh/frame → embedding → Qdrant → danh sách frame được xếp hạng.

## Bộ dữ liệu benchmark đề xuất

Không dùng một dataset duy nhất để chọn model. Mỗi dataset kiểm tra một khía
cạnh khác nhau của Visual Pipeline.

| Tầng | Dataset | Cách dùng | Vai trò |
| --- | --- | --- | --- |
| Regression nhanh | **MSVD** | Chạy toàn bộ split đánh giá sau mỗi thay đổi lớn | Phát hiện model/config mới làm giảm chất lượng; không dùng để chọn model cuối vì clip ngắn và sạch. |
| So sánh model chuẩn | **MSR-VTT 1k-A** | Dùng caption làm text query, clip đúng là relevant item | So sánh công bằng CLIP, SigLIP và model retrieval khác bằng Recall@K/MRR. |
| Benchmark hệ thống | **V3C1 / VBS tasks** | Bắt đầu với V3C1 và task VBS có sẵn; sau đó mới mở rộng V3C2/3 | Đo indexing, dung lượng index, latency và keyframe/shot retrieval ở quy mô lớn. |
| Quyết định cuối | **AIC held-out v1** | Tách hẳn video/query/qrels không dùng khi tinh chỉnh | Chọn model cho cuộc thi vì distribution gần dữ liệu BTC nhất. |
| Mở rộng domain | **Ego4D Episodic Memory** | Chỉ dùng sau khi baseline trên ổn định | Kiểm tra khả năng tổng quát hoá sang truy vấn first-person/lifelogs; cần chấp nhận license. |

Nguồn tham khảo và điều kiện truy cập: [MSR-VTT](https://www.microsoft.com/en-us/research/?p=238188),
[V3C tại NIST](https://www.nist.gov/publications/v3c-research-video-collection),
[VBS data/tools](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/) và
[Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/).
V3C1+2+3 rất lớn, vì vậy không tải toàn bộ ngay: benchmark v1 dùng MSVD,
MSR-VTT 1k-A, V3C1 và AIC held-out; chỉ mở rộng khi hạ tầng đã chịu được.

## Chuẩn bị một lần

1. Tạo một manifest bất biến cho tập benchmark, gồm phiên bản dữ liệu, danh sách
   `frame_id`, query và ground truth (`qrels.jsonl`). Tách riêng tập dùng để tinh
   chỉnh tham số và tập đánh giá; không chỉnh model theo kết quả của tập đánh giá.
2. Đặt fixture nhỏ không nhạy cảm tại `tests/visual_pipeline/benchmark/fixtures/`.
   Dữ liệu ảnh lớn, vector và Qdrant index chỉ đặt trong `data/` hoặc `artifacts/`
   của cùng thư mục; các thư mục này không được commit.
3. Cố định tham số chung khi so sánh: preprocessing ảnh, kích thước embedding,
   metric Cosine, `top_k`, cấu hình Qdrant và cùng tập query/ground truth.
4. Ghi rõ model id/revision, commit mã nguồn, thiết bị, batch size, precision,
   số frame và hash manifest cho mọi lần chạy.

## Cách chạy một lần benchmark

1. Sinh embedding và index cho toàn bộ corpus benchmark bằng một model/config.
2. Chạy tất cả query benchmark, không lọc thủ công kết quả, và xuất một file
   `predictions.jsonl` theo định dạng trong [`tests/README.md`](../../../../tests/README.md).
3. Tính chỉ số truy xuất trên file dự đoán và `qrels.jsonl`.
4. Đo độ trễ query ít nhất 30 lần sau khi index đã sẵn sàng; ghi p50 và p95.
   Tách riêng thời gian tạo embedding/index khỏi độ trễ query online.
5. Lưu raw JSON/CSV tại `tests/visual_pipeline/benchmark/results/<run_id>/` và
   thêm một dòng tóm tắt vào [benchmark_results.md](benchmark_results.md).

## Chỉ số bắt buộc

| Nhóm | Chỉ số | Ý nghĩa |
| --- | --- | --- |
| Chất lượng truy xuất | Recall@1, Recall@5, Recall@10 | Tỷ lệ query có ít nhất một frame đúng trong Top-K; cao hơn tốt hơn. |
| Chất lượng xếp hạng | MRR@10 | Nghịch đảo vị trí frame đúng đầu tiên, trung bình trên các query; cao hơn tốt hơn. |
| Độ trễ online | p50, p95 (ms/query) | Đo sau khi index sẵn sàng; thấp hơn tốt hơn. |
| Hiệu năng offline | frame/giây, thời gian index | Đo embedding và nạp Qdrant; dùng để ước lượng thời gian chạy dữ liệu BTC. |
| Tài nguyên | VRAM/RAM đỉnh, dung lượng index | So sánh chi phí triển khai thực tế. |

Nếu query có nhiều frame đúng, chỉ cần một frame đúng xuất hiện trong Top-K để tính
Recall@K. Query không có ground truth không được đưa vào mẫu số. Nếu một model
không chạy được hết tập benchmark, ghi rõ số query lỗi và không so sánh trực tiếp
với run hoàn chỉnh.

## Quy tắc so sánh và báo cáo

- Baseline ban đầu là `openai/clip-vit-base-patch32` với cấu hình được ghi đầy đủ.
- Chỉ so sánh các run có cùng `benchmark_id`/hash manifest. Thay đổi dataset,
  preprocessing hay qrels tạo một benchmark version mới, không gộp số với bản cũ.
- Chạy ít nhất 3 lần khi đo tốc độ; báo cáo trung vị hoặc trung bình kèm điều kiện
  phần cứng. Không chọn riêng lần chạy đẹp nhất.
- Đưa kết quả chưa tổng hợp vào `results/`; chỉ commit bảng tóm tắt và cấu hình
  đủ để tái lập. Không commit keyframe, `.npy`, local Qdrant DB hoặc checkpoint.
- Model mới chỉ được đề xuất thay baseline khi có bảng so sánh đầy đủ về chất lượng,
  độ trễ và tài nguyên; ghi rõ trade-off thay vì chỉ nêu một chỉ số tốt hơn.
