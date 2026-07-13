# Hướng dẫn benchmark Semantic Pipeline

## Mục tiêu

Mỗi tổ hợp model sinh caption/OCR và cấu hình truy xuất văn bản (ví dụ Florence-2
với BM25, hoặc model thay thế) phải được chấm trên **cùng một tập benchmark**.
Đánh giá cần tách thành hai phần: chất lượng text được trích xuất và chất lượng
truy xuất cuối cùng từ `keywords` đến danh sách frame được xếp hạng.

## Bộ dữ liệu benchmark đề xuất

Không dùng một dataset duy nhất để chọn model. Mỗi dataset kiểm tra một khía
cạnh khác nhau của Semantic Pipeline.

| Tầng | Dataset | Cách dùng | Vai trò |
| --- | --- | --- | --- |
| Regression nhanh | **MSVD** | Dùng caption chuẩn làm query, video đúng là relevant item | Phát hiện Florence/prompt/tokenizer mới làm giảm retrieval; không dùng để chọn model cuối vì clip ngắn và sạch. |
| So sánh model chuẩn | **MSR-VTT 1k-A** | Dùng caption làm query; sinh metadata cho clip rồi truy xuất bằng BM25 | So sánh model caption/OCR và cấu hình BM25 trên text-video retrieval chuẩn. |
| Benchmark hệ thống | **V3C1 / VBS tasks** | Bắt đầu V3C1, dùng metadata/ASR/keyframe và task VBS có sẵn | Đo tốc độ sinh metadata, kích thước index, p95 query và chất lượng search trên corpus lớn. |
| OCR chuyên biệt | **ICDAR 2015 Incidental Scene Text** | Chạy OCR trên split có nhãn, tính CER/Word F1 | Chấm OCR trong cảnh tự nhiên, phù hợp hơn ảnh văn bản chụp chủ đích. |
| Quyết định cuối | **AIC held-out v1** | Tách hẳn video/query/qrels không dùng khi tinh chỉnh | Chọn model/prompt/BM25 cho cuộc thi vì distribution gần dữ liệu BTC nhất. |
| Mở rộng domain | **Ego4D Episodic Memory** | Chỉ dùng sau khi baseline trên ổn định | Kiểm tra tổng quát hoá sang first-person/lifelogs; cần chấp nhận license. |

Nguồn tham khảo và điều kiện truy cập: [MSR-VTT](https://www.microsoft.com/en-us/research/?p=238188),
[V3C tại NIST](https://www.nist.gov/publications/v3c-research-video-collection),
[VBS data/tools](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools),
[ICDAR 2015 Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4) và
[Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/).
V3C1+2+3 rất lớn, vì vậy benchmark v1 dùng MSVD, MSR-VTT 1k-A, V3C1, ICDAR
và AIC held-out; chỉ mở rộng khi hạ tầng đã chịu được.

## Chuẩn bị một lần

1. Tạo manifest bất biến cho benchmark gồm `frame_id`, query, ground truth
   (`qrels.jsonl`) và một tập annotation nhỏ cho OCR/caption. Tách tập tinh
   chỉnh khỏi tập đánh giá; không điều chỉnh prompt/model theo tập đánh giá.
2. Đặt fixture nhỏ không nhạy cảm tại `tests/semantic_pipeline/benchmark/fixtures/`.
   Ảnh lớn, metadata sinh ra và index chỉ đặt trong `data/` hoặc `artifacts/`;
   các thư mục này không được commit.
3. Cố định quy tắc tokenize, ghép caption/OCR, preprocessing ảnh, `top_k` và
   cùng tập query/ground truth khi so sánh model.
4. Lưu model id/revision, commit mã nguồn, prompt/task Florence, thiết bị, batch
   size, precision, số frame và hash manifest cho mỗi run.

## Cách chạy một lần benchmark

1. Chạy model tạo caption/OCR cho toàn bộ corpus benchmark, tạo metadata và dựng
   chỉ mục BM25 theo cấu hình của run.
2. Chạy tất cả query benchmark, xuất `predictions.jsonl` theo định dạng trong
   [`tests/README.md`](../../../../tests/README.md), rồi tính metric truy xuất
   với `qrels.jsonl`.
3. Trên tập annotation OCR/caption, tính metric chất lượng trích xuất riêng;
   không dùng chính query test để chấm chất lượng caption/OCR.
4. Đo độ trễ query ít nhất 30 lần sau khi chỉ mục đã sẵn sàng; ghi p50/p95. Tách
   thời gian sinh metadata offline khỏi latency query online.
5. Lưu raw JSON/CSV tại `tests/semantic_pipeline/benchmark/results/<run_id>/`
   và thêm một dòng tóm tắt vào [benchmark_results.md](benchmark_results.md).

## Chỉ số bắt buộc

| Nhóm | Chỉ số | Ý nghĩa |
| --- | --- | --- |
| Truy xuất | Recall@1, Recall@5, Recall@10 | Tỷ lệ query có ít nhất một frame đúng trong Top-K; cao hơn tốt hơn. |
| Xếp hạng | MRR@10 | Nghịch đảo vị trí frame đúng đầu tiên; cao hơn tốt hơn. |
| OCR | CER và Word F1 | Character Error Rate thấp hơn tốt hơn; Word F1 cao hơn tốt hơn. Chỉ tính trên tập có nhãn OCR. |
| Caption | Concept Recall | Tỷ lệ concept bắt buộc (đối tượng, hành động, vị trí) xuất hiện trong caption đã chuẩn hoá; cao hơn tốt hơn. |
| Vận hành | p50/p95 query, frame/giây, RAM/VRAM đỉnh | So sánh độ trễ online và chi phí sinh metadata/index. |

Nếu chưa có annotation caption/OCR đủ tin cậy, chỉ báo cáo metric truy xuất và độ
trễ, đồng thời ghi rõ giới hạn này; không tự gán một con số "accuracy" không có
ground truth. Query không có ground truth không được đưa vào mẫu số.

## Quy tắc so sánh và báo cáo

- Baseline ban đầu là `microsoft/Florence-2-base` + BM25 với prompt/task và quy
  tắc tokenize được ghi rõ.
- Chỉ so sánh run có cùng `benchmark_id`/hash manifest. Khi thay đổi dataset,
  prompt, tokenizer, qrels hay schema metadata, tạo benchmark version mới.
- Chạy ít nhất 3 lần cho số đo tốc độ; báo cáo trung vị hoặc trung bình và điều
  kiện phần cứng. Không chọn riêng lần chạy đẹp nhất.
- Raw output không commit. Chỉ commit bảng tóm tắt, fixture nhỏ đã ẩn danh và
  cấu hình đủ để tái lập.
- Model mới phải được so sánh đồng thời về retrieval, OCR/caption (nếu có nhãn),
  latency và tài nguyên trước khi đề xuất thay baseline.
