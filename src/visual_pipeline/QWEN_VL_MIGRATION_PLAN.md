# Kế hoạch chuyển Visual Pipeline từ CLIP sang Qwen3-VL-Embedding-8B

## 1. Mục tiêu

Thay model `openai/clip-vit-base-patch32` bằng
`Qwen/Qwen3-VL-Embedding-8B` trong Visual Pipeline hiện tại.

Phạm vi thay đổi chỉ gồm các file đang có:

- `config.py`
- `extractor.py`
- `database.py`
- `server.py`

Không thêm embedding service/module mới. Không thay đổi luồng tìm kiếm, filter,
temporal deduplication hoặc cách định dạng kết quả.

API `POST /internal/search/visual` phải giữ nguyên:

- Request hiện tại: `visual_prompt`, `prompt_variants`, `batch_ids`, `video_ids`,
  `top_k`, `candidate_k`, `temporal_window`.
- Response hiện tại: `{ "status": "success", "data": [...] }`.
- Mỗi phần tử kết quả tiếp tục có `frame_id`, `score`, `normalized_score`,
  `video_name`, `frame_index` với cùng kiểu dữ liệu và ý nghĩa.

## 2. Quyết định kỹ thuật

- Dùng model embedding đa phương thức
  `Qwen/Qwen3-VL-Embedding-8B`, không dùng bản `Qwen3-VL-8B-Instruct` sinh văn bản.
- Ảnh keyframe và text query phải được encode bằng đúng cùng model, revision,
  instruction và kích thước vector.
- Giữ cosine similarity và L2 normalization như pipeline hiện tại.
- Giữ kích thước vector mặc định là **512** bằng khả năng MRL/truncate dimension
  của Qwen. Điều này giữ chi phí lưu trữ gần cấu hình CLIP hiện tại và không làm
  vector tăng từ 512 lên 4096 chiều.
- Dù vẫn là 512 chiều, vector Qwen không tương thích với vector CLIP. Toàn bộ
  keyframe bắt buộc phải được extract và index lại.
- Extraction chạy trên cloud GPU. Model dùng `bfloat16` nếu GPU hỗ trợ, nếu không
  thì dùng `float16`; batch size được cấu hình nhỏ hơn CLIP và có thể chỉnh bằng
  biến môi trường.
- Không thêm reranker và không thay đổi thuật toán ensemble các `prompt_variants`.

## 3. Thay đổi theo từng file

### `config.py`

1. Thay cấu hình mang tên CLIP bằng cấu hình model trung lập/Qwen:
   - Model ID mặc định: `Qwen/Qwen3-VL-Embedding-8B`.
   - Model revision có thể cố định bằng biến môi trường để lần extract và lần
     chạy server dùng đúng cùng checkpoint.
   - `VECTOR_SIZE=512` và cho phép override qua biến môi trường.
   - Batch size mặc định phù hợp model 8B, dự kiến bắt đầu từ `1` hoặc `2` rồi
     tăng theo VRAM cloud.
   - Query instruction cố định, ví dụ:
     `Retrieve images relevant to the user's visual search query.`
2. Tách artifact theo model bằng một tên/fingerprint ổn định trong `NPY_DIR`
   hoặc tên collection. Mục đích là không để resume logic nhận nhầm `.npy` CLIP
   cũ là embedding Qwen.
3. `print_config()` in thêm model revision, vector size, dtype và batch size để
   log cloud đủ thông tin tái chạy.

### `extractor.py`

1. Bỏ `CLIPProcessor` và `CLIPModel`; load
   `Qwen/Qwen3-VL-Embedding-8B` bằng API multimodal embedding được model hỗ trợ.
2. Giữ nguyên các phần sau:
   - Quét folder video bắt đầu bằng `L`.
   - Sắp xếp tên ảnh.
   - Bỏ qua ảnh lỗi.
   - Xử lý theo batch.
   - Lưu `<video>.npy` và `<video>_filenames.npy`.
   - Resume khi đúng artifact Qwen đã tồn tại.
3. Thay phần `get_image_features()` bằng encode batch ảnh của Qwen.
4. Cắt output về `VECTOR_SIZE=512`, L2-normalize, chuyển về CPU và lưu `float16`
   để hạn chế dung lượng artifact.
5. Chạy bằng `model.eval()` và `torch.inference_mode()`; chọn BF16/FP16 theo GPU.
6. Sau mỗi batch kiểm tra:
   - Shape là `(số ảnh hợp lệ, VECTOR_SIZE)`.
   - Không có `NaN` hoặc `Inf`.
   - Thứ tự vector khớp chính xác với `valid_filenames`.
7. Không tái sử dụng `.npy` CLIP hiện có. Output Qwen phải nằm trong namespace
   riêng hoặc chỉ resume khi metadata xác nhận đúng model/revision/dimension.

### `database.py`

1. Giữ nguyên Qdrant local client, cosine distance, payload, stable UUID và batch
   upload hiện tại.
2. Đọc artifact Qwen từ đường dẫn cấu hình mới.
3. Trước khi upload, validate mọi file feature:
   - Mảng hai chiều.
   - Chiều cuối bằng `VECTOR_SIZE`.
   - Số vector khớp số filename.
   - Không có `NaN` hoặc `Inf`.
4. Dùng collection dành riêng cho Qwen hoặc reset collection hiện tại chỉ sau
   khi extraction Qwen đã hoàn tất. Không trộn vector CLIP và Qwen.
5. Không thay đổi payload `video_id`, `frame_id`, `frame_index` để server và UI
   không bị ảnh hưởng.

### `server.py`

1. Bỏ `CLIPProcessor` và `CLIPModel`; load cùng Qwen model/revision/dimension đã
   dùng trong `extractor.py`.
2. Thay duy nhất khối encode text:
   - Encode `visual_prompt` và `prompt_variants` trong cùng batch.
   - Áp dụng query instruction cố định.
   - Lấy vector 512 chiều và L2-normalize từng query.
   - Giữ nguyên phép lấy trung bình các prompt variants rồi normalize lần cuối.
3. Giữ nguyên hoàn toàn:
   - `SearchRequest` và validation.
   - `matches_scope()` và `scoped_video_ids()`.
   - Qdrant filter và `candidate_limit`.
   - `temporal_deduplicate()`.
   - Cách tính `score` và `normalized_score`.
   - Tên field, kiểu dữ liệu, HTTP endpoint và cấu trúc response.
4. Khi khởi động, kiểm tra collection tồn tại và dimension/model metadata tương
   thích. Nếu không đúng thì fail sớm thay vì query nhầm index CLIP.

## 4. Dependency và môi trường cloud

Qwen3-VL-Embedding cần phiên bản thư viện mới hơn pin hiện tại. Môi trường chạy
Visual Pipeline trên cloud cần tối thiểu:

- `transformers>=4.57.0`
- `qwen-vl-utils>=0.0.14`
- Phiên bản `sentence-transformers` hỗ trợ model Qwen3-VL-Embedding
- PyTorch/CUDA phù hợp với GPU cloud
- Có thể cài `flash-attn` nếu image GPU và CUDA tương thích

Không tự động nâng môi trường của các pipeline khác. Nếu dùng chung environment,
cần smoke-test các import còn lại sau khi nâng `transformers`.

Cấu hình cloud đề xuất:

- GPU hỗ trợ BF16, ưu tiên VRAM từ 24 GB trở lên.
- Bắt đầu với batch size `1`; tăng dần lên `2`, `4` nếu VRAM cho phép.
- Dùng Flash Attention 2 khi cài đặt tương thích để giảm VRAM và tăng tốc.
- Model/cache nằm trên persistent volume để không tải lại checkpoint mỗi job.

## 5. Thứ tự triển khai

1. Cập nhật dependency trong environment cloud và smoke-test load model.
2. Sửa `config.py` để hai tiến trình extract/server dùng chung model config.
3. Sửa `extractor.py`; thử một folder nhỏ và xác nhận vector `(N, 512)` hợp lệ.
4. Extract toàn bộ keyframe trên cloud vào namespace Qwen mới.
5. Sửa `database.py`, validate artifact và rebuild Qdrant collection.
6. Sửa `server.py` để encode text bằng Qwen.
7. Chạy smoke test endpoint với request hiện có, kiểm tra response không thay đổi.
8. Chỉ sau khi server Qwen hoạt động mới xoá/archive artifact và index CLIP cũ.

## 6. Điều kiện hoàn thành

- Extractor tạo được embedding Qwen 512 chiều cho toàn bộ keyframe trên cloud.
- Không có vector CLIP trong artifact/index Qwen.
- Database nạp đủ vector và giữ nguyên payload hiện tại.
- Server dùng đúng model/revision/dimension với extractor.
- Endpoint, request schema, response schema và logic search hiện tại không đổi.
- Một request cũ của frontend/backend gọi thành công mà không cần sửa phía client.

## 7. Ngoài phạm vi

- Benchmark CLIP và Qwen.
- Thay đổi API contract hoặc UI.
- Thay đổi filter, candidate retrieval, temporal deduplication hoặc scoring.
- Thêm reranker, video-level embedding hay thuật toán retrieval mới.
- Refactor thành các file/module code mới ngoài bốn file hiện có.
