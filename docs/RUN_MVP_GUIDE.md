# Hướng dẫn Chạy hệ thống MVP (Run MVP Guide)

Tài liệu này hướng dẫn cách khởi động toàn bộ hệ thống AICHCM-2026-Glitch (Giai đoạn MVP - Phase 1) trên máy Local để phục vụ quá trình test và phát triển.

Hệ thống kiến trúc Microservices bao gồm 4 luồng server chạy song song:
1. **Visual Pipeline API** (Port 8001)
2. **Semantic Pipeline API** (Port 8002)
3. **Backend API Tổng** (Port 8000)
4. **Frontend Web UI** (Port mặc định của framework: 5173 hoặc 3000)

---

## 🛠 Yêu cầu tiên quyết & Dữ liệu

1. **Cài đặt môi trường Conda:**
   Chạy lệnh sau để tự động khởi tạo môi trường Python từ file `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate aichcm2026
   ```

2. **Cài đặt Node.js & Dependencies cho Frontend:** Đảm bảo máy đã cài Node.js (v18+). Chạy lệnh cài thư viện Frontend:
   ```bash
   cd src/frontend
   npm install
   ```

3. **Tải Dữ liệu Hình ảnh (Keyframes) & Objects:** 
   - Vì hai thư mục `data/keyframes/` (ảnh thô) và `data/objects/` (hàng vạn file JSON vật thể) bị loại trừ bởi `.gitignore` để tránh nặng repo, thành viên mới khi clone dự án về cần truy cập vào bảng Google Sheet chứa link tải dữ liệu của team: [Google Sheet Link Dữ Liệu Keyframes & Objects](https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0).
   - Tiến hành tải các file/thư mục tương ứng:
     - Tải lần lượt các file có tên bắt đầu bằng chữ `keyframes` (ví dụ: `keyframes_L21.zip`, `keyframes_L22.zip`...), giải nén toàn bộ và đặt vào đúng đường dẫn: `data/keyframes/`.
     - Tải bộ dữ liệu `objects`, giải nén và đặt vào đúng đường dẫn: `data/objects/`.

---

## 🚀 Các bước Khởi động (Mở 4 Terminal độc lập)

### Terminal 1: Chạy Visual Pipeline (Dev 1)
Chịu trách nhiệm load model CLIP (ăn vào GPU) và truy vấn Qdrant Vector DB.
```bash
conda activate aichcm2026
cd src/visual_pipeline
python server.py
```
*Server nội bộ sẽ lắng nghe ở: http://localhost:8001*

### Terminal 2: Chạy Semantic Pipeline (Dev 2)
Chịu trách nhiệm chạy Elasticsearch-backed semantic retrieval và cung cấp API nội bộ ở port 8002.
```bash
# Chạy các lệnh này từ repository root.
conda activate aichcm2026
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml up -d

# Nạp metadata visual hiện tại vào index pilot; không đổi production alias.
export PYTHONPATH=src
python -m semantic_pipeline.retrieval.cli \
  --index-name semantic_frames_l21_v001_pilot \
  ingest \
  --caption-dir data/metadata/caption \
  --require-provenance

# Cho API semantic đọc index pilot trên port 8002.
export ELASTICSEARCH_INDEX=semantic_frames_l21_v001_pilot
uvicorn semantic_pipeline.retrieval.api:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8002
```
*Server nội bộ sẽ lắng nghe ở: http://localhost:8002*

### Resume repair metadata bị CLIP flag

Chạy từ repository root. Lệnh luôn ghi checkpoint sau mỗi batch repair đã ghi
caption thành công. Nếu bị dừng giữa chừng, chạy lại **đúng lệnh đó** với
`--resume`; các frame đã checkpoint sẽ không gọi lại Gemini.

```bash
export PYTHONPATH=src
python -m semantic_pipeline.gemini.repair \
  --report data/metadata/verification/clip_L25_full_report.json \
  --keyframe-dir data/keyframes \
  --output-dir data/metadata/caption \
  --batch-size 20 \
  --resume
```

Checkpoint mặc định là `data/metadata/caption/.repair_resume.json`. Có thể tách
checkpoint cho một job bằng `--resume-state <duong-dan-checkpoint.json>`. Bỏ
`--resume` khi muốn chủ động repair lại toàn bộ frame đang bị flag.

### Terminal 3: Chạy Backend Tổng (Dev 3)
Chịu trách nhiệm làm cầu nối (Gateway), gọi LLM để phân tích câu hỏi ra prompt tiếng Anh, sau đó gộp kết quả RRF.
```bash
conda activate aichcm2026
cd src
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Swagger UI (API Docs test nhanh): http://localhost:8000/docs*

### Terminal 4: Chạy Frontend UI (Dev 4)
Giao diện người dùng chính thức phục vụ Giám khảo / Thí sinh.
```bash
cd src/frontend
npm run dev
```
*Truy cập đường dẫn localhost hiển thị trên Terminal để sử dụng Web UI.*

---

## 🛑 Lưu ý khi Tắt hệ thống (Shutdown)
Để tắt hệ thống an toàn, bạn cần vào từng cửa sổ Terminal và nhấn tổ hợp phím `Ctrl + C`. 
**Đặc biệt lưu ý Terminal 1**: Cần đảm bảo tiến trình Python kết thúc hoàn toàn để giải phóng VRAM của Card đồ họa, tránh lỗi Out of Memory (OOM) cho lần khởi động sau.
