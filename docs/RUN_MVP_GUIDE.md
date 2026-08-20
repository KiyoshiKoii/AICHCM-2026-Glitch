# Hướng dẫn chạy hệ thống MVP

Tài liệu này mô tả cách khởi động hệ thống AICHCM-2026-Glitch trên máy local.
MVP gồm bốn dịch vụ:

| Dịch vụ | Port | Vai trò |
| --- | ---: | --- |
| Visual Pipeline | `8001` | CLIP + Qdrant, tìm kiếm theo hình ảnh |
| Semantic Pipeline | `8002` | Elasticsearch, caption và TRAKE video selection |
| Backend Gateway | `8000` | Phân tích query, hợp nhất kết quả, phục vụ media |
| Frontend | `5173` | Web UI |

> Tất cả lệnh trong tài liệu này được chạy từ **repository root**, trừ khi có
> ghi chú khác. Mỗi khối Git Bash và PowerShell là hai phiên bản tương đương;
> chỉ chọn một khối phù hợp với terminal đang dùng.

---

## 1. Yêu cầu tiên quyết

### 1.1. Tạo môi trường Conda

Chạy một lần sau khi clone repository.

**Git Bash**

```bash
conda env create -f environment.yml
conda activate aichcm2026
```

**PowerShell**

```powershell
conda env create -f .\environment.yml
conda activate aichcm2026
```

Nếu môi trường `aichcm2026` đã tồn tại và `environment.yml` vừa thay đổi, dùng:

**Git Bash**

```bash
conda env update -n aichcm2026 -f environment.yml --prune
```

**PowerShell**

```powershell
conda env update -n aichcm2026 -f .\environment.yml --prune
```

Nếu `conda activate` chưa hoạt động trong terminal, chạy `conda init bash` cho
Git Bash hoặc `conda init powershell` cho PowerShell, sau đó mở terminal mới.

### 1.2. Cài frontend dependencies

Frontend hiện dùng Vite 8, yêu cầu Node.js `^20.19`, `^22.12` hoặc `>=24`.
Repository đã có `package-lock.json`, vì vậy ưu tiên `npm ci`.

**Git Bash**

```bash
node --version
npm --prefix src/frontend ci
```

**PowerShell**

```powershell
node --version
npm.cmd --prefix .\src\frontend ci
```

`npm.cmd` được dùng trong PowerShell để không bị chặn bởi execution policy của
`npm.ps1` trên Windows.

### 1.3. Docker và cấu hình Gemini

Kiểm tra Docker Desktop đang chạy:

```text
docker compose version
```

Tạo file `.env` tại repository root và cấu hình ít nhất:

```dotenv
GEMINI_API_KEY=your_api_key
GEMINI_QUERY_MODEL=gemini-model-name
GEMINI_RERANK_MODEL=gemini-model-name
GEMINI_VQA_MODEL=gemini-model-name
GEMINI_VISUAL_MODEL=gemini-model-name
VISUAL_CLIP_MODEL=openai/clip-vit-base-patch32
```

Không commit API key lên Git.

### 1.4. Dữ liệu cần có

| Đường dẫn | Dùng cho |
| --- | --- |
| `data/keyframes/` | Thumbnail, CLIP và filmstrip của video player |
| `data/videos/` | Phát và seek video từ UI |
| `data/map-keyframes/` | Đổi keyframe sang frame gốc, timestamp và FPS |
| `data/npy_features/` | Vector CLIP đã trích xuất |
| `data/metadata/caption/` | Caption/OCR cho Elasticsearch |
| `data/processed/video_understanding/` | Summary/timeline cho TRAKE |
| `src/visual_pipeline/local_qdrant_db/` | Qdrant local của Visual Pipeline |

Keyframes, videos, NPY features và local Qdrant không được Git quản lý đầy đủ.
Thành viên mới có thể tải dữ liệu dùng chung từ
[Google Sheet dữ liệu của nhóm](https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0).

---

## 2. Chuẩn bị index lần đầu

Không chạy lại toàn bộ phần này trong mỗi lần khởi động. Chỉ chạy khi setup máy
mới hoặc khi dữ liệu/index đã thay đổi.

### 2.1. Tạo CLIP features và Qdrant local

Nếu đã có cả `data/npy_features/` và
`src/visual_pipeline/local_qdrant_db/`, bỏ qua bước này.

**Git Bash**

```bash
conda activate aichcm2026
python src/visual_pipeline/extractor.py
python src/visual_pipeline/database.py
```

**PowerShell**

```powershell
conda activate aichcm2026
python .\src\visual_pipeline\extractor.py
python .\src\visual_pipeline\database.py
```

> `database.py` tạo lại collection Qdrant từ các file NPY. Chỉ chạy lại khi
> cần cập nhật Visual index và phải tắt Visual Pipeline trước khi chạy.

### 2.2. Tạo Elasticsearch frame index

Khởi động Elasticsearch:

**Git Bash**

```bash
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml up -d
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml ps
```

**PowerShell**

```powershell
docker compose -f .\src\semantic_pipeline\retrieval\docker-compose.elasticsearch.yml up -d
docker compose -f .\src\semantic_pipeline\retrieval\docker-compose.elasticsearch.yml ps
```

Sau khi container báo `healthy`, tạo index `semantic_frames_v4`, nạp caption và
trỏ alias ổn định `semantic_frames` vào index đó:

**Git Bash**

```bash
conda activate aichcm2026
python src/semantic_pipeline/elasticsearch_service.py bootstrap \
  --caption-dir data/metadata/caption
```

**PowerShell**

```powershell
conda activate aichcm2026
python .\src\semantic_pipeline\elasticsearch_service.py bootstrap `
  --caption-dir .\data\metadata\caption
```

Sau này, khi chỉ bổ sung caption cho một batch, không cần tạo lại toàn bộ index:

**Git Bash**

```bash
python src/semantic_pipeline/elasticsearch_service.py ingest \
  --caption-dir data/metadata/caption \
  --batch-id L26
```

**PowerShell**

```powershell
python .\src\semantic_pipeline\elasticsearch_service.py ingest `
  --caption-dir .\data\metadata\caption `
  --batch-id L26
```

Có thể lặp `--batch-id` để nạp nhiều batch trong cùng một lệnh.

### 2.3. Tạo video-summary index cho TRAKE

Bước này đọc các file
`data/processed/video_understanding/LXX/LXX_VYYY/pilot/video_summary.json`.
Chạy lại sau khi sinh thêm summary.

**Git Bash**

```bash
conda activate aichcm2026
python src/semantic_pipeline/retrieval/video_selector_cli.py bootstrap \
  --output-root data/processed/video_understanding
```

**PowerShell**

```powershell
conda activate aichcm2026
python .\src\semantic_pipeline\retrieval\video_selector_cli.py bootstrap `
  --output-root .\data\processed\video_understanding
```

Để chỉ cập nhật một batch, thêm `--batch-id L26`.

### 2.4. Tạo timestamped ASR index

ASR Search đọc transcript theo từng video trong
`data/metadata/metadata_asr`. Mỗi document giữ nguyên khoảng `start_ms/end_ms`,
vì vậy kết quả có thể mở video đúng đầu đoạn thoại thay vì chỉ về keyframe.
Chạy bootstrap sau khi có ASR metadata mới:

**Git Bash**

```bash
conda activate aichcm2026
PYTHONPATH=src python -m semantic_pipeline.retrieval.asr_cli bootstrap \
  --asr-dir data/metadata/metadata_asr
```

**PowerShell**

```powershell
conda activate aichcm2026
$env:PYTHONPATH='src'
python -m semantic_pipeline.retrieval.asr_cli bootstrap `
  --asr-dir .\data\metadata\metadata_asr
```

Để chỉ nạp hoặc cập nhật một batch, thêm `--batch-id L26`. Có thể kiểm tra index
trực tiếp trước khi mở UI bằng lệnh sau:

**Git Bash**

```bash
PYTHONPATH=src python -m semantic_pipeline.retrieval.asr_cli search \
  "cho dầu vào chảo" --batch-id L26 --top-k 10
```

**PowerShell**

```powershell
$env:PYTHONPATH='src'
python -m semantic_pipeline.retrieval.asr_cli search `
  "cho dầu vào chảo" --batch-id L26 --top-k 10
```

---

## 3. Khởi động MVP hằng ngày

Mở bốn terminal độc lập. Trong mỗi terminal, chuyển tới repository root rồi
chạy đúng dịch vụ tương ứng.

### Terminal 1 — Visual Pipeline (`8001`)

**Git Bash**

```bash
conda activate aichcm2026
python src/visual_pipeline/server.py
```

**PowerShell**

```powershell
conda activate aichcm2026
python .\src\visual_pipeline\server.py
```

Visual Pipeline dùng Qdrant embedded nên lúc khởi động có thể cần nhiều RAM và
mất thời gian nếu local database lớn. Không chạy `database.py` đồng thời với
server này.

### Terminal 2 — Semantic Pipeline (`8002`)

**Git Bash**

```bash
conda activate aichcm2026
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml up -d
uvicorn semantic_pipeline.retrieval.api:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8002
```

**PowerShell**

```powershell
conda activate aichcm2026
docker compose -f .\src\semantic_pipeline\retrieval\docker-compose.elasticsearch.yml up -d
uvicorn semantic_pipeline.retrieval.api:app `
  --app-dir .\src `
  --host 127.0.0.1 `
  --port 8002
```

Semantic API mặc định đọc alias `semantic_frames`. Không cần đặt
`PYTHONPATH` hoặc `ELASTICSEARCH_INDEX` khi chạy theo guide này.

### Terminal 3 — Backend Gateway (`8000`)

**Git Bash**

```bash
conda activate aichcm2026
uvicorn backend.main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

**PowerShell**

```powershell
conda activate aichcm2026
uvicorn backend.main:app `
  --app-dir .\src `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

Swagger UI: <http://127.0.0.1:8000/docs>

### Terminal 4 — Frontend (`5173`)

**Git Bash**

```bash
npm --prefix src/frontend run dev -- \
  --host 127.0.0.1 \
  --port 5173
```

**PowerShell**

```powershell
npm.cmd --prefix .\src\frontend run dev -- `
  --host 127.0.0.1 `
  --port 5173
```

Mở UI tại <http://127.0.0.1:5173>. Vite tự proxy `/api` và `/media` sang
Backend port `8000`.

---

## 4. Kiểm tra nhanh sau khi khởi động

**Git Bash**

```bash
curl -f http://127.0.0.1:9200/_cluster/health
curl -f http://127.0.0.1:8001/docs >/dev/null
curl -f http://127.0.0.1:8002/health
curl -f http://127.0.0.1:8000/docs >/dev/null
curl -f http://127.0.0.1:5173/ >/dev/null
```

**PowerShell**

```powershell
Invoke-RestMethod http://127.0.0.1:9200/_cluster/health
Invoke-WebRequest http://127.0.0.1:8001/docs -UseBasicParsing
Invoke-RestMethod http://127.0.0.1:8002/health
Invoke-WebRequest http://127.0.0.1:8000/docs -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:5173/ -UseBasicParsing
```

Nếu Semantic health trả `503`, kiểm tra Elasticsearch đã `healthy` và alias
`semantic_frames` đã được tạo bằng lệnh `bootstrap` ở mục 2.2.

---

## 5. Resume repair metadata bị CLIP flag

Đây là lệnh bảo trì dữ liệu, không phải lệnh bắt buộc để khởi động MVP. Chạy từ
repository root. `--resume` bỏ qua các frame đã ghi trong checkpoint
`data/metadata/caption/.repair_resume.json`.

**Git Bash**

```bash
conda activate aichcm2026
python src/semantic_pipeline/repair_flagged_metadata.py \
  --report data/metadata/verification/clip_L25_full_report.json \
  --keyframe-dir data/keyframes \
  --output-dir data/metadata/caption \
  --batch-size 20 \
  --resume
```

**PowerShell**

```powershell
conda activate aichcm2026
python .\src\semantic_pipeline\repair_flagged_metadata.py `
  --report .\data\metadata\verification\clip_L25_full_report.json `
  --keyframe-dir .\data\keyframes `
  --output-dir .\data\metadata\caption `
  --batch-size 20 `
  --resume
```

Muốn dùng checkpoint riêng, thêm
`--resume-state data/metadata/caption/<job-name>.json`. Bỏ `--resume` khi chủ
động repair lại toàn bộ frame đang bị flag.

---

## 6. Tắt hệ thống

Trong bốn terminal dịch vụ, nhấn `Ctrl+C`. Đảm bảo Visual Pipeline đã dừng hoàn
toàn để giải phóng VRAM và file lock của Qdrant.

Sau đó tắt Elasticsearch:

**Git Bash**

```bash
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml down
```

**PowerShell**

```powershell
docker compose -f .\src\semantic_pipeline\retrieval\docker-compose.elasticsearch.yml down
```

Lệnh `down` giữ nguyên Docker volume. Không thêm `-v` trừ khi thực sự muốn xóa
toàn bộ Elasticsearch data và tạo index lại từ đầu.
