# Audio Pipeline (ASR) — Dev 4, Phase 2

Trích xuất transcript (tiếng Việt / tiếng Anh) từ video gốc bằng `faster-whisper`, output ra
`data/metadata/metadata_asr.json` theo format quy định trong
[`task.md`](../../docs/Phase%202%20-%20Advanced%20Search%20&%20Scaling/dev4%20-%20Full-Stack%20Engineer/task.md).

## Cài đặt

```bash
conda activate aichcm2026
pip install -r requirements.txt   # đã bao gồm faster-whisper
```

Video gốc thả vào `data/videos/` (tên file khớp với thư mục trong `data/keyframes/`, VD: `L21_V001.mp4`).
Thư mục này bị chặn trong `.gitignore` (giống `data/keyframes/`) — nếu video đã có sẵn ở nơi khác trên máy,
có thể trỏ vào đó bằng **directory junction** thay vì copy:

```powershell
# Windows — không cần quyền admin (khác với mklink /D)
cmd /c mklink /J "data\videos" "D:\đường\dẫn\đến\video\gốc"
```

## ⚠️ Setup GPU (CUDA)

Mặc định `--device auto` sẽ dùng GPU nếu có. Nhưng riêng gói GPU của `ctranslate2` (backend của
`faster-whisper`) **không tự mang theo cuBLAS/cuDNN** — nếu máy chỉ cài driver NVIDIA (không cài full
CUDA Toolkit), chạy `--device cuda` sẽ lỗi:

```
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

Cách sửa nhanh nhất — cài 2 gói pip mang theo sẵn DLL, không cần cài CUDA Toolkit:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Rồi thêm thư mục chứa DLL của chúng vào `PATH` **trước khi chạy** (Windows không tự nạp DLL từ
`site-packages` như Linux nạp `.so`):

```powershell
# PowerShell — thay <env> bằng đường dẫn conda env đang dùng
$py = "<env>\Lib\site-packages"
$env:PATH = "$py\nvidia\cublas\bin;$py\nvidia\cudnn\bin;$env:PATH"
python extractor.py --device cuda
```

```bash
# Git Bash
PY_SITE="<env>/Lib/site-packages"
PATH="$PY_SITE/nvidia/cublas/bin:$PY_SITE/nvidia/cudnn/bin:$PATH" python extractor.py --device cuda
```

Không muốn setup GPU thì bỏ qua bước trên và chạy `--device cpu` — chậm hơn nhiều lần, nên đổi sang
model nhẹ hơn cho đỡ chờ:

```bash
python extractor.py --device cpu --model-size medium
```

## Sử dụng

```bash
# Quét toàn bộ data/videos/, resume tự động (bỏ qua video đã có trong output)
python extractor.py

# Test nhanh 1 video cụ thể
python extractor.py --video ../../data/videos/L21_V001.mp4 --device cuda

# Xử lý lại từ đầu (bỏ qua resume logic)
python extractor.py --overwrite

# Chỉ lấy N video đầu (test nhanh trên tập nhỏ)
python extractor.py --limit 3
```

Xem toàn bộ flag qua `python extractor.py --help`.

## Test

```bash
pytest tests/ -v
```

Các test dùng stub thay cho `ASRExtractor` thật nên **không cần cài faster-whisper/model** để chạy —
chỉ kiểm tra logic thuần Python (chuẩn hoá tên video, build segment/transcript, resume/checkpoint).
