# Audio Pipeline (ASR) — Dev 4, Phase 2

Trích xuất transcript tiếng Việt từ video gốc, output ra `data/metadata/metadata_asr.json`
theo format quy định trong
[`task.md`](../../docs/Phase%202%20-%20Advanced%20Search%20&%20Scaling/dev4%20-%20Full-Stack%20Engineer/task.md).

## Luồng xử lý

```text
[luồng nền] ffmpeg tách audio (WAV 16kHz mono) ─┐
                                                 ├─> hàng đợi ─> [luồng chính] GPU transcribe ─> checkpoint JSON
[luồng nền] ffmpeg tách audio (WAV 16kHz mono) ─┘
```

3 quyết định thiết kế chính:

1. **Tách audio trước bằng ffmpeg** thay vì ném thẳng `.mp4` vào faster-whisper.
   Video của BTC nặng ~130MB/file; cờ `-vn` của ffmpeg bỏ hẳn luồng video nên chỉ đọc
   phần dữ liệu thực sự cần. Đồng thời việc này chạy được ở **tiến trình riêng**, nên
   audio của video kế tiếp được chuẩn bị sẵn trong lúc GPU còn đang bận — GPU không phải chờ I/O.
2. **PhoWhisper thay cho Whisper gốc** (xem mục bên dưới).
3. **Batched inference trên GPU**: VAD cắt audio thành các đoạn có tiếng nói rồi nhồi
   `batch_size` đoạn vào GPU mỗi lượt forward, thay vì chạy tuần tự từng đoạn.

### ⚠️ `chunk_length` quyết định độ mịn của timestamp

PhoWhisper **không sinh được timestamp token** như Whisper gốc (fine-tune của VinAI đã
bỏ phần này — ép `without_timestamps=False` sẽ ra mốc rác kiểu `[5.04-5.18]` cho cả câu
dài, kể cả khi chạy tuần tự không batch). Nên `start`/`end` của segment được suy ra từ
**ranh giới chunk của VAD**, tức `--chunk-length` quyết định luôn độ mịn segment:

| `chunk_length` | segment trung bình | ghi chú |
|---|---|---|
| 30 (mặc định của faster-whisper) | ~25s | quá thô, tìm mốc thời gian gần như vô dụng |
| 10 | ~8s | |
| **5 (mặc định của pipeline này)** | **~4s** | tương đương Whisper gốc (~2.7s) |

Task yêu cầu segment để "hỗ trợ tìm kiếm mốc thời gian chính xác" nên mặc định để 5.
Đánh đổi: chunk nhỏ thì chậm hơn và thỉnh thoảng cắt ngang câu
(VD `"...đáng chú ý."` / `"ý sau đây."`). Cần câu trọn vẹn hơn thì tăng `--chunk-length`.

### Hiệu năng (đo trên RTX 4060 Laptop, video `L21_V002`, 1057s audio)

| Cấu hình | Thời gian | RTF | Segment |
|---|---|---|---|
| large-v3, tuần tự, đọc thẳng `.mp4` (bản đầu) | 193.1s | 0.183 | 380 (~2.7s) |
| PhoWhisper, batch 8, `chunk_length=30` | 135.9s | 0.129 | 40 (~25.6s ❌) |
| **PhoWhisper, batch 8, `chunk_length=5`** | **145.1s** | **0.137** | **252 (~4.0s)** |

→ nhanh hơn bản đầu **~1.33x** mà vẫn giữ được độ mịn timestamp.
Ước tính chạy hết 873 video (tổng 470,428s audio ≈ 130.7 giờ): **~18 giờ** (bản đầu ~24 giờ).

## Model: vì sao dùng PhoWhisper?

[PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) (VinAI) fine-tune Whisper trên
**844 giờ tiếng Việt** đa giọng vùng miền, đạt SOTA trên các benchmark ASR tiếng Việt.
WER càng thấp càng tốt:

| Model | CMV–Vi | VIVOS | VLSP 2020 T1 | VLSP 2020 T2 |
|---|---|---|---|---|
| PhoWhisper-small | 11.08 | 6.33 | 15.93 | 32.96 |
| PhoWhisper-medium | 8.27 | 4.97 | 14.12 | 26.85 |
| **PhoWhisper-large** | **8.14** | **4.67** | **13.75** | **26.68** |

Dataset của BTC là bản tin thời sự tiếng Việt (HTV) nên đây là lựa chọn mặc định.

VinAI phát hành model dạng HuggingFace transformers, phải convert 1 lần sang CTranslate2:

```bash
pip install torch transformers        # chỉ cần cho bước convert, bản CPU là đủ
python convert_model.py               # -> data/models/PhoWhisper-large-ct2/
```

`extractor.py` tự dùng model này khi có; nếu chưa convert thì tự fallback về `large-v3`.

> ⚠️ PhoWhisper là model **chuyên tiếng Việt**, nên `--language` mặc định ép sẵn `vi`.
> Nếu data có lẫn tiếng Anh cần nhận dạng, dùng Whisper gốc đa ngữ và bật tự nhận diện:
> `python extractor.py --model large-v3 --language ""`

### ⚠️ Lưu ý cho Dev 2 (BM25 / Elasticsearch): PhoWhisper KHÔNG chuẩn hoá số & chữ hoa

PhoWhisper xuất text đã normalize hoàn toàn: **chữ thường hết, không dấu câu, số viết
thành chữ**. Đo trên `L21_V002` (transcript ~18k ký tự):

| | PhoWhisper | Whisper large-v3 |
|---|---|---|
| chữ số (`0-9`) | **0** | 223 |
| chữ HOA | **0** | 473 |
| dấu `.` `,` | 23 | 127 |

Hệ quả trực tiếp lên tìm kiếm văn bản:

| Truy vấn | PhoWhisper | large-v3 |
|---|---|---|
| `60 giây` | ❌ (ghi "sáu mươi giây") | ✅ |
| `2024` | ❌ (ghi "hai ngàn hai mươi bốn") | ✅ |
| `2 tháng 9` | ❌ (ghi "hai tháng chín") | ✅ |
| `TP.HCM` | ❌ (ghi "thành phố hồ chí minh") | ✅ |

Đổi lại PhoWhisper đọc đúng từ tiếng Việt hơn hẳn: `quả bưởi` (large-v3 nghe thành
"quả mưỡi"), `Bộ Lao động Thương binh và Xã hội` (large-v3: "Thương minh").

**Khi index vào Elasticsearch nên xử lý thêm:** hoặc normalize luôn câu truy vấn của
người dùng về dạng chữ (số → chữ) trước khi search, hoặc index song song thêm một
trường sinh từ `large-v3` cho các truy vấn dạng số/viết tắt.
Nếu ưu tiên khớp số/viết tắt hơn độ chính xác ngữ âm, đổi model:
`python extractor.py --model large-v3 --language ""`

## Cài đặt

```bash
conda activate aichcm2026
pip install -r requirements.txt   # đã bao gồm faster-whisper
```

Cần **ffmpeg** trong PATH (`winget install Gyan.FFmpeg` / `sudo apt install ffmpeg`).

Video gốc thả vào `data/videos/` (tên file khớp với thư mục trong `data/keyframes/`, VD: `L21_V001.mp4`).
Thư mục này bị chặn trong `.gitignore` — nếu video đã có sẵn ở nơi khác trên máy,
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

**Hết VRAM (CUDA out of memory)?** Giảm `--batch-size` (mặc định 8). Card 8GB chạy model
large float16 ổn ở batch 8; card 12GB+ có thể nâng lên 16 để nhanh hơn.

Không muốn setup GPU thì chạy `--device cpu` — chậm hơn nhiều lần, nên đổi sang model nhẹ hơn:

```bash
python extractor.py --device cpu --model medium --batch-size 4
```

## Sử dụng

```bash
# Quét toàn bộ data/videos/, resume tự động (bỏ qua video đã có trong output)
python extractor.py

# Test nhanh 1 video cụ thể
python extractor.py --video ../../data/videos/L21_V001.mp4

# Chỉ lấy N video đầu (test nhanh trên tập nhỏ)
python extractor.py --limit 3

# Tăng throughput trên card VRAM lớn
python extractor.py --batch-size 16 --audio-workers 4

# Xử lý lại từ đầu (bỏ qua resume logic)
python extractor.py --overwrite
```

Chạy được nhiều phiên: checkpoint ghi sau **mỗi video**, nên `Ctrl+C` giữa chừng rồi chạy
lại sẽ tự bỏ qua các video đã xong. File WAV tạm cũng tự xoá sau mỗi video (giữ lại bằng
`--keep-audio` nếu cần debug).

Xem toàn bộ flag qua `python extractor.py --help`.

## Test

```bash
pytest tests/ -v
```

Các test dùng stub thay cho `ASRExtractor` và `extract_audio` nên **không cần GPU/model**
để chạy — chỉ kiểm tra logic thuần Python (chuẩn hoá tên video, build segment/transcript,
resume/checkpoint, xử lý lỗi ffmpeg). Riêng `TestExtractAudioReal` có chạy ffmpeg thật
trên 1 video 1 giây tự sinh, tự động skip nếu máy chưa cài ffmpeg.
