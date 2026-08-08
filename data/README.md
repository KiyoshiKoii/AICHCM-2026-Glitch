# 📦 Dataset AIC 2026

Thư mục này **không được commit** (xem `.gitignore`). Mỗi người tự tải data của BTC
về máy mình rồi gom vào đây theo đúng cấu trúc bên dưới, để mọi đường dẫn tương đối
trong code chạy giống nhau trên mọi máy.

Hai file duy nhất được commit trong đây: chính `data/README.md` này và
`data/raw/MANIFEST.csv`.

## Cấu trúc

```
data/
├── raw/                    # NGUYÊN BẢN của BTC — chỉ đọc, không bao giờ sửa/ghi đè
│   ├── keyframes/          # <video_id>/<nnn>.jpg
│   ├── clip-features-32/   # <video_id>.npy
│   ├── map-keyframes/      # <video_id>.csv   (n, pts_time, fps, frame_idx)
│   ├── media-info/         # <video_id>.json
│   ├── objects/            # <video_id>/<nnn>.json
│   └── MANIFEST.csv        # video_id, batch, group, n_keyframes, fps, ... (ĐƯỢC COMMIT)
├── processed/              # output do pipeline của team sinh ra — xoá lúc nào cũng được
│   ├── metadata/           # dev2: metadata_*.json
│   ├── objects_index/      # objects gộp theo video (parquet/jsonl) cho nhanh
│   ├── embeddings/
│   └── reports/            # audit/benchmark có thể tái sinh, không commit
└── external/               # repo/model tham khảo (vbs2023-..., checkpoint, v.v.)
```

`<video_id>` có dạng `L21_V001`, duy nhất trên toàn bộ dataset.

## Quy ước quan trọng

1. **Bỏ hậu tố batch trong tên thư mục.** BTC thả data làm nhiều đợt
   (`-aic25-b1`, `-b2`, ...) nhưng `video_id` là duy nhất toàn cục, nên ta gom
   theo *loại dữ liệu* thay vì theo *đợt phát hành*. Nhờ vậy mọi đường dẫn suy ra
   được trực tiếp từ `video_id`, không phải quét nhiều thư mục. Đợt nào chứa video
   nào thì tra trong `MANIFEST.csv`.
2. **`Keyframes_L21` và `Keyframes_L22` gộp chung vào `raw/keyframes/`.** Chúng chỉ
   là phân mảnh theo nhóm L của cùng một loại dữ liệu; thông tin nhóm đã nằm sẵn
   trong `video_id`.
3. **`raw/` là read-only.** Mọi thứ pipeline sinh ra đều đi vào `processed/`. Nhờ
   vậy lúc nào nghi ngờ data hỏng thì xoá sạch `processed/` chạy lại, không sợ mất
   bản gốc của BTC.

## Cách setup

```powershell
# 1. Giải nén / tải data của BTC ra thư mục gốc repo (hoặc bất kỳ đâu)
# 2. Chạy thử để xem script định làm gì (KHÔNG di chuyển gì cả):
powershell -File scripts/organize_data.ps1

# 3. Ưng thì chạy thật (move cùng ổ đĩa nên gần như tức thì):
powershell -File scripts/organize_data.ps1 -Execute

# 4. Sinh MANIFEST.csv:
python scripts/build_manifest.py
```

## Để data ở ổ đĩa khác

Cách 1 — biến môi trường (chỉ ảnh hưởng máy bạn):

```powershell
$env:AIC_DATA_ROOT = "E:\AIC_DATA"
```

Cách 2 — junction, giữ nguyên đường dẫn `data/` cho mọi tool:

```powershell
New-Item -ItemType Junction -Path data -Target E:\AIC_DATA
```

## Dùng trong code

Không hardcode đường dẫn. Luôn đi qua `src/common/paths.py`:

```python
from common import paths

paths.keyframe_dir("L21_V001")      # data/raw/keyframes/L21_V001
paths.clip_features("L21_V001")     # data/raw/clip-features-32/L21_V001.npy
paths.processed_metadata_dir()      # data/processed/metadata
```
