import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import (
    KEYFRAME_DIR, NPY_DIR,
    CLIP_MODEL_ID, BATCH_SIZE,
    print_config
)

# Fix encoding issue when printing Vietnamese characters in Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo model CLIP + auto device detection
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang tải model CLIP ({CLIP_MODEL_ID}) lên {device.upper()}...")
model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
model.to(device)
print("Model đã sẵn sàng.\n")


def embed_folder(folder_path: str):
    """
    Đọc tất cả ảnh trong folder_path và trả về (features, image_paths).
    - features: torch.Tensor shape (N, 512), đã L2-normalize
    - image_paths: list[str] — tên file tương ứng theo thứ tự
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder không tồn tại: {folder_path}")
        return None

    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not image_paths:
        print(f"  ⚠️  Không tìm thấy ảnh nào trong: {folder_path}")
        return None

    print(f"  Tìm thấy {len(image_paths)} ảnh. Bắt đầu trích xuất...")

    all_features = []

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]

        # Load ảnh — bỏ qua file lỗi thay vì crash cả batch
        images = []
        valid_batch_paths = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_batch_paths.append(p)
            except Exception as e:
                print(f"  ⚠️  Bỏ qua file lỗi ({os.path.basename(p)}): {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Một số phiên bản transformers trả về BaseModelOutputWithPooling
        # thay vì tensor trực tiếp — cần extract đúng trường
        if hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds
        elif hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        # L2 normalize (chuẩn cho cosine similarity)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        all_features.append(image_features.cpu())

        done = min(i + BATCH_SIZE, len(image_paths))
        print(f"  [{done}/{len(image_paths)}] batches xong...")

    if not all_features:
        return None

    final_features = torch.cat(all_features, dim=0)

    # Trả về cả image_paths để database.py dùng — tránh mismatch khi scan lại
    image_filenames = [os.path.basename(p) for p in image_paths]
    return final_features, image_filenames


# ──────────────────────────────────────────────────────────────────────────────
# Main — Chạy trực tiếp: python extractor.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_config()

    # Kiểm tra thư mục keyframe
    if not os.path.exists(KEYFRAME_DIR):
        print(f"❌ Chưa có thư mục keyframe tại: {KEYFRAME_DIR}")
        print("👉 Hãy tạo thư mục và thả data vào:")
        print(f"   {KEYFRAME_DIR}/")
        print(f"   ├── L25_V001/   (chứa các file .jpg/.png)")
        print(f"   ├── L25_V002/")
        print(f"   └── ...")
        sys.exit(0)

    # Tự tạo thư mục npy_features nếu chưa có
    os.makedirs(NPY_DIR, exist_ok=True)

    # Quét tất cả thư mục video bắt đầu bằng "L"
    video_folders = sorted([
        d for d in os.listdir(KEYFRAME_DIR)
        if os.path.isdir(os.path.join(KEYFRAME_DIR, d)) and d.startswith("L")
    ])

    if not video_folders:
        print(f"⚠️  Không tìm thấy thư mục video nào (bắt đầu bằng 'L') trong:")
        print(f"   {KEYFRAME_DIR}")
        sys.exit(0)

    print(f"Tìm thấy {len(video_folders)} thư mục video. Bắt đầu xử lý...\n")

    for folder_name in video_folders:
        print(f"{'='*55}")
        print(f"  VIDEO: {folder_name}")

        npy_path = os.path.join(NPY_DIR, f"{folder_name}.npy")
        names_path = os.path.join(NPY_DIR, f"{folder_name}_filenames.npy")

        # Resume logic: skip nếu đã xử lý rồi
        if os.path.exists(npy_path) and os.path.exists(names_path):
            print(f"  ✅ Đã có sẵn file .npy → Bỏ qua.")
            continue

        folder_path = os.path.join(KEYFRAME_DIR, folder_name)
        result = embed_folder(folder_path)

        if result is not None:
            features, filenames = result
            print(f"  Shape vector: {features.shape}")

            # Lưu features
            np.save(npy_path, features.numpy())
            # Lưu tên file kèm theo để database.py không cần scan lại
            np.save(names_path, np.array(filenames))

            print(f"  💾 Đã lưu: {os.path.basename(npy_path)}")
            print(f"  💾 Đã lưu: {os.path.basename(names_path)}")

    print(f"\n{'='*55}")
    print("✅ Hoàn tất trích xuất toàn bộ vector!")
    print(f"   Kết quả tại: {NPY_DIR}")
