import os
import sys
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    KEYFRAME_DIR, NPY_DIR, QDRANT_DB_PATH,
    COLLECTION_NAME, VECTOR_SIZE,
    print_config
)

# Fix encoding cho in tiếng Việt trên Terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────────────────────
# Kiểm tra điều kiện tiên quyết
# ──────────────────────────────────────────────────────────────────────────────
print_config()

if not os.path.exists(NPY_DIR):
    print(f"❌ Chưa có thư mục npy_features tại: {NPY_DIR}")
    print("👉 Hãy chạy extractor.py trước để sinh ra các file .npy")
    sys.exit(0)

npy_files = sorted([
    f for f in os.listdir(NPY_DIR)
    if f.endswith(".npy") and f.startswith("L") and not f.endswith("_filenames.npy")
])

if not npy_files:
    print(f"⚠️  Không tìm thấy file .npy nào trong: {NPY_DIR}")
    print("👉 Hãy chạy extractor.py trước.")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo Qdrant Client (Local Storage — không cần Docker)
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(QDRANT_DB_PATH, exist_ok=True)
print(f"Đang khởi tạo Qdrant DB tại: {QDRANT_DB_PATH}")
client = QdrantClient(path=QDRANT_DB_PATH)

# Reset Collection (xóa & tạo lại để tránh trùng dữ liệu)
if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Phát hiện Collection '{COLLECTION_NAME}' đã tồn tại. Đang xóa để nạp lại...")
    client.delete_collection(collection_name=COLLECTION_NAME)

print(f"Đang tạo Collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, Cosine)...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print("✅ Tạo Collection thành công!\n")

# ──────────────────────────────────────────────────────────────────────────────
# Nạp dữ liệu từ các file .npy vào Qdrant (Tối ưu Batch 5000 points)
# ──────────────────────────────────────────────────────────────────────────────
print(f"Tìm thấy {len(npy_files)} file .npy. Bắt đầu nạp dữ liệu...\n")

global_id = 0
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
points_buffer = []
UPLOAD_BATCH_SIZE = 5000

for idx, npy_filename in enumerate(npy_files, 1):
    video_name = npy_filename.replace(".npy", "")
    npy_path = os.path.join(NPY_DIR, npy_filename)
    names_path = os.path.join(NPY_DIR, f"{video_name}_filenames.npy")

    features = np.load(npy_path)
    num_vectors = features.shape[0]

    if os.path.exists(names_path):
        image_filenames = list(np.load(names_path, allow_pickle=True))
    else:
        image_dir = os.path.join(KEYFRAME_DIR, video_name)
        if os.path.exists(image_dir):
            image_filenames = sorted([
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ])
        else:
            image_filenames = []

    for i in range(num_vectors):
        vector = features[i].tolist()
        frame_name = image_filenames[i] if i < len(image_filenames) else f"unknown_{i}"

        try:
            frame_index = int(os.path.splitext(frame_name)[0])
        except ValueError:
            frame_index = i

        points_buffer.append(PointStruct(
            id=global_id,
            vector=vector,
            payload={
                "video_id": video_name,
                "frame_id": frame_name,
                "frame_index": frame_index,
            }
        ))
        global_id += 1

    # Nạp vào Qdrant khi bộ đệm đủ 5000 points
    if len(points_buffer) >= UPLOAD_BATCH_SIZE:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points_buffer
        )
        print(f"  ⚡ [{idx}/{len(npy_files)} videos] Đã nạp batch {len(points_buffer)} vectors (Tổng: {global_id})")
        points_buffer = []

# Nạp nốt số points còn lại trong bộ đệm
if points_buffer:
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points_buffer
    )
    print(f"  ⚡ Hoàn tất batch cuối {len(points_buffer)} vectors (Tổng: {global_id})")

# ──────────────────────────────────────────────────────────────────────────────
# Xác nhận kết quả cuối
# ──────────────────────────────────────────────────────────────────────────────
collection_info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"\n{'='*55}")
print(f"✅ HOÀN TẤT! DB đang lưu: {collection_info.points_count} vector")
print(f"   DB path: {QDRANT_DB_PATH}")

# Đóng client chủ động để tránh cảnh báo rác của Python khi thoát trên Windows
client.close()
