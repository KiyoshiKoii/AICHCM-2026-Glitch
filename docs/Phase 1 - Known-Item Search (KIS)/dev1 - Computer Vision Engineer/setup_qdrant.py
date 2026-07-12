import os
import sys
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Fix encoding cho in tiếng Việt trên Terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Khai báo đường dẫn
db_path = os.path.join(script_dir, "local_qdrant_db")

# 1. Khởi tạo Qdrant Client (Chế độ Local Storage - Không cần Docker)
print(f"Đang khởi tạo Qdrant DB tại: {db_path}")
client = QdrantClient(path=db_path)

collection_name = "kis_images"
vector_size = 512

# 2. Reset Collection (Xóa cũ đi tạo lại để tránh trùng lặp dữ liệu)
if client.collection_exists(collection_name=collection_name):
    print(f"Phát hiện Collection '{collection_name}' đã tồn tại. Đang xóa để nạp lại từ đầu...")
    client.delete_collection(collection_name=collection_name)

print(f"Đang tạo Collection mới: '{collection_name}' (kích thước {vector_size}, khoảng cách Cosine)...")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)
print("Tạo Collection thành công!\n")

# 3. Quét tất cả file .npy trong thư mục npy_features và nạp dữ liệu
npy_dir = os.path.join(script_dir, "npy_features")

if not os.path.exists(npy_dir):
    print(f"Không tìm thấy thư mục chứa file npy: {npy_dir}")
    npy_files = []
else:
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy") and f.startswith("L")]

npy_files.sort()

if not npy_files:
    print("Không tìm thấy file .npy nào bắt đầu bằng 'L' trong thư mục npy_features!")
else:
    print(f"Tìm thấy {len(npy_files)} file .npy. Bắt đầu xử lý hàng loạt...")
    
    global_id = 0 # Biến ID toàn cục đếm liên tục cho tất cả các ảnh của mọi thư mục
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Thư mục gốc chứa toàn bộ các folder ảnh (Keyframes)
    keyframe_dir = os.path.join(script_dir, "keyframe")
    
    for npy_filename in npy_files:
        video_name = npy_filename.replace(".npy", "")
        npy_path = os.path.join(npy_dir, npy_filename)
        image_dir = os.path.join(keyframe_dir, video_name)
        
        print(f"\n--- Đang xử lý Video: {video_name} ---")
        features = np.load(npy_path)
        num_vectors = features.shape[0]
        
        image_filenames = []
        if os.path.exists(image_dir):
            for f in os.listdir(image_dir):
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    image_filenames.append(f)
            # Quan trọng: Cần phải sort giống hệt bên file embed_folder.py
            image_filenames.sort()
            
        if len(image_filenames) != num_vectors:
            print(f"CẢNH BÁO: Số ảnh ({len(image_filenames)}) không khớp số vector ({num_vectors})!")
            
        points = []
        for i in range(num_vectors):
            vector = features[i].tolist() # Chuyển numpy array về list
            frame_name = image_filenames[i] if i < len(image_filenames) else f"unknown_{i}"
            
            # Gắn Payload MỚI: Có cả tên video và tên frame ảnh
            payload = {
                "video_id": video_name,
                "frame_id": frame_name
            }
            
            points.append(
                PointStruct(id=global_id, vector=vector, payload=payload)
            )
            global_id += 1 # Tăng ID lên cho ảnh tiếp theo
            
        # Nạp dữ liệu vào Qdrant
        client.upload_points(
            collection_name=collection_name,
            points=points
        )
        print(f"Nạp thành công {num_vectors} vector. (ID chạy đến {global_id - 1})")
        
    # Xác thực lại tổng số vector trong DB
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"\n{'='*50}")
    print(f"=> TÌNH TRẠNG DB HIỆN TẠI: Đã lưu trữ tổng cộng {collection_info.points_count} vector hoàn chỉnh!")
