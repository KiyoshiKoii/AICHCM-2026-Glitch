import numpy as np
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Lấy đường dẫn tuyệt đối của thư mục chứa script
script_dir = os.path.dirname(os.path.abspath(__file__))
npy_file = os.path.join(script_dir, "L25_V001.npy")

if os.path.exists(npy_file):
    # Đọc ma trận từ file .npy
    features = np.load(npy_file)
    print(f"Kích thước toàn bộ ma trận: {features.shape}")
    
    # Lấy 10 dòng đầu tiên
    first_10_vectors = features[:10]
    print(f"\nKích thước ma trận của 10 dòng đầu: {first_10_vectors.shape}")
    
    # In thử 10 dòng đầu (chỉ in 5 giá trị đầu tiên của mỗi vector để dễ nhìn)
    print("\nChi tiết 10 vector đầu tiên (hiển thị 5 chiều đầu):")
    for i in range(len(first_10_vectors)):
        print(f"Ảnh {i+1:02d}: {first_10_vectors[i][:20]} ...")
else:
    print(f"Không tìm thấy file: {npy_file}")
