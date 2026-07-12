import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Fix encoding issue when printing Vietnamese characters in Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo model CLIP
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Sử dụng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed_folder(folder_path):
    """
    Đọc tất cả ảnh trong folder_path và trả về một ma trận đặc trưng.
    Mỗi hàng của ma trận là một vector 512 chiều của 1 ảnh.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    image_paths = []
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(folder_path, f))
        
        # Sắp xếp tên file để luôn đảm bảo thứ tự (ví dụ: 001.jpg -> 002.jpg)
        image_paths.sort()
    else:
        print(f"Folder {folder_path} không tồn tại!")
        return None

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong folder {folder_path}.")
        return None
    
    print(f"Tìm thấy {len(image_paths)} ảnh. Đang trích xuất đặc trưng...")
    
    batch_size = 32
    all_features = []
    
    # Ép hiệu năng (Batch Inference): Xử lý theo lô để tránh tràn RAM/VRAM
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Mở ảnh bằng PIL cho batch hiện tại
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # Tiền xử lý ảnh (resize, normalize...) theo chuẩn của CLIP
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # Trích xuất đặc trưng
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            # Xử lý tương thích với các phiên bản transformers khác nhau
            if isinstance(outputs, torch.Tensor):
                image_features = outputs
            elif hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                pool = outputs.pooler_output
                if hasattr(model, "visual_projection") and pool.shape[-1] == model.visual_projection.in_features:
                    image_features = model.visual_projection(pool)
                else:
                    image_features = pool
            else:
                pool = outputs[1]
                if hasattr(model, "visual_projection") and pool.shape[-1] == model.visual_projection.in_features:
                    image_features = model.visual_projection(pool)
                else:
                    image_features = pool
                
        # Tùy chọn: Chuẩn hóa L2 cho các vector (thường dùng khi tính cosine similarity)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Đưa về CPU và đưa vào danh sách tổng
        all_features.append(image_features.cpu())
        
        # In tiến độ chạy
        print(f"Đã xử lý {min(i + batch_size, len(image_paths))}/{len(image_paths)} ảnh...")
        
    # Gộp tất cả các vector của các lô lại thành một ma trận duy nhất (N, 512)
    final_features = torch.cat(all_features, dim=0)
    
    return final_features, image_paths

if __name__ == "__main__":
    # Lấy thư mục chứa script hiện tại
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev1_dir = os.path.join(script_dir, "..", "..", "docs", "Phase 1 - Known-Item Search (KIS)", "dev1 - Computer Vision Engineer")
    
    # Tạo thư mục con riêng biệt để chứa các file .npy
    npy_dir = os.path.join(dev1_dir, "npy_features")
    os.makedirs(npy_dir, exist_ok=True)
    
    # Thư mục gốc chứa toàn bộ các folder ảnh (Keyframes)
    keyframe_dir = os.path.join(dev1_dir, "keyframe")
    if not os.path.exists(keyframe_dir):
        print(f"Không tìm thấy thư mục: {keyframe_dir}")
        print("Vui lòng tạo thư mục 'keyframe' và chuyển các folder ảnh vào đó.")
        sys.exit(0)
    
    # Tìm tất cả các thư mục con có tên bắt đầu bằng "L" (ví dụ: L25_V001) trong thư mục keyframe
    video_folders = []
    for d in os.listdir(keyframe_dir):
        full_path = os.path.join(keyframe_dir, d)
        # Chỉ lấy thư mục, bắt đầu bằng L
        if os.path.isdir(full_path) and d.startswith("L"):
            video_folders.append(d)
            
    video_folders.sort()
    
    if not video_folders:
        print("Không tìm thấy thư mục video nào bắt đầu bằng 'L' trong thư mục 'keyframe'!")
    else:
        print(f"Tìm thấy {len(video_folders)} thư mục video. Bắt đầu xử lý hàng loạt...")
        
        for folder_name in video_folders:
            print(f"\n{'='*50}")
            print(f"ĐANG XỬ LÝ THƯ MỤC: {folder_name}")
            folder_path = os.path.join(keyframe_dir, folder_name)
            
            # Cập nhật đường dẫn lưu vào thư mục npy_features
            npy_filename = os.path.join(npy_dir, f"{folder_name}.npy")
            
            # Kiểm tra xem file .npy đã có chưa, nếu có thì skip để tiết kiệm thời gian
            if os.path.exists(npy_filename):
                print(f"File '{folder_name}.npy' đã tồn tại trong thư mục 'npy_features'. Bỏ qua.")
                continue
                
            result = embed_folder(folder_path)
            
            if result is not None:
                features, paths = result
                print(f"Kích thước ma trận đặc trưng: {features.shape}")
                
                # Lưu file .npy
                np.save(npy_filename, features.numpy())
                print(f"Đã lưu ma trận đặc trưng vào: {npy_filename}")
