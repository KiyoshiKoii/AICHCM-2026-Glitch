import os
import glob
import cv2
import argparse
import csv
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix encoding issue for Vietnamese characters in Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def calculate_histogram(image_path):
    """Đọc ảnh và tính toán color histogram trong không gian HSV."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return hist

def compare_histograms(hist1, hist2):
    """So sánh 2 histogram bằng phương pháp Correlation (0 đến 1)."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def process_single_video(video_dir, threshold):
    """Xử lý lọc trùng lặp cho một video duy nhất."""
    video_name = video_dir.name
    extensions = ['*.jpg', '*.png', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(video_dir.glob(ext))
    
    image_paths.sort()
    if not image_paths:
        return [], 0, 0, 0
        
    total_frames = len(image_paths)
    last_kept_hist = None
    last_kept_path = None
    
    kept_count = 0
    dup_count = 0
    results = []

    for img_path in image_paths:
        current_hist = calculate_histogram(img_path)
        
        if current_hist is None:
            continue

        if last_kept_hist is None:
            last_kept_hist = current_hist
            last_kept_path = img_path
            kept_count += 1
            results.append({
                "video_name": video_name,
                "frame_name": img_path.name,
                "is_representative": True,
                "representative_frame": img_path.name,
                "similarity": 1.0
            })
            continue
        
        similarity = compare_histograms(current_hist, last_kept_hist)
        
        if similarity > threshold:
            dup_count += 1
            results.append({
                "video_name": video_name,
                "frame_name": img_path.name,
                "is_representative": False,
                "representative_frame": last_kept_path.name,
                "similarity": round(float(similarity), 4)
            })
        else:
            last_kept_hist = current_hist
            last_kept_path = img_path
            kept_count += 1
            results.append({
                "video_name": video_name,
                "frame_name": img_path.name,
                "is_representative": True,
                "representative_frame": img_path.name,
                "similarity": 1.0
            })
            
    return results, kept_count, dup_count, total_frames

def filter_duplicates(input_dir, output_csv=None, threshold=0.90):
    """
    Lọc các frame trùng lặp (chạy đa luồng, hỗ trợ resume) và ghi kết quả ra file CSV.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Thư mục không tồn tại: {input_dir}")
        return

    if output_csv is None:
        output_csv = input_path / "global_filter_results.csv"
    else:
        output_csv = Path(output_csv)

    video_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not video_dirs:
        video_dirs = [input_path]
    else:
        video_dirs.sort()

    # Tính năng Resume: Đọc các video đã xử lý từ file CSV nếu có
    processed_videos = set()
    file_exists = output_csv.exists()
    
    if file_exists:
        try:
            with open(output_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_videos.add(row['video_name'])
            print(f"Đã tìm thấy file CSV cũ. Sẽ bỏ qua {len(processed_videos)} videos đã xử lý thành công.")
        except Exception as e:
            print(f"Lỗi đọc file CSV cũ: {e}. Sẽ chạy lại từ đầu.")
            processed_videos = set()
            file_exists = False
            
    # Lọc lại danh sách video cần xử lý
    video_dirs_to_process = [vd for vd in video_dirs if vd.name not in processed_videos]
    
    if not video_dirs_to_process:
        print("Tất cả videos đều đã được xử lý xong! Không cần làm gì thêm.")
        return

    print(f"Đang tiến hành lọc (Đa luồng) cho {len(video_dirs_to_process)} videos còn lại với threshold={threshold}...")
    
    total_kept = 0
    total_dup = 0
    total_frames = 0
    
    max_workers = os.cpu_count() or 4
    print(f"Sử dụng {max_workers} luồng xử lý đồng thời...")
    
    # Mở file CSV ở chế độ append ('a'), viết header nếu file chưa tồn tại
    mode = 'a' if file_exists else 'w'
    with open(output_csv, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "frame_name", "is_representative", "representative_frame", "similarity"])
        if not file_exists:
            writer.writeheader()
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tất cả các tasks
            futures = {executor.submit(process_single_video, vd, threshold): vd for vd in video_dirs_to_process}
            
            # Lấy kết quả ngay khi xong và ghi luôn vào file để lưu trạng thái resume
            for future in tqdm(as_completed(futures), total=len(video_dirs_to_process), desc="Xử lý videos"):
                results, kept, dup, frames = future.result()
                
                if results:
                    writer.writerows(results)
                    f.flush() # Bắt buộc flush để ghi thẳng xuống ổ đĩa, đảm bảo ko bị mất data nếu dừng đột ngột
                
                total_kept += kept
                total_dup += dup
                total_frames += frames

    print("\n--- TỔNG KẾT PHIÊN CHẠY NÀY ---")
    print(f"Số videos đã xử lý thêm: {len(video_dirs_to_process)}")
    print(f"Tổng số frames ban đầu: {total_frames}")
    print(f"Giữ lại (representative): {total_kept} frames")
    print(f"Loại bỏ (trùng lặp): {total_dup} frames")
    print(f"Tiến độ đã được cộng dồn vào: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc frame gần trùng và xuất kết quả ra CSV")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Thư mục chứa các frames")
    parser.add_argument("-o", "--output_csv", type=str, default=None, help="Đường dẫn file CSV xuất ra (mặc định: input_dir/filter_results.csv)")
    parser.add_argument("-t", "--threshold", type=float, default=0.90, help="Ngưỡng tương đồng (0.0 đến 1.0, mặc định: 0.90)")
    
    args = parser.parse_args()
    
    filter_duplicates(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        threshold=args.threshold
    )
