import sys
import os
import subprocess

# Fix encoding cho in tiếng Việt trên Terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG ẢNH (DÙNG MÔ HÌNH AI)")
print("="*60)
embed_script = os.path.join(script_dir, "embed_folder.py")
# Dùng sys.executable để đảm bảo nó gọi đúng bản Python (Miniconda) đang chạy
subprocess.run([sys.executable, embed_script], check=True)

print("\n" + "="*60)
print("BƯỚC 2: NẠP DỮ LIỆU VÀO DATABASE (QDRANT)")
print("="*60)
qdrant_script = os.path.join(script_dir, "setup_qdrant.py")
subprocess.run([sys.executable, qdrant_script], check=True)

print("\n" + "="*60)
print(" HOÀN TẤT! TOÀN BỘ DỮ LIỆU ĐÃ ĐƯỢC NẠP VÀO KHO SẴN SÀNG.")
print("="*60)
