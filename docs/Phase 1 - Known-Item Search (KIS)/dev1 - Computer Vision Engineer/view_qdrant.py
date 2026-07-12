import os
import sys
from qdrant_client import QdrantClient

# Fix encoding cho in tiếng Việt trên Terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "local_qdrant_db")

print(f"Đang kết nối tới Qdrant DB tại: {db_path}\n")
client = QdrantClient(path=db_path)

collection_name = "kis_images"

if client.collection_exists(collection_name=collection_name):
    # Lấy thông tin tổng quan
    info = client.get_collection(collection_name=collection_name)
    print(f"--- THÔNG TIN TRẠNG THÁI ---")
    print(f"- Tên Collection: {collection_name}")
    print(f"- Số lượng dữ liệu đang lưu: {info.points_count} vectors")
    
    # Kéo thử 3 vector đầu tiên ra để xem tận mắt
    print("\n--- XEM THỬ 3 DÒNG DỮ LIỆU ĐẦU TIÊN ---")
    results, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True, # Quan trọng: lấy Metadata (tên file)
        with_vectors=True  # Lấy dữ liệu Vector
    )
    
    for point in results:
        print(f"\n> ID Của Vector: {point.id}")
        print(f"  Payload (Metadata): {point.payload}")
        # Rút gọn vector lại chỉ in 5 số đầu cho dễ nhìn
        truncated_vector = [round(v, 4) for v in point.vector[:5]]
        print(f"  Vector: {truncated_vector} ... (tổng cộng {len(point.vector)} chiều)")
        
else:
    print(f"Collection '{collection_name}' không tồn tại!")
