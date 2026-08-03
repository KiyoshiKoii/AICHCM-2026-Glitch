# Tasks for Dev 1 (Computer Vision Engineer) - Phase 2

Trong Phase 2, Dev 1 sẽ tập trung vào việc tối ưu hóa cơ sở dữ liệu Vector (Qdrant) để chịu tải 100GB dữ liệu và cải thiện khả năng mở rộng (Scale).

- [ ] **Task 1: Tối ưu hóa Qdrant Scale**
  - Đóng gói Qdrant vào Docker Compose thay vì chạy local memory/disk thuần.
  - Tối ưu HNSW Index parameters (m, ef_construct) cho tập dữ liệu lớn để cân bằng giữa tốc độ và độ chính xác.
- [ ] **Task 2: Fusion Models (Tùy chọn)**
  - Thử nghiệm các mô hình Embedding đa phương thức mạnh hơn CLIP (ví dụ: BLIP-2, SigLIP) nếu CLIP không đủ độ phân giải để nhận diện vật thể nhỏ.
- [ ] **Task 3: Nâng cấp Temporal NMS**
  - Viết thuật toán gom nhóm (Clustering) các khung hình gần nhau hiệu quả hơn so với thuật toán Greedy NMS hiện tại.
