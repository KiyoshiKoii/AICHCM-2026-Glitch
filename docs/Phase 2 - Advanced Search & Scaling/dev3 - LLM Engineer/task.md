# Tasks for Dev 3 (LLM Engineer & Backend Integration) - Phase 2

Trong Phase 2, Dev 3 đóng vai trò là "Tổng tư lệnh" đứng giữa, chịu trách nhiệm kết hợp LLM thế hệ mới và đảm bảo hệ thống chịu được lượng request dồn dập lúc thi.

- [ ] **Task 1: Nâng cấp Prompt LLM cho Spatial Reasoning**
  - Viết lại Prompt cho LLM (Llama-3/OpenAI) để nó tự động parse được các truy vấn không gian phức tạp thành chuẩn JSON Filters (VD: "Tìm người mặc áo đỏ đứng bên trái cái xe" $\rightarrow$ Sinh ra filter `spatial_relations: [{subject: "person red shirt", predicate: "left_of", object: "car"}]`).
- [ ] **Task 2: Tích hợp Redis Caching**
  - Lúc thi đấu, các đội hay search lại các câu query giống nhau. Cần setup Redis để cache kết quả API, giảm tải cho Qdrant và Elasticsearch.
- [ ] **Task 3: Tối ưu hóa RRF (Reciprocal Rank Fusion)**
  - Thay vì RRF trọng số tĩnh `1/(k+60)`, thử nghiệm trọng số động dựa trên độ tự tin (Confidence Score) của LLM về việc câu hỏi thiên về Hình ảnh (Visual) hay Văn bản (Semantic).
