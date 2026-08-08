# Dev 2 - SQL/Code Classification

## Vì sao OCR đọc được chữ nhưng trước đây không trả nhãn SQL?

Pipeline có ba tầng khác nhau:

1. Florence-2 tạo caption; PaddleOCR + VietOCR đọc ký tự; Helsinki-NLP dịch
   phần tiếng Việt.
2. Ollama `llama3.2:3b` trích xuất entity theo schema gồm location, object,
   action, color. Schema entity cũ không có trường ngôn ngữ lập trình, nên model
   chỉ có thể trả object chung là `code`.
3. `code_classifier.py` mới phân loại ngữ nghĩa của OCR. Tầng này không chạy
   lại model ảnh và không cần Ollama.

Với `vid03_f0004`, OCR đã chứa `SELECT`, `FROM`, `WHERE`, `NOT EXISTS` và từ
`correlated`. Classifier lưu:

```json
{
  "language": "sql",
  "statement_type": "select",
  "patterns": ["not exists", "select", "from", "where", "exists", "correlated"],
  "search_terms": ["sql", "sql code", "select statement", "correlated subquery", "sql query", "sql query template"],
  "evidence": ["ocr_text:select", "ocr_text:not exists", "ocr_text_raw:correlated"],
  "classifier_version": "code-rules-v1"
}
```

Classifier dùng tín hiệu cú pháp, không dùng `frame_id`. Quy tắc template nhận
`SELECT ... FROM`, vì vậy có thể áp dụng cho frame/video mới có cùng kiểu nội
dung. `SELECT ... FROM` còn phải có thêm tín hiệu SQL như `WHERE`, `JOIN`,
`GROUP BY`, `EXISTS`... để tránh nhận nhầm câu văn thông thường.

## Chạy classifier

Classifier có thể cập nhật atomically artifact đã có:

```powershell
python src/semantic_pipeline/code_classifier.py `
  --input src/semantic_pipeline/sample_frames/metadata_entities.json `
  --in-place

python src/semantic_pipeline/code_classifier.py `
  --input src/semantic_pipeline/sample_frames/metadata_spatial.json `
  --in-place
```

Kết quả hiện tại trên sample: `24` record được phân loại, gồm `7` SQL và `17`
unknown. `entity_extractor.py` cũng tự chạy classifier cho mọi record, kể cả
record bỏ qua Ollama khi resume. BM25 và Elasticsearch tự phân loại ở biên nạp
dữ liệu để metadata legacy không âm thầm mất nhãn SQL.

## Nạp Elasticsearch v5

Mapping v4 không có object `code`, nên phải dùng physical index v5:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v6 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json
```

Kiểm tra tìm kiếm không filter:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  search "sql query" --top-k 5
```

Kỳ vọng `vid03_f0004` ở rank 1. Kiểm tra exact filter:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  search "sql query" "correlated subquery" `
  --top-k 5 `
  --code-language sql `
  --code-pattern "not exists" `
  --code-pattern correlated
```

API nhận hai filter tương ứng là `code_language` và `code_patterns`. Danh sách
pattern dùng all-of/AND giống các list filter khác.

## Test offline

```powershell
python -m pytest -q `
  src/semantic_pipeline/tests/test_code_classifier.py `
  src/semantic_pipeline/tests/test_database.py `
  src/semantic_pipeline/tests/test_elasticsearch_backend.py `
  src/semantic_pipeline/tests/test_api.py
```

## Hạn chế

- Version v1 mới phân loại SQL; Python/JavaScript và DDL hiếm cần thêm fixture
  trước khi mở rộng taxonomy.
- Classifier phụ thuộc OCR có giữ được cú pháp quan trọng hay không.
- Rules và benchmark hiện mới được kiểm tra trên sample 24 frame; cần holdout
  từ video thật để đo precision/recall độc lập.
- Query chung `sql` có thể trả nhiều frame SQL. KIS nên dùng thêm pattern hoặc
  mô tả nội dung khi muốn một frame cụ thể.
