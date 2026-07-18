# Dev 2 - Entity Extraction

This stage reads `caption` and translated `ocr_text`, asks a local Ollama model
for schema-constrained JSON, validates it with Pydantic, and writes a new
metadata file. The Task 1 source file is never overwritten.

## 1. Start Ollama and obtain a model

Install Ollama on Windows, start its service, then run:

```powershell
ollama pull llama3.2:3b
ollama list
```

The default endpoint is `http://127.0.0.1:11434`. If Dev 3 provides a shared
Ollama server, set `OLLAMA_URL` instead of running another local copy.

## 2. Verify the service/model

```powershell
python src/semantic_pipeline/entity_extractor.py --check
```

Override model or URL when necessary:

```powershell
$env:OLLAMA_URL = "http://127.0.0.1:11434"
$env:OLLAMA_ENTITY_MODEL = "llama3.2:3b"
python src/semantic_pipeline/entity_extractor.py --check
```

## 3. Test only two records first

```powershell
python src/semantic_pipeline/entity_extractor.py `
  --limit 2 `
  --output src/semantic_pipeline/sample_frames/metadata_entities_test.json
```

Inspect `entities` and `processing` in the output before processing everything.
The model is instructed to use `unknown`/empty arrays instead of inventing
unsupported facts. Prompt `entities-v1.3` also uses few-shot examples and a
deterministic evidence guard to remove unsupported locations, colors, and
document-only relations.

## 4. Process the full sample

```powershell
python src/semantic_pipeline/entity_extractor.py
```

Default output:

```text
src/semantic_pipeline/sample_frames/metadata_entities.json
```

If a long run is interrupted, continue from its checkpoint:

```powershell
python src/semantic_pipeline/entity_extractor.py --resume
```

Use `--force` to restart from the original input. Use `--overwrite-existing`
only when intentionally replacing entities already produced by a previous
model/prompt version.

Every record also receives deterministic `code` metadata from caption/OCR,
including records skipped during resume. This does not make another Ollama
call. See `CODE_CLASSIFICATION.md` for SQL rules and standalone reclassification.

## 5. Validate enriched metadata

```powershell
python src/semantic_pipeline/schemas.py `
  --metadata src/semantic_pipeline/sample_frames/metadata_entities.json
```

## 6. Ingest into a new Elasticsearch index version

The current full Task 4 mapping is v5 (entity + code + spatial metadata):

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v5 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json
```

Bootstrap activates alias `semantic_frames` only after successful validation
and ingest. Older physical indices remain intact as rollback points.

## 7. Tests

```powershell
pytest src/semantic_pipeline/tests/test_entity_extractor.py -v
```

## 8. Verify a metadata filter

After bootstrap, this query should return `vid02_f0001` only:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py search image `
  --object pot `
  --top-k 5
```

## 9. Visual-truth benchmark

`entity_visual_ground_truth.json` is manually annotated from frame pixels and
does not copy caption/OCR/model output. Run:

```powershell
python src/semantic_pipeline/benchmark_entities.py
python -m pytest -q src/semantic_pipeline/tests/test_entity_benchmark.py
```

The report separates searchable set fields from easy scalar `unknown` cases.
Current searchable-entity micro-F1 is `0.4194`; treat entity filters as R&D
until a held-out target-video benchmark reaches the production gate.
