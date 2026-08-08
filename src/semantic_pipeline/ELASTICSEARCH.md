# Dev 2 - Local Elasticsearch

This setup is for local development of Task 4. Security is disabled and port
9200 is bound only to `127.0.0.1`; do not reuse this Compose file as a public
production deployment.

## 1. Prerequisites

- Install Docker Desktop for Windows and start it.
- Activate the project environment: `conda activate aichcm2026`.
- Install/update Python dependencies: `python -m pip install -r requirements.txt`.

Verify Docker:

```powershell
docker --version
docker compose version
```

## 2. Start Elasticsearch

From the repository root:

```powershell
docker compose -f src/semantic_pipeline/docker-compose.elasticsearch.yml up -d
docker compose -f src/semantic_pipeline/docker-compose.elasticsearch.yml ps
```

Wait until the service is `healthy`, then verify the HTTP endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:9200
```

## 3. Create, ingest and activate

The current Task 4 bootstrap performs three safe steps:

1. Create the versioned physical index `semantic_frames_v6` if absent.
2. Validate and bulk-index the full entity + spatial metadata using `frame_id`
   as Elasticsearch `_id`.
3. Atomically point the stable alias `semantic_frames` at that index.

Version v5 adds deterministic SQL metadata and code filters. See
`CODE_CLASSIFICATION.md` for the schema, provenance, and standalone command.

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v6 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json
python src/semantic_pipeline/elasticsearch_backend.py health
```

Mutating commands require an explicit versioned `--index-name`. This prevents
an old default from accidentally moving alias `semantic_frames` backwards.

Re-running bootstrap is idempotent for the same 24 frames. The command never
deletes an index or the source `metadata.json`.

Inspect the analyzer:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py analyze "men dropping motorbikes"
```

Run a search:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py search booking entity record --top-k 5

python src/semantic_pipeline/elasticsearch_backend.py `
  search "sql query" --top-k 5
```

## 4. Benchmark against BM25

Use the current alias and keep the filter-aware report beside the Task 4 tools:

```powershell
python src/semantic_pipeline/benchmark_elasticsearch.py `
  --index semantic_frames `
  --latency-runs 20 `
  --output src/semantic_pipeline/baseline_report_elasticsearch_v5_filters.json
```

Compare Recall/MRR/NDCG with `baseline_report_bm25_v0.json`. The evaluation
file currently contains both text-only cases and metadata/spatial-filter cases.

For an isolated 10k-document infrastructure pilot that never changes the stable
alias:

```powershell
python src/semantic_pipeline/benchmark_scale.py `
  --index-name semantic_frames_scale_10k_v2 `
  --documents 10000 `
  --warmup-runs 3 `
  --latency-runs 20 `
  --output src/semantic_pipeline/scale_benchmark_report_10k_v2.json
```

Use `--skip-ingest` on subsequent measurements. Synthetic replication measures
throughput/latency only; it is not a retrieval-quality benchmark.

## 5. Run the API with Elasticsearch

Only the backend changes; `POST /internal/search/text` keeps the existing API
response contract. The request additionally accepts an optional `filters`
object when Elasticsearch is enabled:

```json
{
  "keywords": ["image"],
  "top_k": 5,
  "filters": {
    "setting": "outdoor",
    "locations": ["field"],
    "objects": ["flowers"],
    "colors": ["blue"],
    "code_language": "sql",
    "code_patterns": ["not exists", "correlated"]
  }
}
```

Supported fields are `time_of_day`, `setting`, `locations`, `objects`,
`actions`, `colors`, `code_language`, `code_patterns`, and `spatial_relations`.
A spatial relation is a complete
triple such as `{"subject":"person","predicate":"left_of","object":"car"}`.
Requests without `filters` remain backward-compatible. BM25 returns HTTP 400
for a filter request instead of silently ignoring it.

Lists use **all-of (AND)** semantics. For example, `"objects":["person","car"]`
requires both object labels in the same frame; it does not mean either one.
The same rule applies to `code_patterns`.

```powershell
$env:SEMANTIC_SEARCH_BACKEND = "elasticsearch"
$env:ELASTICSEARCH_URL = "http://127.0.0.1:9200"
$env:ELASTICSEARCH_INDEX = "semantic_frames"
python src/semantic_pipeline/server.py
```

Unset the variables or set `SEMANTIC_SEARCH_BACKEND=bm25` to roll back.

For authentication plus HTTPS/transport TLS, use the isolated secure profile in
`ELASTICSEARCH_SECURE.md`. It uses a separate project, port and named volumes,
and never commits the password or CA private key.

## 6. Tests

Unit tests do not require Docker:

```powershell
pytest src/semantic_pipeline/tests/test_elasticsearch_backend.py -v
```

After bootstrap, enable the read-only live integration test:

```powershell
$env:RUN_ELASTICSEARCH_INTEGRATION = "1"
pytest src/semantic_pipeline/tests/test_elasticsearch_backend.py -v
```

## 7. Stop without deleting data

```powershell
docker compose -f src/semantic_pipeline/docker-compose.elasticsearch.yml down
```

The named Docker volume is retained, so indexed data survives the restart.
