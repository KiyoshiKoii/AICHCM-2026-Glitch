# Gemini visual metadata pipeline

This directory contains only the active visual-metadata path. Gemini analyzes
frames and produces the compact artifact consumed by the
retrieval system.  The former local Florence detector, KIS, and Elasticsearch
experiments have been removed.

| Module | Responsibility |
| --- | --- |
| `gemini/` | Gemini extraction (`extractor.py`) and selective metadata repair (`repair.py`) |
| `quality/` | Local CLIP quality audit for swapped or weak image-caption pairs |
| `core/` | Shared schema, frame IDs, deduplication, program profiles, JSON I/O, rate limits, and spatial reasoning |
| Root CLI files | Compatibility wrappers so existing `python src/semantic_pipeline/<command>.py` commands keep working |

`core/visual_profiles.py` provides the L21-L30 program context derived from
YouTube metadata. The compact persisted schema lives in
`core/compact_metadata.py`.

Model selection is centralized in the repository `.env`: `GEMINI_VISUAL_MODEL`
for extraction and repair, and `CAPTION_AUDIT_CLIP_MODEL` for the local audit.
The CLI reads that file directly, so normal runs do not need `--model`.

Run a small extraction with the default, quality-first 20-frame request size:

```powershell
python src/semantic_pipeline/gemini_visual_extractor.py --input-dir data/keyframes/L21_V001 --output-dir data/metadata/caption --limit 20
```

Run the unit tests:

```powershell
pytest src/semantic_pipeline/tests -q
```

## Elasticsearch text retrieval

`retrieval/` provides the internal semantic service on port 8002. It indexes
the English Gemini captions separately from multilingual OCR/ticker and
program metadata, then uses weighted BM25 clauses plus exact-phrase boosts.
The main backend keeps fusing these lexical results with CLIP through RRF, so
this service does not duplicate the visual embedding pipeline.
The endpoint also accepts optional `batch_ids` (`L21`–`L30`) and `video_ids`
filters. Full IDs such as `L22_V030` and suffixes such as `V030` are supported;
the main backend forwards these constraints to both Elasticsearch and CLIP
retrieval.

Start a local Elasticsearch node, create a versioned index, and atomically
activate its stable alias:

```powershell
docker compose -f src/semantic_pipeline/retrieval/docker-compose.elasticsearch.yml up -d
python src/semantic_pipeline/elasticsearch_service.py bootstrap
```

Start the internal semantic API expected by `src/backend`:

```powershell
uvicorn semantic_pipeline.retrieval.api:app --app-dir src --host 127.0.0.1 --port 8002
```

The indexer reads every `metadata/caption/LXX/LXX_VYYY.json` checkpoint and
joins compact title, description, keywords, and program fields from
`data/metadata/metadata_youtube.jsonl`. Re-run `bootstrap` after new Gemini
caption artifacts are available; `frame_id` is the deterministic Elasticsearch
document ID, so re-indexing is idempotent. Inspect a query directly with:

```powershell
python src/semantic_pipeline/elasticsearch_service.py search "red umbrella" --top-k 10
```

For the current pilot, use only the reviewed L21_V001 artifact and a disposable
physical index. This does not generate metadata and does not change the stable
production alias:

```powershell
python -m semantic_pipeline.retrieval.cli `
  --index-name semantic_frames_l21_v001_pilot `
  ingest `
  --caption-dir data/metadata/caption `
  --require-provenance
```

The v4 mapping stores visual/OCR provenance, quality flags, and queryable nested
spatial relations. Pilot search can collapse visual duplicates by
`visual_source_frame_id` with `--collapse-visual-duplicates` while retaining
each frame's OCR/ticker. The flag stays opt-in until the production alias is
migrated to v4.

The CLI supports up to 100 frames per request for controlled experiments, but
the current default is 20 to reduce cross-frame hallucination. For globally
deduplicated videos, representatives receive the full visual extraction;
duplicate frames receive an OCR-only pass so their captions and detections stay
shared while frame-specific on-screen text remains accurate.

Each video is checkpointed independently, for example:
`data/metadata/caption/L21/L21_V001.json`. Use `--resume` to continue an
existing video artifact, or `--force` to regenerate the selected videos.

Before calling Gemini, the extractor reads `data/global_filter_results.csv`.
Only its representative frames receive full visual extraction. Near-duplicate
frames reuse the representative's visual metadata, then receive an OCR-only
request for their own `ocr_text` and `news_ticker_text`.
The supplied map covers all keyframes and currently avoids 51,842 of 177,321
Gemini frame inputs. Pass `--without-global-filter` only for an A/B run.

The default scheduler starts at most 15 requests in a 60-second window and
runs those requests in parallel. Daily quota is intentionally not tracked
locally; check the provider dashboard manually. The legacy daily-budget CLI
arguments are accepted for command compatibility but do not create a state
file or stop extraction. Temporary Gemini `503 UNAVAILABLE`/service-overload
responses are retried in a later request window up to five times before the
batch is reported as failed.

## Post-run L21 caption audit

After the L21 Gemini extraction has completely finished, run the local CLIP
verifier. It compares each image with all captions in its 20-frame batch and
writes a triage report; it does not call Gemini:

```powershell
python src/semantic_pipeline/verify_caption_alignment.py `
  --video-prefix L21 `
  --video-id L21_V001 `
  --caption-dir data/metadata/caption `
  --keyframe-dir data/keyframes `
  --batch-size 20 `
  --output data/metadata/verification/clip_L21_report.json
```

Review flagged rows in `clip_L21_report.json`. A flag means the claimed
image-caption score is low or another image in the same batch scores higher;
it is a review signal, not an automatic rewrite.

## Repair flagged frames

After reviewing a CLIP report, re-query only its flagged frames. This bypasses
the global deduplication map and overwrites only the selected frame records;
it never uses `--force` or replaces the rest of a video file. Run a dry-run
first, then remove `--dry-run` to call Gemini:

```powershell
python src/semantic_pipeline/repair_flagged_metadata.py `
  --report data/metadata/verification/clip_L21_full_report.json `
  --keyframe-dir data/keyframes `
  --output-dir data/metadata/caption `
  --batch-size 20 `
  --requests-per-minute 15 `
  --max-concurrent-requests 15 `
  --daily-request-limit 500 `
  --only-copied-frames `
  --dry-run
```

For the full L21 report, this intersection currently selects 196 frames in 29
video-local requests. For a smaller high-confidence subset, add both
`--max-claimed-score 0.20` and `--min-alternate-gap 0.05`; a frame is included
if it meets either severity condition. Remove `--dry-run` only after reviewing
the candidate count.
