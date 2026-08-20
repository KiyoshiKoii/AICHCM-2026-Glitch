# Video-understanding pipeline

The pipeline accepts any BTC video ID in `Lxx_Vyyy` format, using Gemini
keyframe metadata, timestamped ASR, and the BTC keyframe map. Source artifacts
are read-only. Each video owns one fixed pilot directory:

```text
data/processed/video_understanding/<batch_id>/<video_id>/pilot/
├── timeline.json
├── video_summary.json
└── validation_report.json
```

The three files are derived outputs. A successful run may replace the existing
pilot after validation; a failed or lower-quality candidate leaves the current
pilot unchanged. Intermediate evidence packets are temporary and are removed
after the run.

## Sources

```text
data/metadata/caption/<batch_id>/<video_id>.json
data/metadata/metadata_asr/<video_id>.json
data/map-keyframes/<video_id>.csv
data/keyframes/<video_id>/
```

`data/map-keyframes/<video_id>.csv` is the identity bridge. `n` joins Gemini's
ordinal keyframe ID, `pts_time` is used for temporal alignment, and
`frame_idx` is the native BTC frame index.

## Run offline validation

```powershell
python src/semantic_pipeline/video_understanding/cli.py --video-id L22_V001
```

The offline mode creates evidence-linked deterministic fallback summaries.
The output never copies the source caption records or full ASR transcript.

Run every video discovered from caption artifacts in one batch:

```powershell
python src/semantic_pipeline/video_understanding/cli.py --batch-id L22
```

Run only an inclusive video-number range from one batch:

```powershell
python src/semantic_pipeline/video_understanding/cli.py `
  --batch-id L22 `
  --video-start 1 `
  --video-end 100
```

`--video-start` and `--video-end` must be used together with `--batch-id`.

## Run with Gemini summarization

The environment must provide `GEMINI_API_KEY`. The optional
`GEMINI_VIDEO_SUMMARY_MODEL` selects the model; otherwise the existing
`GEMINI_VISUAL_MODEL` value is used.

```powershell
python src/semantic_pipeline/video_understanding/cli.py `
  --video-id L22_V001 `
  --llm `
  --require-llm
```

LLM calls receive compact time windows containing caption/object/OCR evidence
and ASR segments. Every story reference is validated against the supplied
evidence before it can reach the fixed pilot output.

The Gemini response is constrained with separate JSON schemas for window events
and the final bilingual summary. Window titles/summaries are Vietnamese;
`summary_vi` and `summary_en` are generated as distinct language fields. The
pipeline also merges duplicate cards emitted by overlapping windows and removes
the generic broadcast opener (for example, “Chương trình tin tức 60 giây”).

Free-tier request limits are handled with a default 4.2-second interval between
requests and bounded retry backoff for HTTP 429 responses. Override the interval
when needed with `GEMINI_REQUEST_INTERVAL_SECONDS`.

## Hierarchical retrieval

`summary_vi` and `summary_en` are display-level summaries, not the only search
surface. `video_summary.search_text` contains every timeline title and story
summary. The retrieval layer builds three in-memory/Elasticsearch-ready document
types directly from the fixed pilot plus source metadata:

```text
video → candidate video
segment → candidate news story and its time range
frame → keyframe_n and BTC native_frame_idx
```

No additional JSON artifact is persisted. The document builder can emit bulk
actions for `semantic_videos_v1`, `semantic_segments_v1`, and
`semantic_frames_v5`; all can be rebuilt from the three pilot files and source
caption/ASR/map data.

Run a reproducible local query:

```powershell
python src/semantic_pipeline/retrieval/video_cli.py `
  "nhiệt độ Barcelona cao nhất trong 110 năm" `
  --top-segments 1 `
  --top-frames 1
```

The result includes the selected segment, `keyframe_n`, `native_frame_idx`, and
timestamp. Queries containing “đầu tiên” choose the earliest frame whose score
is close enough to the best evidence match.

## Tests

```powershell
python -m pytest -q src/semantic_pipeline/tests/test_video_understanding_l22.py
python -m pytest -q src/semantic_pipeline/tests/test_hierarchical_video_search.py
```

The real-data integration tests process `L22_V001` and `L21_V001`, verify that
batch/output/segment IDs are derived from each video ID, check the fixed
three-file publish, and confirm source hashes remain unchanged.
