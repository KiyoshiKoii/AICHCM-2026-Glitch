# Video-understanding pilot

The pilot is intentionally restricted to `L22_V001`, using Gemini keyframe
metadata, timestamped ASR, and the BTC keyframe map. Source artifacts are
read-only. The generated pilot is fixed at:

```text
data/processed/video_understanding/L22/L22_V001/pilot/
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
data/metadata/caption/L22/L22_V001.json
data/metadata/metadata_asr/L22_V001.json
data/map-keyframes/L22_V001.csv
data/keyframes/L22_V001/
```

`data/map-keyframes/L22_V001.csv` is the identity bridge. `n` joins Gemini's
ordinal keyframe ID, `pts_time` is used for temporal alignment, and
`frame_idx` is the native BTC frame index.

## Run offline validation

```powershell
python src/semantic_pipeline/video_understanding/cli.py --video-id L22_V001
```

The offline mode creates evidence-linked deterministic fallback summaries.
The output never copies the source caption records or full ASR transcript.

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

## Tests

```powershell
python -m pytest -q src/semantic_pipeline/tests/test_video_understanding_l22.py
```

The real-data integration test processes only `L22_V001`, checks all 298 frame
map links and all 290 ASR segments, verifies the fixed three-file publish, and
checks that source hashes remain unchanged.
