# Temporal Event Search

Temporal Event Search is the fourth retrieval mode beside Text Search, VQA and
Image Search. It accepts a BTC prompt with shared video context and ordered
`E1:`, `E2:` ... event lines.

The mode first selects one video using the context summary and all event
descriptions, then retrieves every event from that same video in temporal
order. It returns `frame_id`, `keyframe_n`, `native_frame_idx`, `timestamp_ms`
and `thumbnail_url`.

Before retrieval, the service uses one Gemini query-parser call configured by
`GEMINI_TEMPORAL_QUERY_MODEL` (default: `gemini-3.5-flash-lite`). It converts
explicit query constraints into required concept groups while preserving every
user term. For example, `linh vật Olympic Paris` can add aliases such as
`mascot` and `Phryge`; a video that only mentions Olympic Paris cannot win
without also satisfying the mascot concept. If Gemini is unavailable, the
same AND-style constraints are retained deterministically from the original
terms.

No extra pilot artifact is created. Each video still owns exactly:

```text
data/processed/video_understanding/<batch>/<video_id>/pilot/
├── timeline.json
├── video_summary.json
└── validation_report.json
```

`timeline.json` v2 contains compact event records and temporal anchors. Raw
caption, ASR, prompts, Gemini responses and candidate windows are not copied
to the output. Event documents are built in memory for retrieval.

## Local CLI

```powershell
python src/semantic_pipeline/retrieval/temporal_cli.py `
  "News story about Barcelona temperature record`nE1: first visible Barcelona temperature record`nE2: first visible extreme heat" `
  --batch-id L22 `
  --top-k-videos 5
```

For deterministic debugging without the query parser, append
`--without-gemini-query-parser`.

## HTTP endpoints

```text
Dev2:    POST /internal/search/temporal-events
Backend: POST /api/v1/search/temporal-events
```

The UI exposes the mode through the `Temporal Events` tab. A result that has
not been visually verified keeps a conservative confidence and must not be
treated as a guaranteed BTC temporal boundary.
