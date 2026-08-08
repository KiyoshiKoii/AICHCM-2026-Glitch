# BTC object integration (schema v1.1)

The semantic pipeline now treats BTC identifiers and `objects/` as canonical.
Caption/OCR remains optional and can be fused when keyframe images exist.

## Build flow

```powershell
python scripts/download_openimages_metadata.py
python scripts/build_objects_index.py --videos L21,L22 --resume
python -m src.semantic_pipeline.metadata_builder --videos L21,L22 --resume
python -m pytest src/semantic_pipeline/tests -q
```

For all attached videos, omit `--videos` from `build_objects_index.py`, then pass
all required BTC groups or video IDs to `metadata_builder.py`. Outputs are
written per video under `data/processed/` and are intentionally ignored by Git.

To merge Task 1 caption/OCR records, pass `--visual-metadata <file-or-directory>`.
Records without those signals remain searchable through object and media text;
`has_visual_text` tells downstream rank fusion which path was available.

## Identity rules

- `video_name`: BTC video ID without `.mp4`, e.g. `L21_V001`.
- `frame_index`: BTC `map-keyframes.frame_idx` (submission frame).
- `keyframe_n`: BTC `n`, used to open `001.jpg` and `001.json`.
- `frame_id`: `<video_name>_f<frame_index:04d>`.
- `timestamp_ms`: rounded from `pts_time`; it is never recomputed from FPS.

An implementation audit found repeated `frame_idx` values in 192 of the 873
attached map files (never decreasing). For example, the first two rows of
`L21_V006` both map to frame 0. Since AIC submission identity cannot distinguish
them, `canonical_keyframe_map()` keeps the first `n` for each `frame_idx`.
Object files are still joined by `n`; only the duplicate search document is
discarded. This prevents duplicate IDs and removes redundant output.

## Object filtering

BTC's 100 padded detections per frame are processed in this order:

1. score `>= 0.20`;
2. normalized area `>= 0.0001`;
3. class-wise NMS at IoU `0.60`;
4. top 15 by real detector score.

Boxes are converted from BTC `[ymin, xmin, ymax, xmax]` to normalized
`[x1, y1, x2, y2]`. Every kept object preserves its Open Images MID, score,
3x3 grid cell, and area. The 120-class EN/VI lexicon was selected from measured
L21/L22 frequency; Open Images ancestors are expanded only upward.

## Elasticsearch v6

```powershell
python -m src.semantic_pipeline.elasticsearch_backend `
  --index-name semantic_frames_v6 bootstrap `
  --metadata data/processed/metadata
```

The physical v6 index adds object/media fields, nested exact object filtering,
minimum object score, object counts, and Vietnamese ASCII folding. The stable
`semantic_frames` alias is moved atomically after ingest.

API filters remain optional and the four-field response contract is unchanged:

```json
{
  "keywords": ["three people near a car"],
  "filters": {
    "objects": ["car"],
    "min_object_score": 0.6,
    "object_counts": {"person": 3}
  }
}
```
