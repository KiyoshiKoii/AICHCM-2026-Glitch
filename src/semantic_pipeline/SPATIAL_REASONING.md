# Dev 2 - Spatial Reasoning

This stage runs Florence-2 `<OD>` on each frame, converts pixel bounding boxes
to normalised `[x1, y1, x2, y2]`, and derives conservative spatial relations.
It reads enriched metadata and always writes a different output file.

## 1. Check metadata/image matching

This check does not load Florence:

```powershell
python src/semantic_pipeline/spatial_extractor.py --check-input
```

Check a particular frame:

```powershell
python src/semantic_pipeline/spatial_extractor.py `
  --check-input `
  --frame-id vid01_f0007
```

## 2. Smoke-test one or more frames

```powershell
python src/semantic_pipeline/spatial_extractor.py `
  --frame-id vid01_f0007 `
  --frame-id vid02_f0007 `
  --output src/semantic_pipeline/sample_frames/metadata_spatial_test.json `
  --checkpoint-every 1 `
  --force
```

The output still contains all metadata records, but only selected frames are
processed. This preserves frame IDs and makes later resume/validation safe.

## 3. Validate the result

```powershell
python src/semantic_pipeline/schemas.py `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial_test.json
```

Each detection contains a deterministic object ID, lowercase label, normalised
box, and confidence. Each relation stores both IDs and labels so Elasticsearch
can query a complete triple inside one nested document.

Contextual grounding corrects only a short, versioned list of detector
confusions when caption/OCR/entity evidence is present. It never uses frame IDs
or benchmark gold. A correction persists `raw_label`, `label_source`,
`label_evidence`, and `label_grounding_version` for auditability.

## 4. Relation rules

- `left_of` / `right_of`: boxes are separated horizontally and overlap enough
  on the vertical axis.
- `above` / `below`: boxes are separated vertically and overlap enough on the
  horizontal axis.
- `overlapping`: intersection covers at least 20% of the smaller box.
- Directional and overlapping relations are reciprocal.
- Diagonal boxes with no axis projection are ignored to reduce noisy claims.

Thresholds can be tuned with `--min-axis-gap`, `--min-axis-overlap`, and
`--min-overlap-ratio`. Keep one frozen rule version for an evaluation run.

Florence's standard `<OD>` decode does not return calibrated per-box scores.
The pipeline therefore writes an explicit proxy confidence of `0.5`; do not
interpret it as a calibrated probability. Change it only with
`--detection-confidence` and record the experiment.

## 5. Process all frames

CPU inference is supported but can be slow. GPU is selected automatically when
the installed PyTorch build exposes CUDA.

```powershell
python src/semantic_pipeline/spatial_extractor.py `
  --output src/semantic_pipeline/sample_frames/metadata_spatial.json `
  --checkpoint-every 5
```

Resume an interrupted run:

```powershell
python src/semantic_pipeline/spatial_extractor.py --resume
```

Use `--force` to restart from the input. Use `--overwrite-existing` only when
intentionally recomputing records already marked with a detection model.

After changing only grounding/rule code, reuse existing detector boxes without
loading Florence again:

```powershell
python src/semantic_pipeline/spatial_extractor.py `
  --reground-existing `
  --input src/semantic_pipeline/sample_frames/metadata_spatial.json `
  --output src/semantic_pipeline/sample_frames/metadata_spatial.json
```

## 6. Create a new Elasticsearch physical index

The mapping gained spatial labels and provenance fields, so do not re-use the
old v2 physical index. After the full run succeeds:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py `
  --index-name semantic_frames_v4 `
  bootstrap `
  --metadata src/semantic_pipeline/sample_frames/metadata_spatial.json
```

Bootstrap changes the stable alias only after all documents validate and ingest.
Raw metadata keeps all instance relations for evaluation. The indexed projection
keeps one highest-confidence relation per label triple because duplicate object
instances are query-equivalent for the current API.

Test a spatial triple from the CLI:

```powershell
python src/semantic_pipeline/elasticsearch_backend.py search domino pieces `
  --spatial domino left_of domino `
  --top-k 5
```

Equivalent API filter:

```json
{
  "keywords": ["domino", "pieces"],
  "top_k": 5,
  "filters": {
    "spatial_relations": [
      {"subject": "domino", "predicate": "left_of", "object": "domino"}
    ]
  }
}
```

## 7. Tests

```powershell
pytest src/semantic_pipeline/tests/test_spatial_extractor.py -v
pytest src/semantic_pipeline/tests/test_spatial_benchmark.py -v
pytest src/semantic_pipeline/tests/test_elasticsearch_backend.py -v
```

Run the manual benchmark:

```powershell
python src/semantic_pipeline/benchmark_spatial.py
```

The unit tests use a fake detector and never load Florence.

## References

- [Official Florence-2 documentation](https://huggingface.co/docs/transformers/model_doc/florence2)
- [Microsoft Florence-2 model card](https://huggingface.co/microsoft/Florence-2-large)
- [Florence-2 CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)
