# Stress-test tooling

This directory builds and evaluates a deterministic AIC 2026 batch-1 stress
corpus without downloading the complete multi-gigabyte ZIP archives.

## Build or resume

```powershell
python tools/stress_test/build_stress_corpus.py
```

The builder uses HTTP byte ranges and parallel curl transfers. Existing frame
and object files are reused, so the same command safely resumes an interrupted
run. Use `--fresh` only when intentionally rebuilding `data/stress_test_1200`.

## Validate

```powershell
python tools/stress_test/validate_corpus.py
```

Validation checks counts, unique IDs, file presence, image signatures,
SHA-256 hashes, object JSON parsing, query targets, and temporal ordering.

## Run against the backend

Start the four project services, then begin with a bounded sample:

```powershell
python tools/stress_test/run_backend_stress.py --type textual_kis --limit 100
python tools/stress_test/evaluate_predictions.py --only-predicted
```

Omit `--only-predicted` for a strict full-suite report where every missing query
counts as a miss.

The current backend does not generate Q&A answers, so Q&A predictions retain a
null answer and score zero until that capability is implemented. A complete run
can make thousands of requests and may consume paid API quota when Gemini
re-ranking is enabled.

## Ground-truth warning

The queries use pseudo ground truth derived from the supplied Faster R-CNN
objects, video metadata, keyframe maps, and temporally spaced frames. They are
for regression, throughput, failure-mode, and ranking stress tests. They are
not manually annotated and must not be reported as official competition model
accuracy.
