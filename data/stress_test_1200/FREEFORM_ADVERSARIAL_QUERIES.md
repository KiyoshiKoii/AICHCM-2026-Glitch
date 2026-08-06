# Free-form adversarial query pack

This pack extends the deterministic AIC stress corpus with individually written
queries. The prompts deliberately avoid a single sentence template while still
targeting only frames already present in `manifest.jsonl` and `frames/`.

The original generated query files remain unchanged so baseline results stay
comparable.

## Files

| File | Queries | Coverage |
|---|---:|---|
| `queries/freeform_textual_kis.jsonl` | 60 | Six grounded frames from each of L21-L30 |
| `queries/freeform_qa.jsonl` | 10 | One detector-count or count-arithmetic case per dataset |
| `queries/freeform_trake.jsonl` | 10 | One 3-5 event sequence per dataset |

## What this pack stresses

- Vietnamese, English, and code-switched requests.
- Natural questions, terse keyword fragments, conversational wording, typos,
  emoji, indirect clues, and negative instructions.
- Dense object conjunctions, rare label combinations, scene-level descriptions,
  and foreground/background clues.
- Closely related frames from the same video, repeated objects at different
  moments, sparse detector output, and exact temporal ordering.
- Detector taxonomy overlap such as `bird` versus `duck`/`goose`, or `animal`
  versus `dog`/`carnivore`.
- Exact counts and small arithmetic questions grounded in `object_counts`.

Every entry contains its target video and official source-frame range. KIS and
Q&A cases also include the sampled keyframe index and source sample ID. TRAKE
cases contain every ordered source sample ID in their event records.

## Suggested runs

Start the project services, then run one pack at a time. `--limit 0` means all
queries in the selected file.

```powershell
python tools/stress_test/run_backend_stress.py `
  --queries data/stress_test_1200/queries/freeform_textual_kis.jsonl `
  --type textual_kis --limit 0 `
  --output data/stress_test_1200/predictions_freeform_kis.jsonl

python tools/stress_test/evaluate_predictions.py `
  --queries data/stress_test_1200/queries/freeform_textual_kis.jsonl `
  --predictions data/stress_test_1200/predictions_freeform_kis.jsonl
```

Replace the two query/prediction filenames with `freeform_qa.jsonl` or
`freeform_trake.jsonl` for the other task types. The current backend does not
generate natural-language Q&A answers, so the Q&A pack intentionally exposes
that gap until an answer-generation stage is added.

## Ground-truth limitation

These are engineering stress cases, not manual competition annotations. Object
descriptions and counts come from the supplied Faster R-CNN detections; temporal
targets come from the supplied frame map. A detector label can be noisy even
when its frame reference is valid. Use the pack for ranking, robustness,
latency, regression, and failure analysis rather than official accuracy claims.
